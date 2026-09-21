import os
import re
import json
import zipfile
import hashlib
import time
from collections import Counter

import numpy as np
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── 설정 ─────────────────────────────────────────────────────────
PDF_PATH = r"C:\Users\201\Downloads\저작권법(법률)(제21336호)(20260811).pdf"
CACHE_DIR = "embedding_cache"
DB_DIR = "ai_agent_db"

os.makedirs(CACHE_DIR, exist_ok=True)

openai_client = OpenAI(api_key=os.getenv("api_key"))
def load_document_text(path: str) -> str:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if "manifest.json" in names:
                manifest = json.loads(z.read("manifest.json").decode("utf-8"))
                pages = sorted(manifest["pages"], key=lambda p: p["page_number"])
                return "\n".join(
                    z.read(p["text"]["path"]).decode("utf-8") for p in pages
                )
        raise ValueError(f"zip 파일이지만 예상한 manifest.json 구조가 아닙니다: {path}")


    import pypdf
    reader = pypdf.PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


total = load_document_text(PDF_PATH)

CHAPTER_RE = re.compile(r"^제\d+장(?:의\d+)?\s+\S") 


def split_parts(text: str) -> dict[str, str]:
    parts, current = {}, None
    for line in text.splitlines():
        s = line.strip()
        if CHAPTER_RE.match(s):
            current = re.sub(r"\s*<.*?>\s*$", "", s)
        elif s.startswith("부칙"):
            current = None
        elif current:
            parts.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in parts.items()}


parts = split_parts(total)
print("파트:", {k: f"{len(v):,}자" for k, v in parts.items()})

ARTICLE_RE = re.compile(r"^제(\d+(?:조의\d+|조))\(([^)]+)\)")
SECTION_RE = re.compile(r"^제\d+절\s+(.+)$")


def article_records(text: str, part_name: str) -> list[dict]:
    records, section = [], None
    art_no = art_title = None
    buf: list[str] = []

    def flush():
        if art_no:
            records.append({
                "part": part_name,
                "chapter": section or part_name,
                "article": f"제{art_no}({art_title})",
                "text": "\n".join(buf).strip(),
            })

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        sm = SECTION_RE.match(s)
        if sm and len(s) < 30:
            section = s
            continue
        m = ARTICLE_RE.match(s)
        if m:
            flush()
            art_no, art_title = m.group(1), m.group(2)
            buf = [s]
        elif art_no:
            buf.append(s)
    flush()
    return records


records = [r for name, t in parts.items() for r in article_records(t, name)]
docs = [f"[{r['part']}] {r['text']}" for r in records]
print("총 조문 수:", len(records))
print(Counter(r["part"] for r in records))


EMBED_MODEL = "text-embedding-3-small"


def embed_texts(texts: list[str], batch: int = 100) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=chunk)
        vectors.extend(item.embedding for item in resp.data)
        print(f"  임베딩 진행: {min(i + batch, len(texts))}/{len(texts)}")
    return np.array(vectors, dtype=np.float32)


def embed_cached(name: str, texts: list[str]) -> np.ndarray:
    key = hashlib.md5("\u241e".join(texts).encode()).hexdigest()[:10]
    path = f"{CACHE_DIR}/{name}_{key}.npy"
    if os.path.exists(path):
        print(f"[캐시] {name} ({len(texts)}개)")
        return np.load(path)
    t = time.time()
    vecs = embed_texts(texts)
    np.save(path, vecs)
    print(f"[임베딩] {name} ({len(texts)}개) {time.time() - t:.1f}초")
    return vecs


chunk_embeddings = embed_cached("records", docs)


chroma_client = chromadb.PersistentClient(path=DB_DIR)
if "rules" in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection("rules")
collection = chroma_client.create_collection("rules", metadata={"hnsw:space": "cosine"})
collection.add(
    ids=[f"chunk_{i}" for i in range(len(records))],
    documents=docs,
    embeddings=chunk_embeddings.tolist(),
    metadatas=[{k: r[k] for k in ("part", "chapter", "article")} for r in records],
)
print("저장된 청크:", collection.count())


query = "저작권 침해로 인한 손해배상은 어떻게 청구하나요?"
q_vec = embed_texts([query])[0].tolist()
hits = collection.query(query_embeddings=[q_vec], n_results=5)
context = "\n\n".join(hits["documents"][0])
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "아래 저작권법 조문만 근거로 답변하세요."},
        {"role": "user", "content": f"조문:\n{context}\n\n질문: {query}"},
    ],
)
print(response.choices[0].message.content)
import os
import re
import json
import zipfile
import hashlib
import time
from collections import Counter

import numpy as np
import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PDF_PATH = r"C:\Users\201\Downloads\저작권법(법률)(제21336호)(20260811).pdf"
CACHE_DIR = "embedding_cache"
DB_DIR = "ai_agent_db"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"

os.makedirs(CACHE_DIR, exist_ok=True)

def load_document_pages(path: str) -> list[tuple[int, str]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if "manifest.json" in names:
                manifest = json.loads(z.read("manifest.json").decode("utf-8"))
                pages = sorted(manifest["pages"], key=lambda p: p["page_number"])
                return [
                    (p["page_number"], z.read(p["text"]["path"]).decode("utf-8"))
                    for p in pages
                ]
        raise ValueError(f"zip 파일이지만 예상한 manifest.json 구조가 아닙니다: {path}")

    import pypdf
    reader = pypdf.PdfReader(path)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def load_lines_with_pages(path: str) -> list[tuple[int, str]]:
    lines = []
    for page_num, text in load_document_pages(path):
        for line in text.splitlines():
            lines.append((page_num, line))
    return lines


CHAPTER_RE = re.compile(r"^제\d+장(?:의\d+)?\s+\S")


def split_parts(lines_with_pages: list[tuple[int, str]]) -> dict[str, list[tuple[int, str]]]:
    parts: dict[str, list[tuple[int, str]]] = {}
    current = None
    for page_num, line in lines_with_pages:
        s = line.strip()
        if CHAPTER_RE.match(s):
            current = re.sub(r"\s*<.*?>\s*$", "", s)
        elif s.startswith("부칙"):
            current = None
        elif current:
            parts.setdefault(current, []).append((page_num, line))
    return parts


ARTICLE_RE = re.compile(r"^제(\d+(?:조의\d+|조))\(([^)]+)\)")
SECTION_RE = re.compile(r"^제\d+절\s+(.+)$")


def article_records(lines_with_pages: list[tuple[int, str]], part_name: str) -> list[dict]:
    records, section = [], None
    art_no = art_title = art_page = None
    buf: list[str] = []

    def flush():
        if art_no:
            records.append({
                "part": part_name,
                "chapter": section or part_name,
                "article": f"제{art_no}({art_title})",
                "page": art_page,
                "text": "\n".join(buf).strip(),
            })

    for page_num, line in lines_with_pages:
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
            art_no, art_title, art_page = m.group(1), m.group(2), page_num
            buf = [s]
        elif art_no:
            buf.append(s)
    flush()
    return records

def embed_texts(client: OpenAI, texts: list[str], batch: int = 100) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        vectors.extend(item.embedding for item in resp.data)
    return np.array(vectors, dtype=np.float32)


def embed_cached(client: OpenAI, name: str, texts: list[str]) -> np.ndarray:
    key = hashlib.md5(f"{EMBED_MODEL}\u241e{'\u241e'.join(texts)}".encode()).hexdigest()[:10]
    path = f"{CACHE_DIR}/{name}_{key}.npy"
    if os.path.exists(path):
        return np.load(path)
    vecs = embed_texts(client, texts)
    np.save(path, vecs)
    return vecs


@st.cache_resource(show_spinner="저작권법 조문을 불러오고 임베딩하는 중입니다...")
def build_pipeline():
    client = OpenAI(api_key=os.getenv("api_key"))

    lines_with_pages = load_lines_with_pages(PDF_PATH)
    parts = split_parts(lines_with_pages)
    records = [r for name, lp in parts.items() for r in article_records(lp, name)]
    docs = [f"[{r['part']} | {r['page']}페이지] {r['article']} {r['text']}" for r in records]

    chunk_embeddings = embed_cached(client, "records", docs)

    collection_name = f"rules_{EMBED_MODEL.replace('-', '_')}"
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    if collection_name not in [c.name for c in chroma_client.list_collections()]:
        collection = chroma_client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        collection.add(
            ids=[f"chunk_{i}" for i in range(len(records))],
            documents=docs,
            embeddings=chunk_embeddings.tolist(),
            metadatas=[{"part": r["part"], "chapter": r["chapter"], "article": r["article"], "page": r["page"]} for r in records],
        )
    else:
        collection = chroma_client.get_collection(collection_name)

    return client, collection, len(records)


def ask(client: OpenAI, collection, question: str):
    q_vec = embed_texts(client, [question])[0].tolist()
    hits = collection.query(query_embeddings=[q_vec], n_results=5)
    context = "\n\n".join(hits["documents"][0])
    metadatas = hits["metadatas"][0]

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "아래 저작권법 조문만 근거로 답변하세요. 그리고 항상 답변 끝에 근거(조항 번호, 페이지 등)을 명시하세요."},
            {"role": "system", "content": "문서에 없는 내용은 '문서에서 확인되지 않습니다'라고 답변하세요."},
            {"role": "user", "content": f"조문:\n{context}\n\n질문: {question}"},
        ],
    )
    return response.choices[0].message.content, metadatas


st.set_page_config(page_title="저작권법 Q&A", page_icon="⚖️")
st.title("⚖️ 저작권법 Q&A")
st.caption("저작권법 조문을 근거로만 답변하는 챗봇입니다.")

client, collection, n_records = build_pipeline()
st.sidebar.success(f"조문 {n_records}개 로드 완료")
st.sidebar.caption(f"임베딩 모델: {EMBED_MODEL}\n답변 모델: {CHAT_MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("참고한 조문"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['article']}** ({src['part']}, {src['page']}페이지)")

question = st.chat_input("저작권법에 대해 물어보세요")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("조문을 찾아보는 중..."):
            answer, sources = ask(client, collection, question)
        st.markdown(answer)
        with st.expander("참고한 조문"):
            for src in sources:
                st.markdown(f"- **{src['article']}** ({src['part']}, {src['page']}페이지)")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
import os
import re
import hashlib
from collections import Counter

import numpy as np
import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import pypdf

load_dotenv()

PDF_PATH = r"C:\Users\201\Downloads\저작권법(법률)(제21336호)(20260811).pdf"
CACHE_DIR = "embedding_cache"
DB_DIR = "ai_agent_db"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"

os.makedirs(CACHE_DIR, exist_ok=True)

def load_document_pages(path: str) -> list[tuple[int, str]]:
    reader = pypdf.PdfReader(path)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def load_lines_with_pages(path: str) -> list[tuple[int, str]]:
    """모든 페이지를 (page_number, line) 단위로 펼침."""
    lines = []
    for page_num, text in load_document_pages(path):
        for line in text.splitlines():
            lines.append((page_num, line))
    return lines


CHAPTER_RE = re.compile(r"^제\d+장(?:의\d+)?\s+\S")
ARTICLE_RE = re.compile(r"^제(\d+(?:조의\d+|조))\(([^)]+)\)")
SECTION_RE = re.compile(r"^제\d+절\s+(.+)$")


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


def make_article_records(lines_with_pages: list[tuple[int, str]]) -> list[dict]:
    parts = split_parts(lines_with_pages)
    return [r for name, lp in parts.items() for r in article_records(lp, name)]


def flatten_with_chapter(lines_with_pages: list[tuple[int, str]]) -> list[tuple[int, str, str]]:
    """조 단위와 달리 부칙 등을 포함한 문서 전체를 대상으로, 각 줄에 현재
    장(章) 이름을 태깅해서 (page, chapter, line) 리스트로 반환."""
    result = []
    current_chapter = "머리말"
    for page_num, line in lines_with_pages:
        s = line.strip()
        if CHAPTER_RE.match(s):
            current_chapter = re.sub(r"\s*<.*?>\s*$", "", s)
        elif s.startswith("부칙"):
            current_chapter = "부칙"
        result.append((page_num, current_chapter, line))
    return result


def make_chapter_page_records(lines_with_pages: list[tuple[int, str]]) -> list[dict]:
    """장으로 묶되, 같은 장이라도 페이지가 바뀌면 별도 청크로 분리.
    장 하나를 통째로 넣을 때보다 청크 크기가 페이지 단위로 고르게 맞춰지고,
    임베딩 토큰 한도를 넘길 걱정도 자연히 줄어듦."""
    tagged = flatten_with_chapter(lines_with_pages)
    groups: dict[tuple[str, int], list[str]] = {}
    order: list[tuple[str, int]] = []
    for page_num, chapter, line in tagged:
        key = (chapter, page_num)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(line)

    records = []
    for chapter, page_num in order:
        text = "\n".join(groups[(chapter, page_num)]).strip()
        if text:
            records.append({
                "part": chapter,
                "chapter": chapter,
                "article": f"{chapter} ({page_num}페이지)",
                "page": page_num,
                "text": text,
            })
    return records

STRATEGIES = {
    "article": {
        "label": "조(條) 단위",
        "build": make_article_records,
    },
    "chapter_page": {
        "label": "장 단위 + 페이지 분할",
        "build": make_chapter_page_records,
    },
}

def embed_texts(client: OpenAI, texts: list[str], batch: int = 100) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        vectors.extend(item.embedding for item in resp.data)
    return np.array(vectors, dtype=np.float32)


def embed_cached(client: OpenAI, name: str, texts: list[str]) -> np.ndarray:
    """청크 내용 + 모델 + (name으로 넘긴) 전략이 같을 때만 캐시 재사용."""
    key = hashlib.md5(f"{EMBED_MODEL}\u241e{'\u241e'.join(texts)}".encode()).hexdigest()[:10]
    path = f"{CACHE_DIR}/{name}_{key}.npy"
    if os.path.exists(path):
        return np.load(path)
    vecs = embed_texts(client, texts)
    np.save(path, vecs)
    return vecs

@st.cache_resource
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("api_key"))


@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=DB_DIR)

@st.cache_resource(show_spinner="조문을 불러오고 임베딩하는 중입니다...")
def build_pipeline(strategy_key: str):
    client = get_openai_client()
    chroma_client = get_chroma_client()

    lines_with_pages = load_lines_with_pages(PDF_PATH)
    records = STRATEGIES[strategy_key]["build"](lines_with_pages)
    docs = [f"[{r['part']} | {r['page']}페이지] {r['article']} {r['text']}" for r in records]

    chunk_embeddings = embed_cached(client, strategy_key, docs)

    collection_name = f"rules_{strategy_key}_{EMBED_MODEL.replace('-', '_')}"
    existing_names = [c.name for c in chroma_client.list_collections()]
    collection = chroma_client.get_collection(collection_name) if collection_name in existing_names else None

    if collection is not None and collection.count() != len(records):
        chroma_client.delete_collection(collection_name)
        collection = None

    if collection is None:
        collection = chroma_client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        collection.add(
            ids=[f"chunk_{i}" for i in range(len(records))],
            documents=docs,
            embeddings=chunk_embeddings.tolist(),
            metadatas=[{"part": r["part"], "chapter": r["chapter"], "article": r["article"], "page": r["page"]} for r in records],
        )

    return client, collection, len(records)


def ask(client: OpenAI, collection, question: str):
    q_vec = embed_texts(client, [question])[0].tolist()
    hits = collection.query(query_embeddings=[q_vec], n_results=5)
    context = "\n\n".join(hits["documents"][0])
    metadatas = hits["metadatas"][0]

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "아래 저작권법 조문만 근거로 답변하세요. 항상 조문 번호와 해당 조문이 적혀있는 페이지를 함께 언급하세요."},
            {"role": "system", "content": "문서에 없는 내용은 '문서에서 확인되지 않습니다'라고 답변하세요."},
            {"role": "user", "content": f"조문:\n{context}\n\n질문: {question}"},
        ],
    )
    return response.choices[0].message.content, metadatas

st.set_page_config(page_title="저작권법 Q&A", page_icon="⚖️")
st.title("⚖️ 저작권법 Q&A")
st.caption("청킹 전략을 바꿔가며 검색/답변 품질을 비교해볼 수 있습니다.")

strategy_key = st.sidebar.selectbox(
    "청킹 전략",
    options=list(STRATEGIES.keys()),
    format_func=lambda k: STRATEGIES[k]["label"],
)

client, collection, n_records = build_pipeline(strategy_key)
st.sidebar.success(f"[{STRATEGIES[strategy_key]['label']}] 청크 {n_records}개 로드 완료")
st.sidebar.caption(f"임베딩 모델: {EMBED_MODEL}\n답변 모델: {CHAT_MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = {}
history = st.session_state.messages.setdefault(strategy_key, [])

for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("참고한 조문"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['article']}** ({src['part']}, {src['page']}페이지)")

question = st.chat_input("저작권법에 대해 물어보세요")
if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("조문을 찾아보는 중..."):
            answer, sources = ask(client, collection, question)
        st.markdown(answer)
        with st.expander("참고한 조문"):
            for src in sources:
                st.markdown(f"- **{src['article']}** ({src['part']}, {src['page']}페이지)")

    history.append({"role": "assistant", "content": answer, "sources": sources})
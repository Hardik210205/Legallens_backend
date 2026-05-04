#!/usr/bin/env python3
import os
import re
import json
import shutil
import requests
from pathlib import Path
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")

# -------- Retrieval --------
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# -------- LCEL --------
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# =====================================
# CONFIG
# =====================================
DATA_DIR = "data"
UPLOADS_DIR = "uploads"
INDEX_DIR = "index_data"
CHUNK_DIR = f"{INDEX_DIR}/chunks"

EMB_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
HF_API_URL = "https://router.huggingface.co/featherless-ai/v1/chat/completions"

CHUNK_SIZE = 3000
OVERLAP = 300
TOP_BM25 = 20
TOP_DENSE = 20
FINAL_TOPK = 5

os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# =====================================
# LOAD RETRIEVAL MODELS
# (lightweight — no GPU needed)
# =====================================
print("🔄 Loading retrieval models...")
embedder = SentenceTransformer(EMB_MODEL)
reranker = CrossEncoder(RERANK_MODEL)
print("✅ Retrieval models ready\n")

# =====================================
# HF INFERENCE API CALL
# =====================================
def call_hf_api(messages: list) -> str:
    """Call HuggingFace Inference API for Mistral."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.2
    }

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code == 503:
            # Model is loading — wait and retry once
            import time
            wait_time = response.json().get("estimated_time", 30)
            print(f"⏳ Model loading, waiting {wait_time:.0f}s...")
            time.sleep(min(wait_time, 60))
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            return str(result)

        return f"API Error {response.status_code}: {response.text}"

    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Error calling HF API: {str(e)}"


# =====================================
# PROMPT BUILDER
# =====================================
def build_prompt(question: str, context: str, chat_history: list) -> list:
    """Build Mistral instruct format prompt."""
    system = (
        "You are a legal AI assistant. Answer questions based on the provided context. "
        "Be concise and direct. Do not output JSON. Do not repeat the question. "
        "If the context does not contain the answer, say so clearly."
    )

    messages = [{"role": "system", "content": system}]
    
    # Build history
    for msg in chat_history:
        if hasattr(msg, "type"):
            if msg.type == "human":
                messages.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                messages.append({"role": "assistant", "content": msg.content})

    user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
    messages.append({"role": "user", "content": user_content})

    return messages


# =====================================
# DOCX EXTRACTION
# =====================================
def extract_docx(path: Path) -> str:
    """Extract plain text from a .docx file."""
    try:
        import docx
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    paragraphs.append(row_text)
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"⚠️  Could not read {path.name}: {e}")
        return ""


# =====================================
# SYNC UPLOADS → DATA
# =====================================
def sync_uploads_to_data():
    uploads_dir = Path(UPLOADS_DIR)
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)

    if not uploads_dir.exists():
        return

    supported = {".txt", ".md", ".pdf", ".docx"}

    for f in uploads_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in supported:
            continue

        if f.suffix.lower() == ".docx":
            dest = data_dir / (f.stem + "_docx.txt")
            if not dest.exists():
                text = extract_docx(f)
                if text.strip():
                    dest.write_text(text, encoding="utf-8")
                    print(f"📥 Extracted & synced: {f.name} → {dest.name}")
        else:
            dest = data_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                print(f"📥 Synced: {f.name}")


# =====================================
# INDEXING
# =====================================
def chunk_text(text):
    i = 0
    while i < len(text):
        yield text[i:i + CHUNK_SIZE].strip()
        i += CHUNK_SIZE - OVERLAP


def index_documents():
    print("📚 Indexing documents...")
    sync_uploads_to_data()

    files = list(Path(DATA_DIR).rglob("*"))
    meta, corpus, vectors = [], [], []
    cid = 0

    for f in tqdm(files):
        if not f.is_file():
            continue

        suffix = f.suffix.lower()
        if suffix == ".docx":
            text = extract_docx(f)
        else:
            text = f.read_text(errors="ignore")

        if not text.strip():
            continue

        for chunk in chunk_text(text):
            if not chunk:
                continue

            chunk_path = Path(CHUNK_DIR) / f"chunk_{cid}.txt"
            chunk_path.write_text(chunk, encoding="utf-8")

            meta.append({
                "id": cid,
                "doc": f.name,
                "path": str(chunk_path)
            })

            corpus.append(chunk)
            vectors.append(chunk)
            cid += 1

    if cid == 0:
        print("⚠️  No documents found to index.")
        return

    embeddings = embedder.encode(vectors, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")
    json.dump(corpus, open(f"{INDEX_DIR}/bm25.json", "w"))
    json.dump(meta, open(f"{INDEX_DIR}/meta.json", "w"))

    print(f"✅ Indexed {cid} chunks")


# =====================================
# ANSWER PARSER
# =====================================
def parse_answer(raw: str) -> str:
    raw = raw.strip()

    if raw.startswith("Answer:"):
        raw = raw[len("Answer:"):].strip()
    if raw.startswith("Output:"):
        raw = raw[len("Output:"):].strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "answer" in data:
            return data["answer"].strip()
    except Exception:
        pass

    match = re.search(r'"answer"\s*:\s*"(.*?)"(?:\s*,|\s*\})', raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = raw.split("\n")
    clean_lines = []
    skip = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Question:") or stripped.startswith("Output:"):
            skip = True
            continue
        if skip and stripped == "":
            skip = False
            continue
        if not skip:
            clean_lines.append(line)

    cleaned = "\n".join(clean_lines).strip()
    if cleaned:
        return cleaned

    return raw


# =====================================
# HYBRID RETRIEVER
# =====================================
def hybrid_retrieve(query):
    sync_uploads_to_data()

    if not Path(f"{INDEX_DIR}/faiss.index").exists():
        index_documents()
    
    if not Path(f"{INDEX_DIR}/faiss.index").exists():
        return "No documents have been indexed yet. Please upload a document first."

    # Re-index if new files appeared
    data_files = set(f.name for f in Path(DATA_DIR).rglob("*") if f.is_file())
    meta_path = Path(f"{INDEX_DIR}/meta.json")
    if meta_path.exists():
        meta = json.load(open(meta_path))
        indexed_files = set(m["doc"] for m in meta)
        if data_files - indexed_files:
            print("🔄 New files detected, re-indexing...")
            index_documents()

    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    corpus = json.load(open(f"{INDEX_DIR}/bm25.json"))
    meta = json.load(open(f"{INDEX_DIR}/meta.json"))

    if not corpus:
        return "No documents have been indexed yet."

    bm25 = BM25Okapi([c.split() for c in corpus])
    bm25_ids = np.argsort(
        bm25.get_scores(query.split())
    )[::-1][:TOP_BM25]

    q_emb = embedder.encode([query], convert_to_numpy=True)
    _, dense_ids = index.search(q_emb.astype(np.float32), TOP_DENSE)

    candidates = list(set(bm25_ids.tolist() + dense_ids[0].tolist()))

    texts, ids = [], []
    for cid in candidates:
        path = meta[cid]["path"]
        texts.append(Path(path).read_text(errors="ignore"))
        ids.append(cid)

    scores = reranker.predict([[query, t[:512]] for t in texts])

    ranked = sorted(
        zip(ids, scores),
        key=lambda x: x[1],
        reverse=True
    )[:FINAL_TOPK]

    docs = []
    for cid, _ in ranked:
        m = meta[cid]
        content = Path(m["path"]).read_text(errors="ignore")
        docs.append(content)

    return "\n\n".join(docs)


# =====================================
# MEMORY STORE
# =====================================
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# =====================================
# MAIN CHAIN (replaces LCEL + local LLM)
# =====================================
class RAGChain:
    """
    Drop-in replacement for chain_with_memory.
    Uses HF Inference API instead of local Mistral.
    Same interface: .invoke({"question": q}, config={"configurable": {"session_id": sid}})
    """

    def invoke(self, inputs: dict, config: dict = None) -> str:
        question = inputs["question"]
        session_id = "default"

        if config and "configurable" in config:
            session_id = config["configurable"].get("session_id", "default")

        # Get chat history
        history = get_session_history(session_id)

        # Retrieve context
        context = hybrid_retrieve(question)

        # Build prompt
        prompt = build_prompt(question, context, history.messages)

        # Call HF API
        raw_answer = call_hf_api(prompt)

        # Parse and clean
        answer = parse_answer(raw_answer)

        # Save to memory
        history.add_user_message(question)
        history.add_ai_message(answer)

        return answer


chain_with_memory = RAGChain()


# =====================================
# MAIN LOOP (for local testing)
# =====================================
def main():
    print("🧠 LegalLens RAG Ready (HF Inference API)\n")
    session_id = "legal-session"

    while True:
        query = input(">> ")
        if query.lower() == "exit":
            break

        answer = chain_with_memory.invoke(
            {"question": query},
            config={"configurable": {"session_id": session_id}}
        )

        print("\n📄 RESPONSE\n")
        print(answer)
        print()


if __name__ == "__main__":
    main()
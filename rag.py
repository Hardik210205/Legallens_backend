#!/usr/bin/env python3
import os
import re
import json
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")

# -------- Torch / HF --------
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel

# -------- Retrieval --------
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# -------- LCEL --------
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# =====================================
# CONFIG
# =====================================
DATA_DIR = "data"
UPLOADS_DIR = "uploads"
INDEX_DIR = "index_data"
CHUNK_DIR = f"{INDEX_DIR}/chunks"

EMB_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "models/mistral_finetuned"

CHUNK_SIZE = 3000
OVERLAP = 300
TOP_BM25 = 20
TOP_DENSE = 20
FINAL_TOPK = 5

os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# =====================================
# LOAD MODELS
# =====================================
print("🔄 Loading models...")

embedder = SentenceTransformer(EMB_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    use_fast=True
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    token=HF_TOKEN
)

if Path(LORA_PATH).exists():
    model = PeftModel.from_pretrained(model, LORA_PATH, token=HF_TOKEN)
    print("✅ LoRA Loaded")

model.eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.2,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)

print("✅ Model Ready\n")

# =====================================
# DOCX EXTRACTION
# =====================================
def extract_docx(path: Path) -> str:
    """Extract plain text from a .docx file."""
    try:
        import docx
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Also extract tables
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
    """
    Copy newly uploaded files from uploads/ into data/
    so the RAG indexer can find them.
    Supported: .txt  .md  .pdf  .docx
    For .docx: extract text and save as .txt in data/
    """
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
            # Save extracted text as .txt so indexer can read it
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

    # Always sync uploads before indexing
    sync_uploads_to_data()

    files = list(Path(DATA_DIR).rglob("*"))
    meta, corpus, vectors = [], [], []
    cid = 0

    for f in tqdm(files):
        if not f.is_file():
            continue

        # Read text based on file type
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

    print(f"✅ Indexed {cid} chunks from {len(files)} files")

# =====================================
# ANSWER PARSER
# =====================================
def parse_answer(raw: str) -> str:
    """
    Clean up raw Mistral output.
    Handles: JSON strings, Output: prefix, leaked chat history.
    """
    raw = raw.strip()

    # ADD THIS — remove "Answer:" prefix
    if raw.startswith("Answer:"):
        raw = raw[len("Answer:"):].strip()

    # Remove 'Output:' prefix
    if raw.startswith("Output:"):
        raw = raw[len("Output:"):].strip()

    # Try full JSON parse → extract 'answer' field
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "answer" in data:
            return data["answer"].strip()
    except Exception:
        pass

    # Try regex for {"answer": "..."} pattern (handles truncated JSON)
    match = re.search(r'"answer"\s*:\s*"(.*?)"(?:\s*,|\s*\})', raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strip leaked Question:/Output: history blocks
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
    # Sync any new uploads and re-index if needed
    sync_uploads_to_data()

    if not Path(f"{INDEX_DIR}/faiss.index").exists():
        index_documents()

    # Check if new files appeared since last index
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
# PROMPT
# =====================================
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a legal AI assistant. Answer questions based on the provided context. "
        "Be concise and direct. Do not output JSON. Do not repeat the question. "
        "If the context does not contain the answer, say so clearly."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

# =====================================
# LCEL CHAIN
# =====================================
rag_chain = (
    RunnablePassthrough.assign(
        context=lambda inputs: hybrid_retrieve(inputs["question"])
    )
    | prompt
    | llm
)

# =====================================
# MEMORY
# =====================================
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# =====================================
# MAIN LOOP
# =====================================
def main():
    print("🧠 Pure LCEL Legal RAG Ready\n")
    session_id = "legal-session"

    while True:
        query = input(">> ")
        if query.lower() == "exit":
            break

        response = chain_with_memory.invoke(
            {"question": query},
            config={"configurable": {"session_id": session_id}}
        )

        print("\n📄 RESPONSE\n")
        print(parse_answer(str(response)))

if __name__ == "__main__":
    main()
# agents/memory/semantic.py

import os
import ast
import sys
import chromadb
from sentence_transformers import SentenceTransformer

# Index lives on disk — persists across restarts
INDEX_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "../../memory/codebase_index")
)

# all-MiniLM-L6-v2: small, fast, good for code similarity
# Downloads once (~90MB), cached locally after that
_model  = None
_client = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_collection():
    """Lazy-init ChromaDB collection. Creates index dir if needed."""
    global _client, _collection
    if _collection is None:
        os.makedirs(INDEX_PATH, exist_ok=True)
        _client     = chromadb.PersistentClient(path=INDEX_PATH)
        _collection = _client.get_or_create_collection(
            name     = "codebase",
            metadata = {"hnsw:space": "cosine"}   # use cosine similarity
        )
    return _collection


# ── Chunking ───────────────────────────────────────────────────────────────

def extract_chunks(file_path: str, file_content: str) -> list[dict]:
    """
    Split a Python file into function/class chunks using the AST.
    Each chunk = one function or class with its source lines.
    Falls back to whole-file chunk if parsing fails (non-Python files).

    Returns list of:
      {id, content, metadata: {file, name, type, start_line}}
    """
    chunks = []
    try:
        tree = ast.parse(file_content)
        lines = file_content.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Only top-level and class-level — skip nested functions
                start = node.lineno - 1
                end   = node.end_lineno
                chunk_source = "\n".join(lines[start:end])

                chunk_id = f"{file_path}::{node.name}::{start}"
                chunks.append({
                    "id":      chunk_id,
                    "content": f"# File: {file_path}\n{chunk_source}",
                    "metadata": {
                        "file":       file_path,
                        "name":       node.name,
                        "type":       type(node).__name__,
                        "start_line": node.lineno,
                    }
                })

    except SyntaxError:
        # Non-Python or unparseable — index the whole file as one chunk
        chunks.append({
            "id":      f"{file_path}::_whole_file",
            "content": f"# File: {file_path}\n{file_content[:3000]}",
            "metadata": {
                "file":       file_path,
                "name":       "_whole_file",
                "type":       "file",
                "start_line": 1,
            }
        })

    return chunks


# ── Indexing ───────────────────────────────────────────────────────────────

def index_file(file_path: str, file_content: str) -> int:
    """
    Chunk a file, embed each chunk, store in ChromaDB.
    Safe to call multiple times — upsert overwrites existing chunks.
    Returns number of chunks indexed.
    """
    chunks     = extract_chunks(file_path, file_content)
    collection = _get_collection()
    model      = _get_model()

    if not chunks:
        return 0

    ids       = [c["id"]       for c in chunks]
    contents  = [c["content"]  for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Embed all chunks in one batch — faster than one-by-one
    embeddings = model.encode(contents, show_progress_bar=False).tolist()

    collection.upsert(
        ids        = ids,
        documents  = contents,
        embeddings = embeddings,
        metadatas  = metadatas,
    )

    return len(chunks)


def index_directory(directory: str, extensions: list[str] = None) -> dict:
    """
    Walk a directory and index all matching files.
    Default extensions: .py .js .ts .go .java .rs
    Returns summary: {files_indexed, chunks_indexed}
    """
    if extensions is None:
        extensions = [".py", ".js", ".ts", ".go", ".java", ".rs", ".rb"]

    files_indexed  = 0
    chunks_indexed = 0

    for root, _, files in os.walk(directory):
        for fname in files:
            if any(fname.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, fname)
                rel_path  = os.path.relpath(full_path, directory)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    n = index_file(rel_path, content)
                    chunks_indexed += n
                    files_indexed  += 1
                except Exception:
                    pass   # skip unreadable files silently

    return {"files_indexed": files_indexed, "chunks_indexed": chunks_indexed}


# ── Querying ───────────────────────────────────────────────────────────────

def query(text: str, top_k: int = 5, file_filter: str = None) -> list[dict]:
    """
    Find the top_k most semantically similar chunks to the query text.
    Optional file_filter restricts results to one file.

    Returns list of:
      {content, file, name, start_line, score}
    """
    collection = _get_collection()
    model      = _get_model()

    if collection.count() == 0:
        return []

    embedding = model.encode([text], show_progress_bar=False).tolist()

    where = {"file": file_filter} if file_filter else None

    results = collection.query(
        query_embeddings = embedding,
        n_results        = min(top_k, collection.count()),
        where            = where,
        include          = ["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "content":    doc,
            "file":       meta["file"],
            "name":       meta["name"],
            "start_line": meta["start_line"],
            "score":      round(1 - dist, 4),  # cosine distance → similarity
        })

    return chunks


def get_relevant_files(issue_text: str, top_k: int = 5) -> list[str]:
    """
    Given an issue description, return the most relevant file names.
    Used by PlannerAgent to populate TaskCard.context.relevant_files
    without reading every file.
    """
    chunks = query(issue_text, top_k=top_k * 2)
    # Deduplicate — multiple chunks from same file, keep unique files
    seen  = set()
    files = []
    for chunk in chunks:
        if chunk["file"] not in seen:
            seen.add(chunk["file"])
            files.append(chunk["file"])
        if len(files) >= top_k:
            break
    return files
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from loaders_3 import load_to_text, SUPPORTED_EXTS

DOCS_DIR = r".\docs"
KB_DIR = r".\kb_store"
MODEL_DIR = r".\sentence-transformer\sentence-transformer\all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
SAVE_EVERY_N_FILES = 10


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    i = 0
    n = len(text)

    while i < n:
        j = min(n, i + chunk_size)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)

        if j == n:
            break

        i = max(0, j - overlap)

    return chunks


def ensure_dirs() -> None:
    os.makedirs(KB_DIR, exist_ok=True)


def paths() -> Dict[str, str]:
    return {
        "index": os.path.join(KB_DIR, "faiss.index"),
        "meta": os.path.join(KB_DIR, "meta.jsonl"),
        "state": os.path.join(KB_DIR, "state.json"),
    }


def load_state() -> Dict[str, Any]:
    p = paths()["state"]
    if not os.path.exists(p):
        return {"files": {}}

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, Any]) -> None:
    with open(paths()["state"], "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def append_meta(rows: List[Dict[str, Any]]) -> None:
    with open(paths()["meta"], "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def list_docs_recursive(docs_dir: str) -> List[str]:
    base = Path(docs_dir)
    files: List[str] = []

    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(os.path.normpath(str(p)))

    return sorted(files)


def save_checkpoint(index: faiss.Index, state: Dict[str, Any]) -> None:
    faiss.write_index(index, paths()["index"])
    save_state(state)
    print("[CHECKPOINT] Saved index and state.")


def safe_relpath(file_path: str, docs_dir_abs: str) -> str:
    """
    Safely compute a relative path. If relpath fails because of mount/path-style
    mismatch (e.g., Z: vs UNC), fall back to basename so ingestion still continues.
    """
    try:
        return os.path.relpath(file_path, docs_dir_abs)
    except Exception:
        return os.path.basename(file_path)


def main() -> None:
    ensure_dirs()

    docs_dir_abs = os.path.normpath(os.path.abspath(DOCS_DIR))
    kb_paths = paths()

    if not os.path.isdir(docs_dir_abs):
        raise SystemExit(f"[ERROR] DOCS_DIR not found: {docs_dir_abs}")

    if not os.path.isdir(MODEL_DIR):
        raise SystemExit(f"[ERROR] MODEL_DIR not found: {MODEL_DIR}")

    embedder = SentenceTransformer(MODEL_DIR)
    dim = embedder.get_sentence_embedding_dimension()

    if os.path.exists(kb_paths["index"]):
        index = faiss.read_index(kb_paths["index"])
        print("[INFO] Loaded existing FAISS index.")
    else:
        index = faiss.IndexFlatIP(dim)
        print("[INFO] Creating new FAISS index.")

    state = load_state()
    docs = list_docs_recursive(docs_dir_abs)

    if not docs:
        raise SystemExit(f"[ERROR] No supported documents found in {docs_dir_abs}")

    added_files = 0
    added_chunks = 0
    skipped_files = 0
    failed_files = 0

    total_files = len(docs)
    print(f"[INFO] Found {total_files} supported files.")

    for file_num, fp in enumerate(docs, start=1):
        fp = os.path.normpath(fp)
        fname = os.path.basename(fp)

        try:
            fh = sha256_file(fp)
            prev = state["files"].get(fp)

            if prev and prev.get("sha256") == fh:
                skipped_files += 1
                print(f"[SKIP] ({file_num}/{total_files}) {fname} unchanged")
                continue

            text, meta = load_to_text(fp)
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

            if not chunks:
                print(f"[SKIP] ({file_num}/{total_files}) No extractable text: {fname}")
                skipped_files += 1
                state["files"][fp] = {"sha256": fh, "n_chunks": 0}
                continue

            embs = embedder.encode(
                chunks,
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype("float32")

            faiss.normalize_L2(embs)
            index.add(embs)

            rel_path = safe_relpath(fp, docs_dir_abs)

            rows = []
            for i, ch in enumerate(chunks):
                rows.append({
                    "source": meta["source"],
                    "relative_path": rel_path,
                    "path": meta["path"],
                    "ext": meta["ext"],
                    "chunk_id": i,
                    "text": ch,
                })

            append_meta(rows)

            state["files"][fp] = {
                "sha256": fh,
                "n_chunks": len(chunks),
            }

            added_files += 1
            added_chunks += len(chunks)

            print(f"[ADD] ({file_num}/{total_files}) {rel_path}  chunks={len(chunks)}")

            if added_files > 0 and added_files % SAVE_EVERY_N_FILES == 0:
                save_checkpoint(index, state)

        except Exception as e:
            failed_files += 1
            print(f"[FAIL] ({file_num}/{total_files}) {fname} -> {e}")
            continue

    faiss.write_index(index, kb_paths["index"])
    save_state(state)

    print("\n[DONE]")
    print(f"  Added files   : {added_files}")
    print(f"  Added chunks  : {added_chunks}")
    print(f"  Skipped files : {skipped_files}")
    print(f"  Failed files  : {failed_files}")
    print(f"  KB directory  : {KB_DIR}")
    print(f"  Index file    : {kb_paths['index']}")
    print(f"  Meta file     : {kb_paths['meta']}")


if __name__ == "__main__":
    main()
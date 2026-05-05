"""
Builds ChromaDB from chunks.json (produced by chunker.py).
Run once: python setup/build_db.py
"""

import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CHUNKS_FILE = os.path.join(BASE_DIR, "..", "chunks.json")
DB_DIR      = r"E:\chroma_db"   # chroma_db is built on E disk; code runs from D disk
COLLECTION  = "materia_medica"
BATCH_SIZE  = 200

print("[build_db] Loading chunks...")
chunks = json.load(open(CHUNKS_FILE, encoding="utf-8"))
print(f"[build_db] {len(chunks)} chunks loaded")

print("[build_db] Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("[build_db] Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=DB_DIR)

try:
    client.delete_collection(COLLECTION)
    print(f"[build_db] Deleted old '{COLLECTION}' collection")
except Exception:
    pass

db = client.create_collection(COLLECTION)
print(f"[build_db] Created collection '{COLLECTION}'")

total    = len(chunks)
inserted = 0

for start in range(0, total, BATCH_SIZE):
    batch = chunks[start: start + BATCH_SIZE]

    docs   = [c["content"]                      for c in batch]
    metas  = [{"remedy": c["remedy"],
                "source": c["source"],
                "type":   c["type"]}             for c in batch]
    ids    = [f"{c['source']}_{c['remedy']}_{start+i}"
                                                 for i, c in enumerate(batch)]
    embeds = model.encode(docs, show_progress_bar=False).tolist()

    db.add(documents=docs, embeddings=embeds, metadatas=metas, ids=ids)
    inserted += len(batch)

    if inserted % 1000 == 0 or inserted == total:
        print(f"  inserted {inserted}/{total}")

print(f"\n[build_db] DONE — {inserted} chunks in ChromaDB collection '{COLLECTION}'")

import os
import time
import chromadb
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from dotenv import load_dotenv
load_dotenv()
VAULT_PATH    = os.getenv("VAULT_PATH", "./vault")
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80

client = chromadb.PersistentClient(path="./db")
model  = SentenceTransformer("all-MiniLM-L6-v2")

def split_chunks(texto, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    palabras = texto.split()
    chunks = []
    i = 0
    while i < len(palabras):
        chunk = " ".join(palabras[i:i+size])
        chunks.append(chunk)
        i += size - overlap
    return chunks

def get_collection():
    return client.get_or_create_collection("obsidian")

def indexar_nota(path, collection):
    try:
        with open(path, "r", encoding="utf-8") as f:
            contenido = f.read()
        if not contenido.strip():
            return
        titulo  = os.path.basename(path).replace(".md", "")
        carpeta = os.path.relpath(os.path.dirname(path), VAULT_PATH)
        try:
            existentes = collection.get(where={"path": path})
            if existentes["ids"]:
                collection.delete(ids=existentes["ids"])
        except:
            pass
        chunks = split_chunks(contenido)
        ids, docs, embeddings, metas = [], [], [], []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            ids.append(f"{path}_chunk_{i}")
            docs.append(chunk)
            embeddings.append(model.encode(chunk).tolist())
            metas.append({"titulo": titulo, "carpeta": carpeta, "path": path, "chunk": i, "total": len(chunks)})
        if ids:
            collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
        print(f"  ✓ {titulo} ({len(chunks)} chunks)")
    except Exception as e:
        print(f"  ✗ Error en {path}: {e}")

def indexar_todo():
    try:
        client.delete_collection("obsidian")
    except:
        pass
    collection = client.create_collection("obsidian")
    print("Indexando con chunking...\n")
    total = 0
    for root, _, files in os.walk(VAULT_PATH):
        for file in files:
            if file.endswith(".md"):
                indexar_nota(os.path.join(root, file), collection)
                total += 1
    print(f"\nTotal: {total} notas")
    return collection

class NotaHandler(FileSystemEventHandler):
    def __init__(self):
        self.collection = get_collection()
    def on_modified(self, event):
        if event.src_path.endswith(".md"):
            indexar_nota(event.src_path, self.collection)
    def on_created(self, event):
        if event.src_path.endswith(".md"):
            indexar_nota(event.src_path, self.collection)
    def on_deleted(self, event):
        if event.src_path.endswith(".md"):
            try:
                existentes = self.collection.get(where={"path": event.src_path})
                if existentes["ids"]:
                    self.collection.delete(ids=existentes["ids"])
            except:
                pass

if __name__ == "__main__":
    indexar_todo()
    print("\nVigilando cambios...\n")
    handler = NotaHandler()
    observer = Observer()
    observer.schedule(handler, VAULT_PATH, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

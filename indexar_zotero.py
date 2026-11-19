import chromadb
from sentence_transformers import SentenceTransformer
import httpx

ZOTERO_LOCAL = "http://localhost:23119/api"
DB_PATH      = "./db"
BATCH        = 25

client = chromadb.PersistentClient(path=DB_PATH)
model  = SentenceTransformer("all-MiniLM-L6-v2")

def get_all_items():
    items = []
    start = 0
    while True:
        r = httpx.get(f"{ZOTERO_LOCAL}/users/0/items",
                      params={"limit": BATCH, "start": start, "format": "json"})
        batch = r.json()
        if not batch:
            break
        items.extend(batch)
        start += BATCH
        print(f"  Descargados: {len(items)}")
        if len(batch) < BATCH:
            break
    return items

def indexar_zotero():
    try:
        client.delete_collection("zotero")
    except:
        pass
    col = client.create_collection("zotero")

    print("Descargando items de Zotero local...")
    items = get_all_items()
    print(f"Total: {len(items)} items")

    ids, docs, embeddings, metas = [], [], [], []

    for item in items:
        data = item.get("data", {})
        tipo = data.get("itemType", "")
        if tipo in ("attachment", "note"):
            continue

        titulo   = data.get("title", "").strip()
        abstract = data.get("abstractNote", "").strip()
        if not titulo:
            continue

        creators = data.get("creators", [])
        authors  = ", ".join(
            c.get("lastName", "") for c in creators[:3]
            if c.get("creatorType") == "author"
        )
        if len([c for c in creators if c.get("creatorType") == "author"]) > 3:
            authors += " et al."

        year    = data.get("date", "")[:4] if data.get("date") else ""
        journal = data.get("publicationTitle", data.get("publisher", ""))
        doi     = data.get("DOI", "")
        key     = item.get("key", "")

        texto = f"Titulo: {titulo}"
        if authors: texto += f"\nAutores: {authors} ({year})" if year else f"\nAutores: {authors}"
        if journal: texto += f"\nRevista: {journal}"
        if abstract: texto += f"\nResumen: {abstract[:1000]}"

        ids.append(key)
        docs.append(texto)
        embeddings.append(model.encode(texto).tolist())
        metas.append({
            "titulo":   titulo,
            "autores":  authors,
            "year":     year,
            "journal":  journal,
            "doi":      doi,
            "key":      key,
        })
        print(f"  ✓ {titulo[:60]}")

    if ids:
        col.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    print(f"\nIndexados: {len(ids)} papers en ChromaDB")

if __name__ == "__main__":
    indexar_zotero()

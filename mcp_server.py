#!/usr/bin/env python3
"""
Servidor MCP — Segundo Cerebro
Herramientas:
  RAG Obsidian: search_notes, list_folders
  Zotero:       zotero_search, zotero_get_item, zotero_get_collections
"""

import asyncio
import json
import os
import sys
import httpx
from pathlib import Path

# ── Dependencias MCP ──────────────────────────────────────────────────────────
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("ERROR: pip install mcp", file=sys.stderr)
    sys.exit(1)

# ── Dependencias RAG ──────────────────────────────────────────────────────────
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: pip install chromadb sentence-transformers", file=sys.stderr)
    sys.exit(1)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════════════════

VAULT_PATH       = Path("/home/heisser/Documentos/Obsidian/Vault")
CHROMA_DB_PATH   = Path("/home/heisser/segundo_cerebro/db")
CHROMA_COLLECTION = "obsidian"

ZOTERO_USER_ID   = "17878671"
ZOTERO_API_KEY   = "VswwEqtD9lAwKvxTmtjvdD3Q"
ZOTERO_LOCAL_URL = "http://localhost:23119/api"   # Zotero app local
ZOTERO_WEB_URL   = "https://api.zotero.org"

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 5

# ═════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ═════════════════════════════════════════════════════════════════════════════

print("Cargando modelo de embeddings...", file=sys.stderr)
embedder   = SentenceTransformer(EMBED_MODEL)
chroma     = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = chroma.get_or_create_collection(CHROMA_COLLECTION)
server     = Server("segundo-cerebro")

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS ZOTERO
# ═════════════════════════════════════════════════════════════════════════════

async def zotero_request(path: str, params: dict = None) -> dict | list | None:
    """
    Intenta primero Zotero local (puerto 23119).
    Si falla (app cerrada o error), hace fallback a Zotero web.
    Retorna None si ambos fallan.
    """
    headers_web = {
        "Zotero-API-Key": ZOTERO_API_KEY,
        "Zotero-API-Version": "3",
    }

    # 1) Intento local
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            url = f"{ZOTERO_LOCAL_URL}/users/{ZOTERO_USER_ID}{path}"
            r = await client.get(url, params=params)
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass  # App cerrada o no disponible → fallback

    # 2) Fallback web
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{ZOTERO_WEB_URL}/users/{ZOTERO_USER_ID}{path}"
            r = await client.get(url, headers=headers_web, params=params)
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass

    return None


def format_item(item: dict) -> str:
    """Formatea un ítem Zotero para presentación."""
    data = item.get("data", item)  # local y web difieren en estructura
    parts = []

    title     = data.get("title", "Sin título")
    creators  = data.get("creators", [])
    year      = data.get("date", "")[:4] if data.get("date") else ""
    journal   = data.get("publicationTitle", data.get("publisher", ""))
    doi       = data.get("DOI", "")
    abstract  = data.get("abstractNote", "")[:400]
    item_key  = data.get("key", "")
    item_type = data.get("itemType", "")

    authors = ", ".join(
        f"{c.get('lastName', '')} {c.get('firstName', '')}".strip()
        for c in creators[:3]
        if c.get("creatorType") == "author"
    )
    if len(creators) > 3:
        authors += " et al."

    parts.append(f"**{title}**")
    if authors: parts.append(f"Autores: {authors}")
    if year:    parts.append(f"Año: {year}")
    if journal: parts.append(f"Fuente: {journal}")
    if doi:     parts.append(f"DOI: {doi}")
    if item_key:parts.append(f"Key: {item_key} (tipo: {item_type})")
    if abstract:parts.append(f"Resumen: {abstract}...")

    return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# DEFINICIÓN DE HERRAMIENTAS
# ═════════════════════════════════════════════════════════════════════════════

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ── RAG Obsidian ──────────────────────────────────────────────────
        Tool(
            name="search_notes",
            description=(
                "Busca en las notas de Obsidian usando similitud semántica. "
                "Retorna los fragmentos más relevantes con nombre de archivo y score."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Pregunta o término a buscar en las notas"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Número de resultados (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="list_folders",
            description="Lista las carpetas del vault de Obsidian.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        # ── Zotero ────────────────────────────────────────────────────────
        Tool(
            name="zotero_search",
            description=(
                "Busca ítems en la biblioteca Zotero (papers, libros, artículos). "
                "Usa Zotero local si está disponible, sino API web. "
                "Retorna título, autores, año, DOI y resumen."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Término de búsqueda (título, autor, tema)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Máximo de resultados (default 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="zotero_get_item",
            description=(
                "Obtiene todos los detalles de un ítem Zotero por su key. "
                "Incluye abstract completo, tags, colecciones y notas adjuntas."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "item_key": {
                        "type": "string",
                        "description": "Key del ítem Zotero (ej: ABC12345)"
                    }
                },
                "required": ["item_key"]
            }
        ),
        Tool(
            name="zotero_get_collections",
            description="Lista todas las colecciones/carpetas de la biblioteca Zotero.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# LÓGICA DE HERRAMIENTAS
# ═════════════════════════════════════════════════════════════════════════════

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:

    # ── search_notes ──────────────────────────────────────────────────────
    if name == "search_notes":
        query = arguments["query"]
        top_k = arguments.get("top_k", TOP_K)

        embedding = embedder.encode(query).tolist()
        results   = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, collection.count() or 1)
        )

        if not results["documents"][0]:
            return [TextContent(type="text", text="No se encontraron notas relevantes.")]

        output = [f"# Resultados para: '{query}'\n"]
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            score    = round(1 - dist, 3)
            filename = meta.get("filename", "desconocido")
            output.append(f"## [{i+1}] {filename} (score: {score})\n{doc}\n")

        return [TextContent(type="text", text="\n".join(output))]

    # ── list_folders ──────────────────────────────────────────────────────
    elif name == "list_folders":
        if not VAULT_PATH.exists():
            return [TextContent(type="text", text=f"Vault no encontrado: {VAULT_PATH}")]

        folders = sorted([
            str(p.relative_to(VAULT_PATH))
            for p in VAULT_PATH.rglob("*")
            if p.is_dir() and not p.name.startswith(".")
        ])
        text = "# Carpetas del Vault\n" + "\n".join(f"- {f}" for f in folders) if folders else "Vault sin subcarpetas."
        return [TextContent(type="text", text=text)]

    # ── zotero_search ─────────────────────────────────────────────────────
    elif name == "zotero_search":
        query = arguments["query"]
        limit = arguments.get("limit", 10)

        data = await zotero_request("/items", params={
            "q": query,
            "limit": limit,
            "itemType": "-attachment -note",
            "format": "json"
        })

        if data is None:
            return [TextContent(type="text", text="Error: no se pudo conectar a Zotero local ni web.")]

        items = data if isinstance(data, list) else data.get("items", [])

        if not items:
            return [TextContent(type="text", text=f"Sin resultados para: '{query}'")]

        output = [f"# Resultados Zotero: '{query}' ({len(items)} encontrados)\n"]
        for i, item in enumerate(items):
            output.append(f"## [{i+1}]\n{format_item(item)}\n")

        return [TextContent(type="text", text="\n".join(output))]

    # ── zotero_get_item ───────────────────────────────────────────────────
    elif name == "zotero_get_item":
        key = arguments["item_key"]

        data = await zotero_request(f"/items/{key}")

        if data is None:
            return [TextContent(type="text", text=f"No se encontró ítem con key: {key}")]

        output = [f"# Ítem Zotero: {key}\n", format_item(data)]

        # Tags
        tags = data.get("data", data).get("tags", [])
        if tags:
            output.append("\n**Tags:** " + ", ".join(t.get("tag", "") for t in tags))

        # Abstract completo
        abstract = data.get("data", data).get("abstractNote", "")
        if abstract:
            output.append(f"\n**Abstract completo:**\n{abstract}")

        return [TextContent(type="text", text="\n".join(output))]

    # ── zotero_get_collections ────────────────────────────────────────────
    elif name == "zotero_get_collections":
        data = await zotero_request("/collections")

        if data is None:
            return [TextContent(type="text", text="Error conectando a Zotero.")]

        cols = data if isinstance(data, list) else []

        if not cols:
            return [TextContent(type="text", text="No se encontraron colecciones.")]

        output = ["# Colecciones Zotero\n"]
        for col in cols:
            d    = col.get("data", col)
            name = d.get("name", "Sin nombre")
            key  = d.get("key", "")
            count= d.get("numItems", "?")
            output.append(f"- **{name}** (key: {key}, ítems: {count})")

        return [TextContent(type="text", text="\n".join(output))]

    else:
        return [TextContent(type="text", text=f"Herramienta desconocida: {name}")]


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    print("Servidor MCP iniciado — Segundo Cerebro + Zotero", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())

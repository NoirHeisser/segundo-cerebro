import os
import httpx
import asyncio
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

# ── Configuración ────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DB_PATH            = os.getenv("DB_PATH", "./db")
COLLECTION_NAME    = "obsidian"

ZOTERO_USER_ID  = os.getenv("ZOTERO_USER_ID")
ZOTERO_API_KEY  = os.getenv("ZOTERO_API_KEY")
ZOTERO_LOCAL    = "http://localhost:23119/api"
ZOTERO_WEB      = "https://api.zotero.org"

SYSTEM_PROMPT_DEFAULT = (
    "Eres un asesor académico experto y proactivo que ayuda al usuario a desarrollar su tesis. "
    "Tienes acceso a sus notas de Obsidian y papers de Zotero.\n\n"
    "Tu estilo:\n"
    "- Directo y técnico, sin relleno\n"
    "- Siempre sugieres conexiones entre notas y papers que el usuario quizás no vio\n"
    "- Identificas vacíos en la investigación y los señalas\n"
    "- Propones preguntas de investigación concretas\n"
    "- Cuando citas un paper de Zotero, usa SOLO el titulo exacto que aparece en PAPERS DE ZOTERO. NUNCA inventes autores, años o detalles que no esten en el contexto\n"
    "- Cuando usas una nota de Obsidian, mencionas su nombre\n"
    "- Si hay contradiccion entre fuentes, lo señalas explicitamente\n""- Si la informacion no esta en el contexto proporcionado, dilo explicitamente en vez de inventar\n"
    "- Al final de cada respuesta, sugieres 1-2 preguntas de seguimiento relevantes"
)

# Palabras que activan búsqueda en Zotero automáticamente
ZOTERO_KEYWORDS = {
    "paper", "artículo", "estudio", "investigación", "autor",
    "publicación", "revista", "journal", "doi", "referencia", "cita",
    "literatura", "bibliografía", "fuente", "según", "demostró", "encontró",
    "reportó", "publicó", "hallazgo", "evidencia", "zotero",
    "especie", "taxonomía", "experimento", "método", "metodología",
    "diversidad", "comunidad", "abundancia", "riqueza", "ecología",
    "suelo", "soil", "nematodo", "nematode", "fauna", "macrófauna",
    "macrofauna", "invertebrado", "lombriz", "artropodo", "microartropodo",
    "ecosistema", "bosque", "selva", "amazonia", "tropical", "agroforestal",
    "coleoptera", "scarabaeidae", "escarabajo", "insecto", "entomología",
    "cacao", "theobroma", "tratamiento", "parcela", "cobertura",
    "ave", "pájaro", "bird", "abeja", "polinizador",
}

TRADUCCION_QUERY = {
    "suelo": "soil",
    "fauna": "fauna",
    "nematodo": "nematode",
    "lombriz": "earthworm",
    "bosque": "forest",
    "tropical": "tropical",
    "diversidad": "diversity",
    "comunidad": "community",
    "ecosistema": "ecosystem",
    "recuperación": "recovery",
    "ave": "bird",
    "abeja": "bee",
    "escarabajo": "beetle",
    "insecto": "insect",
    "suelos": "soil",
    "invertebrado": "invertebrate",
    "agroforestal": "agroforest",
}

# ── Modelos cargados una sola vez al arrancar ─────────────────────────────────
client_chroma = chromadb.PersistentClient(path=DB_PATH)

def get_collection():
    return client_chroma.get_or_create_collection(COLLECTION_NAME)
embed_model   = SentenceTransformer("all-MiniLM-L6-v2")

client_llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Segundo Cerebro API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class Mensaje(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    pregunta: str
    historial: List[Mensaje] = []
    carpeta: Optional[str] = None
    system_prompt: Optional[str] = None  # personalizable desde el frontend

class ChatResponse(BaseModel):
    respuesta: str
    fuentes: List[str]
    fuentes_zotero: List[str] = []

# ── Helpers Zotero ────────────────────────────────────────────────────────────

def construir_queries_zotero(pregunta: str) -> list[str]:
    palabras = pregunta.lower().split()
    queries  = set()

    for p in palabras:
        p_limpio = p.strip(".,;:¿?¡!")
        if p_limpio in TRADUCCION_QUERY:
            queries.add(TRADUCCION_QUERY[p_limpio])

    for p in palabras:
        p_limpio = p.strip(".,;:¿?¡!")
        if len(p_limpio) > 5:
            queries.add(p_limpio)

    queries.add(pregunta[:50])
    return list(queries)[:4]


async def _zotero_get(query: str, limit: int, headers_web: dict) -> list:
    params = {"q": query, "limit": limit, "format": "json"}

    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"{ZOTERO_LOCAL}/users/{ZOTERO_USER_ID}/items", params=params)
            if r.status_code == 200 and isinstance(r.json(), list):
                return r.json()
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            r = await c.get(f"{ZOTERO_WEB}/users/{ZOTERO_USER_ID}/items",
                            params=params, headers=headers_web)
            if r.status_code == 200 and isinstance(r.json(), list):
                return r.json()
    except Exception:
        pass

    return []


def zotero_buscar_chromadb(pregunta: str, limit: int = 8) -> list:
    try:
        col_zotero = client_chroma.get_collection('zotero')
        embedding  = embed_model.encode(pregunta).tolist()
        resultados = col_zotero.query(query_embeddings=[embedding], n_results=limit)
        items = []
        for doc, meta in zip(resultados['documents'][0], resultados['metadatas'][0]):
            items.append({'_chromadb': True, 'documento': doc, 'meta': meta})
        return items
    except Exception as e:
        print(f'Error ChromaDB zotero: {e}')
        return []

async def zotero_buscar(pregunta: str, limit: int = 8) -> list:
    return zotero_buscar_chromadb(pregunta, limit)


def formatear_item_zotero(item: dict) -> tuple[str, str]:
    data     = item.get("data", item)
    titulo   = data.get("title", "Sin título")
    creators = data.get("creators", [])
    year     = data.get("date", "")[:4] if data.get("date") else ""
    abstract = data.get("abstractNote", "")[:800]
    journal  = data.get("publicationTitle", data.get("publisher", ""))
    doi      = data.get("DOI", "")

    authors = ", ".join(
        f"{c.get('lastName', '')}".strip()
        for c in creators[:3]
        if c.get("creatorType") == "author"
    )
    if len([c for c in creators if c.get("creatorType") == "author"]) > 3:
        authors += " et al."

    partes = [f"Título: {titulo}"]
    if authors: partes.append(f"Autores: {authors} ({year})" if year else f"Autores: {authors}")
    if journal: partes.append(f"Revista: {journal}")
    if doi:     partes.append(f"DOI: {doi}")
    if abstract: partes.append(f"Resumen: {abstract}")

    return titulo, "\n".join(partes)


def necesita_zotero(pregunta: str, n_notas_encontradas: int) -> bool:
    palabras_limpias = {p.strip(".,;:¿?¡!") for p in pregunta.lower().split()}
    hay_keywords  = bool(palabras_limpias & ZOTERO_KEYWORDS)
    pocas_notas   = n_notas_encontradas < 2
    menciona_zot  = "zotero" in pregunta.lower() or "paper" in pregunta.lower()
    return hay_keywords or pocas_notas or menciona_zot


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "notas_indexadas": get_collection().count()}


@app.get("/carpetas")
def listar_carpetas():
    datos    = get_collection().get(include=["metadatas"])
    carpetas = sorted({m.get("carpeta", ".") for m in datos["metadatas"]})
    return {"carpetas": ["Todas"] + carpetas}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    # 1. Buscar en Obsidian (ChromaDB)
    embedding = embed_model.encode(req.pregunta).tolist()

    where = None
    if req.carpeta and req.carpeta != "Todas":
        where = {"carpeta": req.carpeta}

    resultados = get_collection().query(
        query_embeddings=[embedding],
        n_results=3,
        where=where,
    )

    fragmentos = resultados["documents"][0]
    metadatas  = resultados["metadatas"][0]
    titulos    = [m["titulo"] for m in metadatas]

    # 2. Contexto Obsidian
    contexto_obsidian = ""
    for titulo, fragmento in zip(titulos, fragmentos):
        contexto_obsidian += f"\n--- Nota: {titulo} ---\n{fragmento[:1500]}\n"

    # 3. Zotero automático
    titulos_zotero  = []
    contexto_zotero = ""

    if necesita_zotero(req.pregunta, len(fragmentos)):
        items = await zotero_buscar(req.pregunta, limit=8)
        if items:
            partes_zotero = []
            for item in items:
                titulo_z, resumen_z = formatear_item_zotero(item)
                titulos_zotero.append(titulo_z)
                partes_zotero.append(f"\n--- Paper: {titulo_z} ---\n{resumen_z}\n")
            contexto_zotero = "".join(partes_zotero)

    # 4. Construir contexto
    hay_obsidian = bool(contexto_obsidian.strip())
    hay_zotero   = bool(contexto_zotero.strip())

    if not hay_obsidian and not hay_zotero:
        return ChatResponse(
            respuesta="No encontré notas ni papers relevantes para tu pregunta.",
            fuentes=[],
            fuentes_zotero=[],
        )

    secciones = []
    if hay_obsidian:
        secciones.append(f"NOTAS DE OBSIDIAN:\n{contexto_obsidian}")
    if hay_zotero:
        secciones.append(f"PAPERS DE ZOTERO:\n{contexto_zotero}")

    contexto_completo = "\n\n".join(secciones)

    # 5. Historial + pregunta
    mensajes_api = [{"role": m.role, "content": m.content} for m in req.historial]
    mensajes_api.append({
        "role": "user",
        "content": f"{contexto_completo}\n\nPregunta: {req.pregunta}",
    })

    # 6. System prompt: usa el del frontend si viene, si no el default
    system_content = (
        req.system_prompt.strip()
        if req.system_prompt and req.system_prompt.strip()
        else SYSTEM_PROMPT_DEFAULT
    )

    # 7. Llamar al LLM
    respuesta_llm = client_llm.chat.completions.create(
        model="anthropic/claude-3-5-haiku",
        messages=[{"role": "system", "content": system_content}] + mensajes_api,
        max_tokens=1500,
    )

    texto = respuesta_llm.choices[0].message.content
    return ChatResponse(
        respuesta=texto,
        fuentes=titulos,
        fuentes_zotero=titulos_zotero,
    )


@app.get("/zotero/count")
async def zotero_count():
    headers = {"Zotero-API-Key": ZOTERO_API_KEY, "Zotero-API-Version": "3"}
    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            r = await c.get(
                f"{ZOTERO_WEB}/users/{ZOTERO_USER_ID}/items",
                params={"limit": 1, "format": "json"},
                headers=headers
            )
            total = int(r.headers.get("Total-Results", 0))
            return {"total": total}
    except Exception:
        return {"total": 0}


# ── Servir frontend estático ──────────────────────────────────────────────────

if os.path.isdir("./static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

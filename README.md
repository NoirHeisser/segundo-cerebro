# Segundo Cerebro

Sistema RAG que conecta tus notas de Obsidian y papers de Zotero con Claude via OpenRouter. Permite hacer preguntas en lenguaje natural y recibir respuestas basadas en tu propio conocimiento acumulado.

## Arquitectura

| Componente | Tecnologia |
|---|---|
| Base de vectores | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 (local, sin costo) |
| LLM | Claude via OpenRouter |
| Backend | FastAPI |
| Frontend | HTML/JS puro |

## Requisitos

- Python 3.10+
- Obsidian con un vault local
- Cuenta en OpenRouter
- Cuenta en Zotero (opcional)

## Instalacion

    git clone https://github.com/tuusuario/segundo-cerebro.git
    cd segundo-cerebro
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Configuracion

Copia el archivo de ejemplo:

    cp .env.example .env

Edita .env con tus datos:

    OPENROUTER_API_KEY=tu_key_de_openrouter
    ZOTERO_API_KEY=tu_key_de_zotero
    ZOTERO_USER_ID=tu_id_de_zotero
    DB_PATH=./db
    VAULT_PATH=/ruta/a/tu/vault/de/obsidian

## Uso

Terminal 1 - indexar notas:

    source venv/bin/activate
    python indexar.py

Terminal 2 - iniciar API:

    source venv/bin/activate
    uvicorn api:app --host 127.0.0.1 --port 8000

Abre el navegador en: http://127.0.0.1:8000

## Creditos

Construido con ChromaDB, SentenceTransformers, FastAPI y Claude.

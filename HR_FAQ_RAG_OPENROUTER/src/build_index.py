"""
Modulo de construccion del indice vectorial para el sistema RAG de FAQs de HR.

Este script implementa el pipeline completo de indexacion:
  1. Carga del documento fuente (UTF-8)
  2. Division en chunks semanticos con RecursiveCharacterTextSplitter
  3. Generacion de embeddings con OpenAI text-embedding-3-large (via OpenRouter)
  4. Almacenamiento persistente en ChromaDB local

Uso:
    python src/build_index.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# -- Configuracion de rutas y entorno --
# Subimos un nivel desde src/ para llegar a la raiz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# -- Variables de proyecto configurables --
# Elegimos chunk_size=500 y overlap=100 para lograr fragmentos de entre 50-500 tokens.
# Con un ratio promedio de ~1.3 caracteres por token en espanol,
# 500 caracteres producen ~380 tokens, dentro del rango requerido.
# El overlap de 100 caracteres (~75 tokens) preserva contexto entre chunks adyacentes,
# evitando cortes abruptos en medio de ideas relacionadas.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Directorio de persistencia para ChromaDB
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")

# Ruta del documento fuente
FAQ_DOCUMENT_PATH = PROJECT_ROOT / "data" / "faq_document.txt"

# Modelo de embeddings (OpenAI compatible, accedido via OpenRouter)
EMBEDDING_MODEL = "text-embedding-3-large"

# Nombre de la coleccion en ChromaDB
COLLECTION_NAME = "hr_faq_collection"

# -- Configuracion de OpenRouter --
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


def load_document(file_path: Path) -> str:
    """
    Carga el contenido completo de un archivo de texto en formato UTF-8.

    Args:
        file_path: Ruta absoluta al archivo de texto.

    Returns:
        Contenido del archivo como cadena de texto.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontro el documento en: {file_path}"
        )

    # Leemos el archivo con codificacion UTF-8 para soportar caracteres en espanol
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    print(f"  [Load] Documento cargado: {file_path.name} ({len(content)} caracteres)")
    return content


def chunk_document(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list:
    """
    Divide el texto en chunks semanticos usando RecursiveCharacterTextSplitter.

    Estrategia de chunking:
      - Usamos separadores jerarquicos (parrafos, lineas, oraciones, espacios)
        para respetar los limites naturales del texto.
      - chunk_size=500 genera fragmentos de ~380 tokens (ratio espanol ~1.3 char/token),
        dentro del rango de 50-500 tokens requerido.
      - chunk_overlap=100 asegura continuidad semantica entre chunks adyacentes.

    Args:
        text: Texto completo del documento a dividir.
        chunk_size: Tamano maximo de cada chunk en caracteres.
        chunk_overlap: Solapamiento entre chunks consecutivos.

    Returns:
        Lista de objetos Document de LangChain con metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " "],
        length_function=len,
    )

    # Creamos documentos a partir del texto plano
    from langchain_core.documents import Document
    doc = Document(page_content=text, metadata={"source": "faq_document.txt"})
    chunks = splitter.split_documents([doc])

    # Enriquecemos la metadata de cada chunk con su indice
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    print(f"  [Chunk] {len(chunks)} chunks generados (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def create_embeddings() -> OpenAIEmbeddings:
    """
    Inicializa el modelo de embeddings de OpenAI via OpenRouter.

    Usa el modelo text-embedding-3-large a traves de la API compatible
    de OpenRouter, que ofrece vectores de alta calidad para busqueda
    semantica en espanol.

    Returns:
        Instancia configurada de OpenAIEmbeddings apuntando a OpenRouter.

    Raises:
        EnvironmentError: Si OPENROUTER_API_KEY no esta configurada.
    """
    api_key = OPENROUTER_API_KEY
    if not api_key or api_key.startswith("sk-or-v1-REEMPLAZA"):
        raise EnvironmentError(
            "OPENROUTER_API_KEY no esta configurada.\n"
            "  -> Copia .env.example a .env y agrega tu clave de OpenRouter."
        )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
    )
    print(f"  [Embed] Modelo de embeddings: {EMBEDDING_MODEL} (via OpenRouter)")
    return embeddings


def build_vector_store(
    chunks: list,
    embeddings: OpenAIEmbeddings,
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """
    Genera los embeddings y los almacena en una instancia local de ChromaDB.

    ChromaDB se eligio por ser una base de datos vectorial ligera,
    de codigo abierto y con persistencia local, ideal para prototipos RAG
    sin necesidad de infraestructura externa.

    Args:
        chunks: Lista de documentos LangChain a indexar.
        embeddings: Modelo de embeddings para generar vectores.
        persist_directory: Ruta del directorio de persistencia de ChromaDB.

    Returns:
        Instancia de Chroma con los documentos indexados.
    """
    print(f"  [Store] Indexando {len(chunks)} chunks en ChromaDB...")
    print(f"  [Store] Directorio de persistencia: {persist_directory}")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )

    # Verificamos la cantidad de vectores almacenados
    count = vector_store._collection.count()
    print(f"  [Store] ChromaDB: {count} vectores almacenados exitosamente")
    return vector_store


if __name__ == "__main__":
    print("=" * 60)
    print("  BUILD INDEX — Pipeline de Indexacion RAG para HR FAQs")
    print("  (Usando OpenRouter como proveedor de API)")
    print("=" * 60)
    print()

    # Paso 1: Cargar documento
    text = load_document(FAQ_DOCUMENT_PATH)

    # Paso 2: Dividir en chunks
    chunks = chunk_document(text)

    # Validacion: el prompt requiere minimo 20 chunks
    if len(chunks) < 20:
        print(f"\n  [WARN] Solo se generaron {len(chunks)} chunks. Se requieren minimo 20.")
        print("  Ajusta CHUNK_SIZE o agrega mas contenido al documento.")
    else:
        print(f"\n  [OK] Requisito de 20+ chunks cumplido: {len(chunks)} chunks generados.")

    # Paso 3: Crear modelo de embeddings
    embeddings = create_embeddings()

    # Paso 4: Construir vector store
    vector_store = build_vector_store(chunks, embeddings)

    print()
    print("=" * 60)
    print("  BUILD INDEX completado exitosamente")
    print("=" * 60)

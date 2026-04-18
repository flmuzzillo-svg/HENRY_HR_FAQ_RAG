"""
Modulo de consulta del sistema RAG de FAQs de HR.

Implementa el flujo de consulta en dos pasos:
  1. Recuperacion: convierte la pregunta a embedding y busca en ChromaDB
  2. Generacion: usa un prompt estricto para que el LLM responda solo con el contexto

Uso:
    python src/query.py
    python src/query.py "Mi pregunta aqui"
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# -- Configuracion de rutas y entorno --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# -- Variables de proyecto configurables --
# K_NN define la cantidad de chunks a recuperar por consulta (entre 2 y 5)
K_NN = 3

# Configuracion del vector store
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "hr_faq_collection"
EMBEDDING_MODEL = "text-embedding-3-large"

# Modelo de generacion (familia GPT de OpenAI)
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1

# Prompt estricto para que el LLM responda exclusivamente con el contexto proporcionado
SYSTEM_PROMPT = """Eres un asistente especializado en politicas de Recursos Humanos.
Tu UNICA fuente de informacion son los fragmentos de contexto proporcionados.

REGLAS ESTRICTAS:
1. Responde EXCLUSIVAMENTE con la informacion contenida en los fragmentos de contexto.
2. Si la informacion no esta en el contexto, responde: "No tengo informacion suficiente en el documento para responder esta pregunta."
3. No inventes, no asumas, no uses conocimiento externo.
4. Responde de forma clara, concisa y en espanol.
5. Si es relevante, menciona los numeros especificos (dias, porcentajes, montos) que aparecen en el contexto.
"""


def load_vector_store(
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """
    Carga la instancia de ChromaDB existente con los embeddings indexados.

    Args:
        persist_directory: Ruta al directorio de persistencia de ChromaDB.

    Returns:
        Instancia de Chroma lista para consultas.

    Raises:
        FileNotFoundError: Si el directorio de ChromaDB no existe.
    """
    if not Path(persist_directory).exists():
        raise FileNotFoundError(
            f"ChromaDB no encontrada en: {persist_directory}\n"
            "  -> Ejecuta primero: python src/build_index.py"
        )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    count = vector_store._collection.count()
    print(f"  [Store] ChromaDB cargada: {count} vectores disponibles")
    return vector_store


def retrieve_chunks(
    vector_store: Chroma,
    query: str,
    k: int = K_NN,
) -> list:
    """
    Recupera los K chunks mas relevantes para una pregunta dada.

    Convierte la pregunta a embedding y calcula la similitud coseno
    contra los vectores almacenados en ChromaDB.

    Args:
        vector_store: Instancia de Chroma con los documentos indexados.
        query: Pregunta del usuario en lenguaje natural.
        k: Numero de chunks a recuperar (entre 2 y 5).

    Returns:
        Lista de objetos Document con los fragmentos mas relevantes.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    print(f"  [Retrieve] {len(docs)} chunks recuperados para la consulta")
    return docs


def generate_answer(
    question: str,
    chunks: list,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """
    Genera una respuesta anclada al contexto usando un modelo GPT de OpenAI.

    El prompt esta disenado para que el LLM responda SOLO con la informacion
    contenida en los chunks recuperados, sin inventar ni asumir datos.

    Args:
        question: Pregunta original del usuario.
        chunks: Lista de documentos recuperados como contexto.
        model: Nombre del modelo de OpenAI a utilizar.
        temperature: Temperatura de generacion (baja = mas determinista).

    Returns:
        Respuesta generada por el LLM como cadena de texto.
    """
    # Construimos el contexto a partir de los chunks recuperados
    context_parts = []
    for i, doc in enumerate(chunks, 1):
        context_parts.append(f"[Fragmento {i}]: {doc.page_content}")
    context = "\n\n".join(context_parts)

    # Prompt del usuario con contexto y pregunta
    user_prompt = f"""CONTEXTO (fragmentos del documento de HR):
{context}

PREGUNTA DEL USUARIO:
{question}

Responde basandote UNICAMENTE en el contexto proporcionado."""

    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    print(f"  [Generate] Respuesta generada ({len(response.content)} caracteres)")
    return response.content


def run_query(question: str) -> dict:
    """
    Orquesta el flujo completo de consulta RAG: recuperacion + generacion.

    Args:
        question: Pregunta del usuario en lenguaje natural.

    Returns:
        Diccionario JSON validado con las llaves:
          - user_question: la pregunta original
          - system_answer: la respuesta generada
          - chunks_related: lista de fragmentos usados como contexto
    """
    # Paso 1: Cargar el vector store
    vector_store = load_vector_store()

    # Paso 2: Recuperar chunks relevantes
    chunks = retrieve_chunks(vector_store, question)

    # Paso 3: Generar respuesta con el contexto
    answer = generate_answer(question, chunks)

    # Paso 4: Construir el JSON de salida validado
    chunks_related = []
    for doc in chunks:
        chunks_related.append({
            "chunk_index": doc.metadata.get("chunk_index", -1),
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
        })

    result = {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": chunks_related,
    }

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  QUERY — Sistema de Consulta RAG para HR FAQs")
    print("=" * 60)
    print()

    # Permite pasar la pregunta como argumento o usa una por defecto
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Cuantos dias de vacaciones tengo en mi primer anio?"

    print(f"  Pregunta: {question}\n")

    result = run_query(question)

    print()
    print("-" * 60)
    print("  RESULTADO (JSON):")
    print("-" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))

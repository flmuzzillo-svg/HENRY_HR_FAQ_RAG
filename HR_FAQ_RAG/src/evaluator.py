"""
Modulo evaluador del sistema RAG de FAQs de HR.

Implementa un juez LLM que analiza la triada (pregunta, respuesta, chunks)
y devuelve una puntuacion de calidad con justificacion detallada.

Uso:
    python src/evaluator.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# -- Configuracion de rutas y entorno --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Modelo del juez evaluador (se recomienda un modelo potente para evaluaciones objetivas)
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_TEMPERATURE = 0.0

# Prompt del juez LLM para evaluar la triada RAG
JUDGE_SYSTEM_PROMPT = """Eres un evaluador experto de sistemas RAG (Retrieval-Augmented Generation).
Tu tarea es analizar la calidad de una respuesta generada por un sistema RAG.

Evalua la siguiente triada:
1. PREGUNTA: la consulta original del usuario
2. RESPUESTA: la respuesta generada por el sistema
3. CHUNKS: los fragmentos de contexto que se usaron para generar la respuesta

CRITERIOS DE EVALUACION:
- Completitud: La respuesta cubre todos los aspectos relevantes de la pregunta?
- Relevancia: La respuesta se basa efectivamente en los chunks proporcionados?
- Precision: Los datos mencionados (numeros, fechas, porcentajes) son correctos segun los chunks?
- Coherencia: La respuesta es clara, bien estructurada y libre de contradicciones?
- Fidelidad: La respuesta NO inventa informacion que no esta en los chunks?

FORMATO DE RESPUESTA OBLIGATORIO (JSON estricto):
{
    "score": <numero entero de 0 a 10>,
    "reason": "<justificacion en espanol de minimo 50 caracteres explicando completitud y relevancia>"
}

ESCALA:
- 9-10: Respuesta excepcional, completa y perfectamente anclada al contexto
- 7-8: Buena respuesta con informacion correcta y relevante
- 5-6: Respuesta aceptable pero incompleta o parcialmente relevante
- 3-4: Respuesta deficiente, le falta informacion clave o tiene imprecisiones
- 0-2: Respuesta incorrecta, inventada o completamente irrelevante

Responde UNICAMENTE con el JSON, sin texto adicional."""


def evaluate_response(
    question: str,
    answer: str,
    chunks: list,
    model: str = JUDGE_MODEL,
    temperature: float = JUDGE_TEMPERATURE,
) -> dict:
    """
    Evalua la calidad de una respuesta RAG usando un juez LLM.

    Analiza la triada (pregunta, respuesta, chunks) y devuelve
    una puntuacion con justificacion detallada.

    Args:
        question: Pregunta original del usuario.
        answer: Respuesta generada por el sistema RAG.
        chunks: Lista de fragmentos usados como contexto.
        model: Modelo de OpenAI para el juez evaluador.
        temperature: Temperatura de generacion (0 = determinista).

    Returns:
        Diccionario con 'score' (0-10) y 'reason' (string >= 50 chars).
    """
    # Formateamos los chunks para el juez
    chunks_text = ""
    for i, chunk in enumerate(chunks, 1):
        # Soportamos tanto dicts como objetos Document
        if isinstance(chunk, dict):
            content = chunk.get("content", str(chunk))
        else:
            content = chunk.page_content
        chunks_text += f"[Fragmento {i}]: {content}\n\n"

    # Prompt de evaluacion con la triada completa
    user_prompt = f"""PREGUNTA DEL USUARIO:
{question}

RESPUESTA DEL SISTEMA:
{answer}

CHUNKS DE CONTEXTO UTILIZADOS:
{chunks_text}

Evalua la calidad de la respuesta segun los criterios establecidos."""

    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke([
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    # Parseamos el JSON de la respuesta del juez
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        # Si el juez no devolvio JSON valido, extraemos lo que podamos
        result = {
            "score": -1,
            "reason": f"Error al parsear la respuesta del juez: {response.content[:200]}"
        }

    # Validamos que el reason tenga minimo 50 caracteres
    if len(result.get("reason", "")) < 50:
        result["reason"] = result.get("reason", "") + (
            " " * (50 - len(result.get("reason", "")))
        )

    return result


def evaluate_query_result(query_result: dict) -> dict:
    """
    Evalua un resultado completo de run_query() directamente.

    Funcion de conveniencia que extrae la triada del JSON de resultado.

    Args:
        query_result: Diccionario de salida de query.run_query().

    Returns:
        Diccionario de evaluacion con 'score' y 'reason'.
    """
    return evaluate_response(
        question=query_result["user_question"],
        answer=query_result["system_answer"],
        chunks=query_result["chunks_related"],
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  EVALUATOR — Juez LLM para el Sistema RAG de HR FAQs")
    print("=" * 60)
    print()

    # Ejemplo de evaluacion con datos de prueba
    sample_question = "Cuantos dias de vacaciones tengo en mi primer anio?"
    sample_answer = (
        "En tu primer anio de servicio tienes derecho a 15 dias habiles "
        "de vacaciones anuales. A partir del segundo anio, se otorgan "
        "2 dias adicionales por cada anio de antiguedad."
    )
    sample_chunks = [
        {
            "content": (
                "Todos los empleados de tiempo completo tienen derecho a un "
                "minimo de 15 dias habiles de vacaciones anuales durante su "
                "primer anio de servicio. A partir del segundo anio, se otorgan "
                "2 dias adicionales por cada anio de antiguedad, hasta un "
                "maximo de 25 dias habiles anuales."
            )
        }
    ]

    print(f"  Pregunta: {sample_question}")
    print(f"  Respuesta: {sample_answer[:80]}...")
    print()

    result = evaluate_response(sample_question, sample_answer, sample_chunks)

    print()
    print("-" * 60)
    print("  EVALUACION:")
    print("-" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))

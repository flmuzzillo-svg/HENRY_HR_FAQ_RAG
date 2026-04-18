"""
Script principal para ejecutar el pipeline completo del sistema RAG de HR FAQs.

Ejecuta en secuencia:
  1. Construccion del indice (build_index.py)
  2. Consultas de ejemplo (query.py)
  3. Evaluacion de calidad (evaluator.py)
  4. Generacion de outputs/sample_queries.json

Nota: Este proyecto usa OpenRouter como proveedor de API (compatible con OpenAI).

Uso:
    python run_all.py
"""

import json
import sys
from pathlib import Path

# -- Configuracion de rutas --
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    """Ejecuta el pipeline completo del sistema RAG."""
    print()
    print("=" * 70)
    print("  HR FAQ RAG — Pipeline Completo (via OpenRouter)".center(70))
    print("=" * 70)
    print()

    # ======================================================
    # PASO 1: Construir el indice vectorial
    # ======================================================
    print("[PASO 1/3] Construyendo el indice vectorial...")
    print("-" * 50)

    from build_index import (
        load_document,
        chunk_document,
        create_embeddings,
        build_vector_store,
        FAQ_DOCUMENT_PATH,
    )

    text = load_document(FAQ_DOCUMENT_PATH)
    chunks = chunk_document(text)

    if len(chunks) < 20:
        print(f"\n  [WARN] Solo {len(chunks)} chunks generados. Minimo requerido: 20.")
    else:
        print(f"\n  [OK] {len(chunks)} chunks generados (requisito: 20+)")

    embeddings = create_embeddings()
    vector_store = build_vector_store(chunks, embeddings)
    print()

    # ======================================================
    # PASO 2: Ejecutar consultas de ejemplo
    # ======================================================
    print("[PASO 2/3] Ejecutando consultas de ejemplo...")
    print("-" * 50)

    from query import run_query

    # 3 preguntas representativas que cubren diferentes secciones del documento
    sample_questions = [
        "Cuantos dias de vacaciones tengo en mi primer anio y como se acumulan?",
        "Cual es el proceso de onboarding para nuevos empleados y cuanto dura?",
        "Que beneficios de salud y bienestar ofrece la empresa?",
    ]

    results = []
    for i, question in enumerate(sample_questions, 1):
        print(f"\n  --- Consulta {i}/3 ---")
        print(f"  Pregunta: {question}")
        result = run_query(question)
        results.append(result)
        print(f"  Respuesta: {result['system_answer'][:100]}...")
        print(f"  Chunks recuperados: {len(result['chunks_related'])}")

    print()

    # ======================================================
    # PASO 3: Evaluar cada respuesta con el juez LLM
    # ======================================================
    print("[PASO 3/3] Evaluando calidad con juez LLM...")
    print("-" * 50)

    from evaluator import evaluate_query_result

    evaluated_results = []
    for i, result in enumerate(results, 1):
        print(f"\n  --- Evaluando consulta {i}/3 ---")
        evaluation = evaluate_query_result(result)
        result["evaluation"] = evaluation
        evaluated_results.append(result)
        print(f"  Score: {evaluation.get('score', '?')}/10")
        print(f"  Reason: {evaluation.get('reason', '?')[:80]}...")

    # ======================================================
    # Guardar resultados en outputs/sample_queries.json
    # ======================================================
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    output_path = outputs_dir / "sample_queries.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluated_results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print(f"  Pipeline completado exitosamente!")
    print(f"  Resultados guardados en: {output_path}")
    print("=" * 70)

    # Resumen final
    scores = [r["evaluation"].get("score", 0) for r in evaluated_results]
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\n  Chunks generados : {len(chunks)}")
    print(f"  Consultas ejecutadas : {len(results)}")
    print(f"  Score promedio : {avg_score:.1f}/10")
    print()


if __name__ == "__main__":
    main()

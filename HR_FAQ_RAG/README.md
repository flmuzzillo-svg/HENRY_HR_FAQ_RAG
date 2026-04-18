# Sistema RAG de Preguntas Frecuentes para Recursos Humanos

## Descripcion del Proyecto

Este proyecto implementa un sistema de Preguntas Frecuentes (FAQs) basado en la arquitectura RAG (Retrieval-Augmented Generation) para el departamento de Recursos Humanos de una empresa SaaS. El sistema procesa un documento extenso de politicas de HR, lo divide en fragmentos semanticos, genera embeddings vectoriales con OpenAI, los almacena en una base de datos vectorial ChromaDB, y permite realizar consultas en lenguaje natural. Las respuestas se generan exclusivamente a partir del contexto recuperado, garantizando precision y trazabilidad. Adicionalmente, incluye un modulo de evaluacion automatica que actua como juez LLM para medir la calidad de cada respuesta.

## Arquitectura

```
Pregunta del Usuario
        |
        v
  [1. Retrieve] -- Convierte a embedding --> Busca en ChromaDB --> K chunks
        |
        v
  [2. Generate] -- Prompt estricto + contexto --> OpenAI GPT --> Respuesta
        |
        v
  [3. Evaluate] -- Triada (pregunta, respuesta, chunks) --> Juez LLM --> Score
```

## Stack Tecnologico

| Componente | Tecnologia |
|---|---|
| Embeddings | OpenAI `text-embedding-3-large` |
| Generacion | OpenAI GPT (`gpt-4o-mini`) |
| Vector Store | ChromaDB (local persistente) |
| Framework | LangChain |
| Chunking | `RecursiveCharacterTextSplitter` |

## Estructura del Proyecto

```
HR_FAQ_RAG/
├── data/
│   └── faq_document.txt          # Documento de politicas HR (1000+ palabras)
├── src/
│   ├── build_index.py            # Pipeline: Load -> Chunk -> Embed -> Store
│   ├── query.py                  # Pipeline: Retrieve -> Generate -> JSON
│   └── evaluator.py              # Juez LLM (score 0-10 + justificacion)
├── outputs/
│   └── sample_queries.json       # 3 ejemplos reales con evaluacion
├── chroma_db/                    # (generado) Persistencia de ChromaDB
├── run_all.py                    # Ejecuta el pipeline completo
├── requirements.txt              # Dependencias del proyecto
├── .env.example                  # Template de variables de entorno
└── README.md                     # Esta documentacion
```

## Setup — Instalacion

### Paso 1: Clonar el repositorio e instalar dependencias

```bash
cd HR_FAQ_RAG
pip install -r requirements.txt
```

### Paso 2: Configurar variables de entorno

```bash
cp .env.example .env
# Edita .env y agrega tu clave de OpenAI:
# OPENAI_API_KEY=sk-proj-tu-clave-aqui
```

### Paso 3: Ejecutar el pipeline completo

```bash
python run_all.py
```

Esto ejecutara automaticamente:
1. Construccion del indice vectorial (indexa el documento en ChromaDB)
2. 3 consultas de ejemplo contra el sistema RAG
3. Evaluacion de calidad con el juez LLM
4. Generacion de `outputs/sample_queries.json`

### Ejecucion individual de modulos

```bash
# Solo construir el indice
python src/build_index.py

# Solo consultar (pregunta por defecto)
python src/query.py

# Consulta personalizada
python src/query.py "Cuantos dias de vacaciones tengo?"

# Solo evaluar (con datos de ejemplo)
python src/evaluator.py
```

## Justificacion de la Estrategia de Chunking

Se utiliza `RecursiveCharacterTextSplitter` con los siguientes parametros:
- **chunk_size = 500 caracteres**: En espanol, 500 caracteres equivalen a aproximadamente 380 tokens (ratio ~1.3 chars/token). Esto situa cada chunk dentro del rango requerido de 50-500 tokens, asegurando que cada fragmento contenga suficiente contexto semantico sin ser demasiado extenso.
- **chunk_overlap = 100 caracteres**: El solapamiento de 100 caracteres (~75 tokens) evita que se corten ideas a mitad de parrafo. Esto es especialmente importante en documentos de politicas donde una regla puede depender de informacion mencionada unas lineas antes.
- **Separadores jerarquicos**: Se priorizan cortes en `\n\n` (parrafos), `\n` (lineas), `. ` (oraciones), etc., para respetar los limites naturales del texto.

## Por que ChromaDB + OpenAI?

- **ChromaDB**: Base de datos vectorial ligera, de codigo abierto, con persistencia local. Ideal para prototipos y sistemas autocontenidos que no requieren infraestructura externa. Su API es sencilla y se integra nativamente con LangChain.
- **OpenAI text-embedding-3-large**: Modelo de embeddings de ultima generacion con excelente rendimiento en busqueda semantica multilingue, incluyendo espanol. Genera vectores de alta calidad que permiten recuperacion precisa.
- **Beneficios del patron RAG**: Al anclar las respuestas del LLM al contexto recuperado del documento, se eliminan las alucinaciones, se garantiza la trazabilidad de la informacion, y se permite actualizar el conocimiento del sistema simplemente actualizando el documento fuente sin reentrenar ningun modelo.

## Metricas de Calidad

- El documento se procesa al **100%** (todas las secciones son indexadas).
- Se generan **20+ chunks** semanticos a partir del documento.
- Cada chunk tiene entre **50 y 500 tokens**.
- El sistema implementa un flujo RAG explicito en **dos pasos**: recuperacion vectorial + generacion contextual.
- El evaluador LLM valida la calidad de cada respuesta con scores de **0 a 10**.

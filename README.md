# Chatbot de Documentos Empresariales con IA 🤖📂

Asistente inteligente basado en IA que permite consultar información contenida en documentos empresariales y obtener respuestas precisas con contexto real extraído de los archivos cargados.

Soporta múltiples formatos y utiliza una arquitectura avanzada de búsqueda híbrida + LLM, optimizada para entornos empresariales con grandes volúmenes de documentación.

## ✨ Características principales

- 📑 Soporte completo de formatos: PDF, DOCX, TXT, Markdown (.md), CSV, PowerPoint (.pptx), Excel (.xlsx) y HTML.
- 📂 Carga múltiple de documentos: Permite indexar varios archivos en una sola sesión y gestionarlos dinámicamente.
- 🧠 Búsqueda híbrida avanzada (RAG moderno). Combina:
  - FAISS (búsqueda semántica)
  - BM25 (búsqueda léxica)
  - Reciprocal Rank Fusion (RRF)
- 🎯 Re-ranker inteligente: Mejora la precisión de resultados usando Cross-Encoder (ms-marco-MiniLM-L-6-v2).
- 🌍 Soporte multilingüe automático: Detecta el idioma de la pregunta y responde en consecuencia (ES / EN / FR / DE / IT / PT…).
- ⚡ Optimización de rendimiento
  - Caché de embeddings por archivo
  - Carga paralela de documentos
  - Chunking basado en oraciones con solapamiento
- 💬 Generación de respuestas con IA (LLM local). Respuestas generadas con modelos como:
  - google/flan-t5-base
  - (opcional) microsoft/Phi-3-mini-4k-instruct
- 📊 Historial inteligente persistente
  - Guardado en SQLite
  - Búsqueda por palabras clave
  - Trazabilidad de respuestas (archivo + página + score)
- 📈 Analítica integrada
  - Estadísticas de uso
  - Distribución de confianza
  - Términos más consultados
- 📤 Exportación de historial. Exporta conversaciones en:
  - JSON
  - CSV
  - Markdown
  - TXT
  - HTML
- 🧹 Gestión dinámica de documentos
  - Añadir / eliminar documentos sin reiniciar el sistema
  - Reindexado automático
- 🌐 Interfaz web moderna con Gradio. Organizada en pestañas:
  - Chat
  - Documentos
  - Historial
  - Estadísticas
- 🧠 Arquitectura del sistema. El flujo de procesamiento es el siguiente:
  - Carga de documentos
  - Chunking semántico por oraciones
  - Generación de embeddings multilingües
  - Indexación dual (FAISS + BM25)
  - Fusión de resultados (RRF)
  - Re-ranking con Cross-Encoder
  - Generación de respuesta con LLM local
  - Citación de fuente y score de confianza
## 🛠️ Tecnologías utilizadas
- Python 3.x
- Gradio → interfaz web interactiva
- FAISS → búsqueda vectorial eficiente
- rank-bm25 → búsqueda léxica
- SentenceTransformers → embeddings multilingües
- Cross-Encoder (Hugging Face) → re-ranking de resultados
- Transformers → generación de texto (Flan-T5 / Phi-3)
- PyPDF2 → lectura de PDFs
- python-docx → lectura de DOCX
- python-pptx → lectura de PPTX
- openpyxl → lectura de XLSX
- BeautifulSoup → parsing HTML
- NLTK → segmentación de texto
- SQLite3 → persistencia de historial
- NumPy / Pandas → procesamiento de datos
- Matplotlib → visualización de estadísticas
## 🚀 Instalación

- Instala las dependencias:

  pip install -r requirements.txt

## ▶️ Ejecución

  python app.py

  Luego abre el enlace generado por Gradio en tu navegador.

## 📂 Formatos soportados
- PDF (.pdf)
- Word (.docx)
- PowerPoint (.pptx)
- Excel (.xlsx)
- Texto (.txt)
- Markdown (.md)
- CSV (.csv)
- HTML (.html, .htm)
## 💡 Casos de uso ideales

- 🏢 Empresas que necesitan consultar documentación interna rápidamente
- 📚 Análisis de documentos técnicos o académicos
- 📑 Búsqueda inteligente en contratos, informes y manuales
- 🧠 Sistemas de soporte interno basado en conocimiento

## ⚠️ Notas importantes
- Funciona mejor con texto digital (no escaneado).
- La calidad depende del contenido de los documentos.
- El sistema prioriza respuestas basadas en evidencia del contexto.
- Si la confianza es baja, el sistema evita inventar respuestas.

## 📄 Licencia
Este proyecto se distribuye bajo una **licencia propietaria con acceso al código (source-available)**.

El código fuente se pone a disposición únicamente para fines de **visualización, evaluación y aprendizaje**.

❌ No está permitido copiar, modificar, redistribuir, sublicenciar, ni crear obras derivadas del software o de su código fuente sin autorización escrita expresa del titular de los derechos.

❌ El uso comercial del software, incluyendo su oferta como servicio (SaaS), su integración en productos comerciales o su uso en entornos de producción, requiere un **acuerdo de licencia comercial independiente**.

📌 El texto **legalmente vinculante** de la licencia es la versión en inglés incluida en el archivo `LICENSE`. 

Se proporciona una traducción al español en `LICENSE_ES.md` únicamente con fines informativos. En caso de discrepancia, prevalece la versión en inglés.


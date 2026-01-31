# Chatbot de Documentos Empresariales con IA 🤖📂
Asistente inteligente basado en IA que responde preguntas sobre documentos cargados (PDF, DOCX, TXT, MD, CSV, PPTX, ODT).

Permite subir múltiples archivos, procesarlos de manera semántica y realizar consultas con respuestas contextuales, mostrando información relevante directamente desde los textos, con historial, búsqueda y exportación.

## ✨ Características principales

- 📑 Soporte completo de formatos: PDF, DOCX, TXT, Markdown (.md), CSV, PowerPoint (.pptx) y OpenDocument (.odt).

- 📂 Carga múltiple de documentos en una sola sesión.

- 🧹 Fragmentación semántica de textos para preservar contexto y mejorar la calidad de las respuestas.

- 🧠 Incrustaciones semánticas avanzadas con SentenceTransformers (all-mpnet-base-v2).

- ⚡ Búsqueda vectorial eficiente con FAISS, optimizada con caché para consultas rápidas.

- 🎯 Respuestas contextuales robustas usando el modelo QA deepset/roberta-base-squad2.

- 💬 Interfaz tipo chat con historial completo, incluyendo timestamps y exportación en JSON o Markdown.

- 🔍 Búsqueda en historial por palabras clave para encontrar conversaciones pasadas rápidamente.

- 🌐 Interfaz web accesible desarrollada con Gradio, dividida en secciones para preguntas, historial y exportación.

## 🛠️ Tecnologías usadas

- Python 3.x

- Gradio → Interfaz web interactiva

- SentenceTransformers → Embeddings semánticos (all-mpnet-base-v2)

- Transformers (Hugging Face) → Modelo de QA (roberta-base-squad2)

- FAISS → Búsqueda vectorial rápida

- PyPDF2 → Lectura de PDFs

- python-docx → Lectura de DOCX

- python-pptx → Lectura de PPTX

- odfpy → Lectura de ODT

- NLTK → Tokenización semántica de oraciones

- NumPy → Cálculo numérico

## 🚀 Uso

Instala dependencias:

pip install -r requirements.txt


Ejecuta la aplicación:

python app.py


Abre la interfaz web de Gradio en tu navegador.

Sube tus documentos (PDF, DOCX, TXT, MD, CSV, PPTX, ODT) y comienza a hacer preguntas.

## 🎯 Ideal para

- 🏢 Empresas que quieren consultar documentos internos de forma rápida y segura.

- 📚 Investigadores y estudiantes que necesitan extraer información puntual de múltiples fuentes.

- 🧠 Proyectos empresariales que requieren un asistente inteligente para manejar documentación sensible con historial y búsquedas.

## 💡 Notas
- Se recomienda usar documentos con texto digital (no escaneados como imagen).

- El rendimiento puede variar según el tamaño y tipo de los documentos cargados.

- Las consultas son semánticas, no solo coincidencias literales.

## 📄 Licencia
Este proyecto se distribuye bajo una **licencia propietaria con acceso al código (source-available)**.

El código fuente se pone a disposición únicamente para fines de **visualización, evaluación y aprendizaje**.

❌ No está permitido copiar, modificar, redistribuir, sublicenciar, ni crear obras derivadas del software o de su código fuente sin autorización escrita expresa del titular de los derechos.

❌ El uso comercial del software, incluyendo su oferta como servicio (SaaS), su integración en productos comerciales o su uso en entornos de producción, requiere un **acuerdo de licencia comercial independiente**.

📌 El texto **legalmente vinculante** de la licencia es la versión en inglés incluida en el archivo `LICENSE`. 

Se proporciona una traducción al español en `LICENSE_ES.md` únicamente con fines informativos. En caso de discrepancia, prevalece la versión en inglés.


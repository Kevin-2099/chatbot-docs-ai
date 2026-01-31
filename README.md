# Chatbot de Documentos Empresariales con IA 🤖📂
Asistente inteligente basado en IA que responde preguntas sobre documentos cargados (PDF, DOCX, TXT, MD, CSV, PPTX, ODT).

Permite subir múltiples archivos, procesarlos y realizar consultas con respuestas contextuales, mostrando la información relevante directamente desde el texto.

## ✨ Características principales

📑 Soporte ampliado para múltiples formatos: PDF, DOCX, TXT, Markdown (.md), CSV, PowerPoint (.pptx) y OpenDocument (.odt).

📂 Carga de múltiples documentos en una sola sesión.

🧹 Limpieza y fragmentación automática del texto con solapamiento para preservar el contexto.

🧠 Incrustaciones semánticas avanzadas con SentenceTransformers (all-mpnet-base-v2).

⚡ Búsqueda vectorial eficiente y precisa con FAISS.

🎯 Respuestas contextuales mejoradas con el modelo QA deepset/roberta-base-squad2 de Hugging Face.

💬 Interfaz tipo chat con historial de conversación completo.

🌐 Interfaz web accesible creada con Gradio.

🔍 Análisis contextual mejorado: combina fragmentos relevantes de varios documentos para dar respuestas más completas.

## 🛠️ Tecnologías usadas

Python 3.x

Gradio → Interfaz de usuario web.

SentenceTransformers → Embeddings semánticos (all-mpnet-base-v2).

Transformers (Hugging Face) → Modelo de preguntas y respuestas (roberta-base-squad2).

FAISS → Búsqueda vectorial rápida.

PyPDF2 → Lectura de PDFs.

python-docx → Lectura de archivos DOCX.

python-pptx → Lectura de presentaciones PPTX.

odfpy → Lectura de documentos ODT.

NumPy → Cálculo numérico.

## 🚀 Uso

Instala dependencias:

pip install -r requirements.txt


Ejecuta la aplicación:

python app.py


Abre la interfaz web de Gradio en tu navegador.

Sube tus documentos (PDF, DOCX, TXT, MD, CSV, PPTX, ODT) y comienza a hacer preguntas.

## 🎯 Ideal para

🏢 Empresas que quieren consultar documentos internos de forma rápida y segura.

📚 Investigadores y estudiantes que necesitan extraer información puntual de múltiples fuentes.

🧠 Proyectos empresariales que requieren un asistente inteligente para manejar documentación sensible.

## 💡 Notas
Se recomienda usar documentos con texto digital (no escaneados como imagen).

El rendimiento puede variar según el tamaño y tipo de los documentos cargados.

Admite consultas semánticas, no solo coincidencias literales.

## 📄 Licencia
Este proyecto se distribuye bajo una **licencia propietaria con acceso al código (source-available)**.

El código fuente se pone a disposición únicamente para fines de **visualización, evaluación y aprendizaje**.

❌ No está permitido copiar, modificar, redistribuir, sublicenciar, ni crear obras derivadas del software o de su código fuente sin autorización escrita expresa del titular de los derechos.

❌ El uso comercial del software, incluyendo su oferta como servicio (SaaS), su integración en productos comerciales o su uso en entornos de producción, requiere un **acuerdo de licencia comercial independiente**.

📌 El texto **legalmente vinculante** de la licencia es la versión en inglés incluida en el archivo `LICENSE`. 

Se proporciona una traducción al español en `LICENSE_ES.md` únicamente con fines informativos. En caso de discrepancia, prevalece la versión en inglés.


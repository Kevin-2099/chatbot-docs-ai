# Chatbot de Documentos Empresariales con IA 🤖📂

Asistente inteligente basado en IA que responde preguntas sobre documentos cargados (PDF, DOCX, TXT, MD, CSV). Permite subir múltiples archivos, procesarlos y realizar consultas con respuestas contextuales, resaltando la información relevante directamente en el texto.

## ✨ Características principales

📑 Soporte para múltiples formatos: PDF, DOCX, TXT, Markdown (.md) y CSV.

📂 Carga de múltiples documentos en una sola sesión.

🧹 Limpieza y fragmentación automática del texto con solapamiento para mejorar contexto.

🧠 Embeddings semánticos con SentenceTransformers (all-MiniLM-L6-v2).

⚡ Búsqueda rápida y precisa con FAISS.

🎯 Respuestas contextuales con modelo DistilBERT QA de Hugging Face.

🔍 Resaltado automático de la parte del documento usada para responder.

💬 Interfaz de chat con historial, que guarda la conversación completa.

🌐 Interfaz web accesible con Gradio.

## 🛠️ Tecnologías usadas

Python 3.x

Gradio (interfaz de usuario)

SentenceTransformers (embeddings semánticos)

FAISS (búsqueda vectorial eficiente)

Transformers (Hugging Face) para Question Answering

PyPDF2 para extracción de texto en PDF

python-docx para extracción de texto en DOCX

## 🚀 Uso

Instala dependencias:

pip install -r requirements.txt


Ejecuta la aplicación:

python app.py


Abre la interfaz web de Gradio en tu navegador.

Sube tus documentos (PDF, DOCX, TXT, MD o CSV) y comienza a hacer preguntas.

## 🎯 Ideal para

Empresas que quieren consultar documentos internos de manera centralizada.

Investigadores y estudiantes que trabajan con múltiples documentos y necesitan encontrar información puntual.

Proyectos empresariales que requieren un asistente inteligente y privado para documentos sensibles.

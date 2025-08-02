import os
import gradio as gr
import faiss
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ======================
# CONFIGURACIÓN DE MODELOS
# ======================

modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")
modelo_qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# ======================
# FUNCIÓN: CARGAR DOCUMENTOS
# ======================

def cargar_documentos_desde_archivos(archivos):
    textos = []
    for archivo in archivos:
        nombre = archivo.name
        if nombre.endswith(".pdf"):
            lector = PdfReader(archivo)
            texto = ""
            for pagina in lector.pages:
                texto += pagina.extract_text() or ""
            textos.append(texto)
        elif nombre.endswith(".docx"):
            doc = Document(archivo)
            texto = "\n".join([p.text for p in doc.paragraphs])
            textos.append(texto)
    return textos

# ======================
# EMBEDDINGS Y FAISS
# ======================

def generar_embeddings(textos):
    return modelo_embeddings.encode(textos)

def construir_faiss(embeds):
    dim = embeds.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeds)
    return index

# ======================
# SISTEMA DE RESPUESTA
# ======================

def responder(pregunta, index, textos):
    query_embed = modelo_embeddings.encode([pregunta])
    D, I = index.search(query_embed, k=1)
    contexto = textos[I[0][0]]
    respuesta = modelo_qa(question=pregunta, context=contexto)
    return respuesta["answer"]

# ======================
# INTERFAZ GRADIO
# ======================

def crear_chatbot_con_documentos(archivos, pregunta):
    if not archivos:
        return "Por favor, sube al menos un documento PDF o DOCX."
    textos = cargar_documentos_desde_archivos(archivos)
    embeddings = generar_embeddings(textos)
    index = construir_faiss(np.array(embeddings))
    return responder(pregunta, index, textos)

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Chatbot de Empresa con documentos PDF/DOCX")
    archivos_subidos = gr.File(label="Sube documentos", file_types=[".pdf", ".docx"], file_count="multiple")
    entrada_pregunta = gr.Textbox(label="Haz una pregunta")
    boton = gr.Button("Preguntar")
    salida = gr.Textbox(label="Respuesta")

    boton.click(fn=crear_chatbot_con_documentos,
                inputs=[archivos_subidos, entrada_pregunta],
                outputs=salida)

if __name__ == "__main__":
    demo.launch()

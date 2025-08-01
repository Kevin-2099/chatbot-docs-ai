import os
import gradio as gr
import faiss
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document

# ======================
# CONFIGURACIÓN Y MODELOS
# ======================

modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")
modelo_qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# ======================
# CARGAR DOCUMENTOS DESDE ARCHIVOS SUBIDOS
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
# INTERFAZ GRADIO CON SUBIDA DE ARCHIVOS
# ======================

def crear_chatbot_con_documentos(archivos, pregunta):
    if not archivos:
        return "Por favor, sube al menos un documento PDF o DOCX primero."

    textos = cargar_documentos_desde_archivos(archivos)
    embeddings = generar_embeddings(textos)
    index = construir_faiss(embeddings)
    respuesta = responder(pregunta, index, textos)
    return respuesta

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Chatbot de Empresa con carga de documentos")
    
    archivos_subidos = gr.File(label="Sube documentos PDF o DOCX", file_types=['.pdf', '.docx'], file_count="multiple")
    pregunta_input = gr.Textbox(label="Haz una pregunta sobre los documentos")
    boton_preguntar = gr.Button("Preguntar")
    salida = gr.Textbox(label="Respuesta")

    def manejar_pregunta(archivos, pregunta):
        return crear_chatbot_con_documentos(archivos, pregunta)

    boton_preguntar.click(fn=manejar_pregunta, inputs=[archivos_subidos, pregunta_input], outputs=salida)

if __name__ == "__main__":
    demo.launch()

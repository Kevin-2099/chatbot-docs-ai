# ======================================
# CHATBOT DE DOCUMENTOS EMPRESARIALES v3.1
# ======================================
import os
import gradio as gr
import faiss
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
import csv
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ======================
# CONFIGURACIÓN DE MODELOS
# ======================
# Embeddings más potentes: modelo MPNet
modelo_embeddings = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Modelo de QA optimizado (más robusto que distilbert)
modelo_qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

# ======================
# LIMPIEZA Y TROZOS DE TEXTO
# ======================
def limpiar_texto(texto: str) -> str:
    return " ".join((texto or "").split()).strip()

def dividir_en_fragmentos(texto: str, max_chars: int = 800, solapamiento: int = 100):
    texto = limpiar_texto(texto)
    if not texto:
        return []
    fragmentos, i = [], 0
    while i < len(texto):
        fragmentos.append(texto[i:i + max_chars])
        i += max(1, max_chars - solapamiento)
    return fragmentos

# ======================
# CARGA DE DOCUMENTOS
# ======================
def leer_pdf(ruta: str) -> str:
    try:
        reader = PdfReader(ruta)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""

def leer_docx(ruta: str) -> str:
    try:
        doc = Document(ruta)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def leer_pptx(ruta: str) -> str:
    try:
        prs = Presentation(ruta)
        texto = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    texto.append(shape.text)
        return "\n".join(texto)
    except Exception:
        return ""

def leer_csv(ruta: str) -> str:
    try:
        with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            return "\n".join([", ".join(row) for row in reader])
    except Exception:
        return ""

def leer_texto(ruta: str) -> str:
    try:
        with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def leer_odt(ruta: str) -> str:
    try:
        from odf.opendocument import load
        from odf.text import P
        doc = load(ruta)
        textos = [str(p) for p in doc.getElementsByType(P)]
        return "\n".join(textos)
    except Exception:
        return ""

def cargar_documentos_desde_archivos(archivos):
    fragmentos_total = []
    for archivo in archivos:
        nombre = getattr(archivo, "name", "")
        nombre_lc = nombre.lower()
        texto = ""
        if nombre_lc.endswith(".pdf"):
            texto = leer_pdf(nombre)
        elif nombre_lc.endswith(".docx"):
            texto = leer_docx(nombre)
        elif nombre_lc.endswith(".pptx"):
            texto = leer_pptx(nombre)
        elif nombre_lc.endswith(".csv"):
            texto = leer_csv(nombre)
        elif nombre_lc.endswith(".odt"):
            texto = leer_odt(nombre)
        elif nombre_lc.endswith((".txt", ".md")):
            texto = leer_texto(nombre)

        fragmentos = dividir_en_fragmentos(texto, max_chars=800, solapamiento=100)
        fragmentos_total.extend(fragmentos)
    return fragmentos_total

# ======================
# EMBEDDINGS Y FAISS
# ======================
def generar_embeddings(textos):
    return modelo_embeddings.encode(textos, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

def construir_faiss(embeds: np.ndarray):
    if embeds.ndim != 2 or embeds.size == 0:
        raise ValueError("No hay incrustaciones para indexar.")
    dim = embeds.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP = producto interno (mejor para embeddings normalizados)
    index.add(embeds)
    return index

# ======================
# SISTEMA DE RESPUESTA
# ======================
def respondedor(pregunta: str, index, textos, k: int = 4):
    if not pregunta.strip():
        return "Por favor, escribe una pregunta.", ""
    if not textos:
        return "No se ha podido extraer texto válido de los archivos.", ""

    k_seguro = max(1, min(k, len(textos)))
    consulta_embed = modelo_embeddings.encode([pregunta], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = index.search(consulta_embed, k=k_seguro)
    candidatos = [textos[i] for i in I[0] if 0 <= i < len(textos)]
    contexto = " ... ".join(candidatos).strip()

    if not contexto:
        return "No encontré contexto relevante en los documentos.", ""

    try:
        resultado = modelo_qa(question=pregunta, context=contexto)
        respuesta = (resultado.get("answer") or "").strip()

        if not respuesta:
            respuesta = "No estoy seguro; intenta reformular la pregunta o cargar documentos más específicos."

        texto_respuesta = f"**Respuesta:** {respuesta}\n\n**Contexto relacionado:** {contexto[:1200]}..."
        return texto_respuesta, respuesta
    except Exception as e:
        return f"Ocurrió un error al generar la respuesta: {e}", ""

# ======================
# INTERFAZ GRADIO
# ======================
def crear_chatbot_con_documentos(archivos, pregunta, historial):
    historial = historial or []
    if not archivos:
        nuevo_mensaje = ("Usuario: " + (pregunta or ""), "Por favor, sube al menos un documento válido.")
        historial.append(nuevo_mensaje)
        return historial, historial

    textos = cargar_documentos_desde_archivos(archivos)
    if not textos:
        nuevo_mensaje = ("Usuario: " + (pregunta or ""), "No pude leer contenido de los archivos subidos.")
        historial.append(nuevo_mensaje)
        return historial, historial

    try:
        embeds = generar_embeddings(textos)
        index = construir_faiss(embeds)
    except Exception as e:
        nuevo_mensaje = ("Usuario: " + (pregunta or ""), f"Error al construir el índice: {e}")
        historial.append(nuevo_mensaje)
        return historial, historial

    salida, _ = respondedor(pregunta, index, textos, k=4)
    historial.append(("Usuario: " + (pregunta or ""), salida))
    return historial, historial

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Chatbot Empresarial con Documentos (PDF, DOCX, TXT, MD, CSV, PPTX, ODT)")
    archivos_subidos = gr.File(
        label="Sube documentos",
        file_types=[".pdf", ".docx", ".txt", ".md", ".csv", ".pptx", ".odt"],
        file_count="multiple"
    )
    chatbot = gr.Chatbot(label="Historial de conversación")
    entrada_pregunta = gr.Textbox(label="Haz una pregunta", placeholder="Escribe tu pregunta aquí...")
    boton = gr.Button("Preguntar")
    limpiar = gr.ClearButton([chatbot, entrada_pregunta])
    estado_historial = gr.State([])

    boton.click(
        fn=crear_chatbot_con_documentos,
        inputs=[archivos_subidos, entrada_pregunta, estado_historial],
        outputs=[chatbot, estado_historial]
    )

if __name__ == "__main__":
    demo.launch(share=True)

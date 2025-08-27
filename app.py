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
# LIMPIEZA Y CHUNKS
# ======================
def limpiar_texto(texto: str) -> str:
    return " ".join((texto or "").split()).strip()

def dividir_en_fragmentos(texto: str, max_chars: int = 500, solapamiento: int = 50):
    texto = limpiar_texto(texto)
    if not texto:
        return []
    fragmentos = []
    i = 0
    while i < len(texto):
        fragmentos.append(texto[i:i + max_chars])
        i += max(1, max_chars - solapamiento)
    return fragmentos

# ======================
# CARGA DE DOCUMENTOS
# ======================
def leer_pdf(path: str) -> str:
    out = []
    try:
        reader = PdfReader(path)
        for p in reader.pages:
            out.append(p.extract_text() or "")
    except Exception:
        return ""
    return "\n".join(out)

def leer_docx(path: str) -> str:
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def leer_texto(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")
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
        elif nombre_lc.endswith((".txt", ".md", ".csv")):
            texto = leer_texto(nombre)
        # dividir en fragmentos con solapamiento
        fragmentos = dividir_en_fragmentos(texto, max_chars=500, solapamiento=50)
        fragmentos_total.extend(fragmentos)
    return fragmentos_total

# ======================
# EMBEDDINGS Y FAISS
# ======================
def generar_embeddings(textos):
    # convert_to_numpy -> float32, más estable para FAISS
    return modelo_embeddings.encode(textos, convert_to_numpy=True, normalize_embeddings=False)

def construir_faiss(embeds: np.ndarray):
    if embeds.ndim != 2 or embeds.size == 0:
        raise ValueError("No hay embeddings para indexar.")
    if embeds.dtype != np.float32:
        embeds = embeds.astype(np.float32)
    dim = embeds.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeds)
    return index

# ======================
# SISTEMA DE RESPUESTA
# ======================
def responder(pregunta: str, index, textos, k: int = 3):
    if not pregunta or not pregunta.strip():
        return "Por favor, escribe una pregunta.", ""
    if not textos:
        return "No se ha podido extraer texto válido de los archivos.", ""

    # Ajustar k para no exceder el tamaño del índice
    k_seguro = max(1, min(k, len(textos)))

    query_embed = modelo_embeddings.encode([pregunta], convert_to_numpy=True)
    if query_embed.dtype != np.float32:
        query_embed = query_embed.astype(np.float32)

    D, I = index.search(query_embed, k=k_seguro)
    candidatos = [textos[i] for i in I[0] if 0 <= i < len(textos)]
    contexto = " ... ".join(candidatos).strip()
    if not contexto:
        return "No encontré contexto relevante en los documentos.", ""

    try:
        resultado = modelo_qa(question=pregunta, context=contexto)
        answer = (resultado.get("answer") or "").strip()
        start = resultado.get("start", -1)
        end = resultado.get("end", -1)

        # Resaltado simple si vienen índices válidos
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(contexto):
            ctx_resaltado = contexto[:start] + "**" + contexto[start:end] + "**" + contexto[end:]
        else:
            # fallback: marcar la primera aparición de la respuesta si existe
            if answer and answer in contexto:
                ctx_resaltado = contexto.replace(answer, f"**{answer}**", 1)
            else:
                ctx_resaltado = contexto

        if not answer:
            answer = "No estoy seguro; intenta reformular la pregunta o carga documentos más específicos."

        texto_respuesta = f"**Respuesta:** {answer}\n\n**Contexto:** {ctx_resaltado}"
        return texto_respuesta, answer

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

    embeds = generar_embeddings(textos)
    try:
        index = construir_faiss(embeds)
    except Exception as e:
        nuevo_mensaje = ("Usuario: " + (pregunta or ""), f"Error construyendo el índice: {e}")
        historial.append(nuevo_mensaje)
        return historial, historial

    salida, _ = responder(pregunta, index, textos, k=3)
    historial.append(("Usuario: " + (pregunta or ""), salida))
    return historial, historial

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Chatbot de Empresa con documentos (PDF, DOCX, TXT, MD, CSV)")
    archivos_subidos = gr.File(
        label="Sube documentos",
        file_types=[".pdf", ".docx", ".txt", ".md", ".csv"],
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
    # share=True si quieres exponer temporalmente en red local/externa
    demo.launch()

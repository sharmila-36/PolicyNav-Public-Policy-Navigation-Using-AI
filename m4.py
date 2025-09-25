# app.py
import streamlit as st
import pandas as pd
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from io import BytesIO

st.set_page_config(page_title="File Upload with OCR, Chunking & Ollama", page_icon="📂", layout="wide")
st.title("File Upload with OCR, Chunking & Ollama")
st.caption("Streamed responses from local Ollama + retrieval-augmented answers.")

# --- Session state defaults ---
st.session_state.setdefault("chat", [])  # list of tuples (role, text)
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "vectorizer" not in st.session_state:
    st.session_state["vectorizer"] = None
if "tfidf_matrix" not in st.session_state:
    st.session_state["tfidf_matrix"] = None
if "last_json_filename" not in st.session_state:
    st.session_state["last_json_filename"] = None

# ---------- helpers ----------
def extract_pdf_text(file_bytes: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except Exception:
        # fallback to OCR
        pass

    if not text.strip():
        try:
            images = convert_from_bytes(file_bytes)
            for img in images:
                text += pytesseract.image_to_string(img)
        except Exception:
            pass
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    if not text:
        return []
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max(1, chunk_size - overlap)
    return chunks

def build_tfidf(chunks):
    if not chunks:
        return None, None
    vectorizer = TfidfVectorizer().fit(chunks)
    tfidf_matrix = vectorizer.transform(chunks)
    return vectorizer, tfidf_matrix

def retrieve_top_k(query, chunks, vectorizer, tfidf_matrix, k=3):
    if not chunks or vectorizer is None or tfidf_matrix is None:
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:k]
    results = [{"id": int(i)+1, "score": float(sims[i]), "content": chunks[i]} for i in top_idx if sims[i] > 0]
    return results

def call_ollama_stream(prompt: str, model: str = "llama3.1:8b", timeout: int = 300):
    """
    Generator that yields tokens/partial text from Ollama server.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                # decode if it's bytes
                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8", errors="ignore")
                else:
                    line = str(raw_line)

                line = line.strip()
                if not line:
                    continue

                # Some Ollama servers prefix with 'data: ' for SSE
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()

                # Try to parse JSON
                try:
                    data = json.loads(line)
                except Exception:
                    yield line
                    continue

                token = None
                if isinstance(data, dict):
                    for k in ("response", "text", "delta", "token", "content"):
                        if k in data and data[k]:
                            token = data[k]
                            break
                    if token is None and "result" in data and isinstance(data["result"], str):
                        token = data["result"]

                if token is None:
                    continue

                if isinstance(token, dict):
                    token = token.get("content") or str(token)

                yield token
    except requests.exceptions.ReadTimeout:
        yield f"\n\n[Ollama timeout after {timeout}s]"
    except Exception as e:
        yield f"\n\n[Ollama error: {e}]"

def call_ollama_blocking(prompt: str, model: str = "llama3.1:8b", timeout: int = 300):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response") or data.get("text") or str(data)
    except Exception as e:
        return f"[Ollama error: {e}]"

# ---------- UI: sidebar ----------
st.sidebar.header("Settings")
chunk_size = st.sidebar.number_input("Chunk size (chars)", value=1000, min_value=200, step=100)
overlap = st.sidebar.number_input("Overlap (chars)", value=200, min_value=0, step=50)
top_k = st.sidebar.number_input("Top-K chunks to retrieve", value=3, min_value=1, max_value=10)
model_choice = st.sidebar.selectbox("Ollama model", options=["llama3.1:8b", "llama3:latest", "mistral", "ggml-alpaca"], index=0)
timeout = st.sidebar.number_input("Ollama timeout (seconds)", value=300, min_value=30)
use_streaming = st.sidebar.checkbox("Use streaming from Ollama (recommended)", value=True)
download_json = st.sidebar.checkbox("Provide JSON download button", value=True)

# ---------- main: file upload ----------
with st.expander("Upload file (txt / csv / xlsx / pdf)"):
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv", "xlsx", "xls", "pdf"])

if uploaded_file:
    st.success(f"{uploaded_file.name} uploaded")
    extracted_text = ""

    # read depending on type
    try:
        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith(".txt"):
            extracted_text = uploaded_file.read().decode(errors="ignore")
            st.text_area("Preview (first 2000 chars)", extracted_text[:2000], height=200)
        elif uploaded_file.type == "text/csv" or uploaded_file.name.endswith(".csv"):
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            extracted_text = df.to_string()
        elif "spreadsheet" in uploaded_file.type or uploaded_file.name.endswith((".xls", ".xlsx")):
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            st.dataframe(df.head())
            extracted_text = df.to_string()
        elif uploaded_file.type == "application/pdf" or uploaded_file.name.endswith(".pdf"):
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            extracted_text = extract_pdf_text(file_bytes)
            st.text_area("Preview (first 2000 chars)", extracted_text[:2000], height=200)
        else:
            # fallback
            uploaded_file.seek(0)
            try:
                extracted_text = uploaded_file.read().decode(errors="ignore")
                st.text_area("Preview (first 2000 chars)", extracted_text[:2000], height=200)
            except Exception:
                st.warning("Could not preview file type. We'll still attempt to process it.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        extracted_text = ""

    # chunking and indexing
    chunks = chunk_text(extracted_text, chunk_size=chunk_size, overlap=overlap)
    st.session_state["chunks"] = chunks
    vectorizer, tfidf_matrix = build_tfidf(chunks)
    st.session_state["vectorizer"] = vectorizer
    st.session_state["tfidf_matrix"] = tfidf_matrix

    # save chunks to JSON file (in-memory)
    json_data = {"file_name": uploaded_file.name, "chunks": [{"id": i+1, "content": c} for i, c in enumerate(chunks)]}
    base_name = os.path.splitext(uploaded_file.name)[0]
    json_file_name = f"{base_name}_chunks.json"
    json_bytes = json.dumps(json_data, ensure_ascii=False, indent=2).encode("utf-8")
    st.session_state["last_json_filename"] = json_file_name

    st.success(f"Extracted, chunked (total {len(chunks)} chunks).")
    if len(chunks) > 0:
        st.subheader("Sample chunk (first)")
        st.code(chunks[0][:1000] + ("..." if len(chunks[0]) > 1000 else ""))

    if download_json:
        st.download_button("Download chunks JSON", data=json_bytes, file_name=json_file_name, mime="application/json")

# ---------- Chat / Q&A ----------
st.markdown("---")
st.subheader("Ask a question (retrieval + Ollama)")
user_q = st.chat_input("Ask something about the document...")

if user_q:
    st.session_state["chat"].append(("You", user_q))

    # Retrieval
    results = retrieve_top_k(user_q, st.session_state["chunks"], st.session_state["vectorizer"], st.session_state["tfidf_matrix"], k=top_k)
    if not results:
        context_text = ""
    else:
        context_text = "\n\n---\n\n".join([r["content"] for r in results])

    system_prompt = (
        "You are an assistant. Answer concisely using ONLY the information in the provided CONTEXT. "
        "If the answer is not in the context, say 'I don't know from the document.' Keep answers short and factual."
    )
    prompt = f"{system_prompt}\n\nCONTEXT:\n{context_text}\n\nQUESTION: {user_q}\n\nANSWER:"

    # UI placeholder for progressive bot output
    response_placeholder = st.empty()
    response_text = ""
    bot_role_text = ""  # to store final answer

    if use_streaming:
        with st.spinner("Ollama is generating (streaming)..."):
            for token in call_ollama_stream(prompt, model=model_choice, timeout=timeout):
                # token may be partial string or an error message
                response_text += str(token)
                # update UI progressively
                response_placeholder.markdown(f"**Bot:** {response_text}")
        bot_role_text = response_text.strip()
    else:
        # blocking call (non-stream)
        with st.spinner("Ollama is generating..."):
            bot_role_text = call_ollama_blocking(prompt, model=model_choice, timeout=timeout)
            response_placeholder.markdown(f"**Bot:** {bot_role_text}")

    # append to session chat and render full chat
    st.session_state["chat"].append(("Bot", bot_role_text))

# render chat history at bottom (most recent last)
if st.session_state["chat"]:
    st.markdown("### Chat history")
    for role, text in st.session_state["chat"]:
        if role == "You":
            st.markdown(f"**{role}:** {text}")
        else:
            st.markdown(f"**{role}:** {text}")

# PolicyNav - Public Policy Navigation Using AI

PolicyNav is an AI-powered tool designed to help users navigate, analyze, and summarize public policies.  
It integrates **OCR**, **chunking**, and **LLM-based retrieval (Ollama)** for efficient exploration.

## 🚀 Features
- Upload TXT, CSV, XLSX, and PDF documents
- Extract text with OCR (PDFPlumber + Tesseract)
- Chunk large documents for processing
- Query and chat with documents using Ollama (Llama 3.1:8B)
- Export extracted text to JSON

## 🛠️ Tech Stack
- Python
- Streamlit
- pdfplumber, pytesseract, pdf2image
- Ollama (local LLM API)

## 📦 Installation
```bash
git clone https://github.com/<your-username>/PolicyNav-Public-Policy-Navigation-Using-AI.git
cd PolicyNav-Public-Policy-Navigation-Using-AI
pip install -r requirements.txt
streamlit run m4.py

# RAG Flask Web App

This is a Retrieval-Augmented Generation (RAG) web application using Flask, LangChain, Google Gemini API, FAISS, and HuggingFace embeddings.

## Features
- Query your PDF documents using natural language
- Uses Google Gemini (AI Studio) for LLM responses
- Fast vector search with FAISS
- Clean, modern web UI (Tailwind CSS)

## 🟢 Deploy on Render.com (Free Tier)

### 1. Prerequisites
- [Create a free Render.com account](https://render.com)
- [Create a GitHub repo](https://github.com/new) and push your code
- Get a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 2. Files Needed
- `requirements.txt` — Python dependencies
- `Procfile` — tells Render how to run your app (already included)
- `app.py`, `rag_pipeline.py`, etc.

### 3. Deploy Steps
1. **Connect your GitHub repo to Render**
2. **Create a new Web Service**
    - Runtime: Python 3
    - Build Command: `pip install -r requirements.txt`
    - Start Command: `gunicorn app:app`
3. **Set Environment Variables in Render Dashboard:**
    - `GOOGLE_API_KEY` — your Gemini API key
    - (Optional) adjust `PDF_DATA_PATH`, `FAISS_INDEX_PATH` if needed
4. **Deploy!**
    - Render will build and deploy your app
    - Access your app at `https://your-app-name.onrender.com`

### 4. Notes
- Free tier sleeps after inactivity (cold start ~30s)
- If you change your API key, update it in the Render dashboard
- For private PDFs, upload them to `/data/pdfs/` before deploying (or mount persistent storage)

### 5. Local Development
```bash
pip install -r requirements.txt
python app.py
```

---

## Environment Variables
- `GOOGLE_API_KEY` — **required** for Gemini API
- `PDF_DATA_PATH` — path to PDF folder (default: `data/pdfs`)
- `FAISS_INDEX_PATH` — path to FAISS index (default: `data/faiss_index`)

---

## Troubleshooting
- **Model errors:** Check your Google API key and model name
- **Vectorstore errors:** Make sure your FAISS index exists or PDFs are present
- **App crashes:** Check logs in Render dashboard

---

## License
MIT

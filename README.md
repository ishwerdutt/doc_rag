---

## 🐳 Running with Docker

You can run this project in a containerized environment using Docker Compose. This setup uses **Python 3.13 (slim)** and installs all dependencies in a virtual environment inside the container.

### 1. Build and Start the App

```bash
docker compose up --build
```

This will build the image and start the Flask app using Gunicorn, listening on port **5000**.

### 2. Environment Variables

The following environment variables are required for the app to function:
- `GOOGLE_API_KEY` — **required** for Gemini API (set this in your environment or via a `.env` file)
- `PDF_DATA_PATH` — path to PDF folder (default: `data/pdfs`)
- `FAISS_INDEX_PATH` — path to FAISS index (default: `data/faiss_index`)

You can create a `.env` file in the project root and uncomment the `env_file` line in `docker-compose.yml` to load these automatically.

### 3. Ports

- The app is exposed on **port 5000** (mapped to `localhost:5000` by default).

### 4. Data and Configuration

- PDF files should be placed in the `data/pdfs/` directory.
- FAISS index files should be in `data/faiss_index/`.
- You can mount these directories as volumes for development (see commented `volumes` section in `docker-compose.yml`).

### 5. Customization

- The Dockerfile uses a multi-stage build for smaller images and non-root execution for security.
- If you need to change the app port or other settings, update the `EXPOSE` and `CMD` lines in the Dockerfile and the `ports` section in `docker-compose.yml`.

---

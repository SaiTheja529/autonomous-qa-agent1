FROM python:3.11-slim

WORKDIR /app

# System deps for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables (override GEMINI_API_KEY in your host/CI)
ENV GEMINI_MODEL=gemini-2.5-flash \
    EMBEDDING_MODEL=all-MiniLM-L6-v2 \
    BACKEND_URL=http://localhost:8000

EXPOSE 8000 8501

# Run FastAPI backend on 8000 and Streamlit frontend on 8501
CMD bash -lc "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501"


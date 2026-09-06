FROM python:3.11-slim

LABEL maintainer="Mubashir Uddin <mubashiruddin01>" \
      description="AgriDoc-PK: AI Crop Pathology Advisory for Pakistani Farmers" \
      version="1.0.0"

# System dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY core/ core/
COPY knowledge_base/ knowledge_base/
COPY models/ models/
COPY data_store/ data_store/
COPY app.py .

# Build ChromaDB index if not already present
RUN python knowledge_base/build_index.py || true

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]

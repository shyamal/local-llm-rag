FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
# libgomp1 is required by faiss-cpu at runtime (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY benchmarks/ ./benchmarks/
COPY evaluation/ ./evaluation/
COPY smoke_test.py .

# Create directories for persistent data
RUN mkdir -p vector_store data/documents benchmarks/results evaluation/results

# Expose Streamlit port
EXPOSE 8501

# Streamlit config: disable browser auto-open and set server address
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

CMD ["python", "-m", "streamlit", "run", "app/ui.py"]

FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

RUN uv pip install --system --no-cache \
    huggingface-hub==0.27.0 numpy==1.26.4 onnxruntime==1.20.1 \
    openai==1.93.0 pandas==2.2.3 pgvector==0.3.6 psycopg[binary]==3.2.4 \
    pydantic==2.10.4 python-dotenv==1.0.1 streamlit==1.41.1 \
    tokenizers==0.21.0 tqdm==4.67.1 || \
    pip install huggingface-hub==0.27.0 numpy==1.26.4 onnxruntime==1.20.1 \
    openai==1.93.0 pandas==2.2.3 pgvector==0.3.6 psycopg[binary]==3.2.4 \
    pydantic==2.10.4 python-dotenv==1.0.1 streamlit==1.41.1 \
    tokenizers==0.21.0 tqdm==4.67.1

COPY . .

# Download ONNX models at build time so the container is self-contained
RUN python download_model.py && \
    python ingest.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.9-slim

# Define a pasta de trabalho
WORKDIR /app

RUN pip install --default-timeout=1000 --no-cache-dir \
    mlflow \
    boto3 \
    psycopg2-binary \
    pandas \
    pyarrow \
    fastparquet \
    pymysql \
    scikit-learn \
    joblib \
    numpy \
    openpyxl \
    python-dotenv \
    pymilvus \
    ollama \
    sentence-transformers \
    fastapi \
    uvicorn \
    pydantic \ 
    pytest \
    httpx

# Expõe a porta da API
EXPOSE 8000

# Comando embutido no Dockerfile para iniciar o FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import psycopg2
from datetime import datetime

# Importando a lógica da Sprint 6
from rag_milvus_query import embed_text, search_milvus, build_context, ask_ollama

app = FastAPI(
    title="GenAcademy RAG API",
    description="API para consulta inteligente de logs de infraestrutura usando RAG.",
    version="1.0.0"
)

# Modelos de Validação (Input/Output)
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class SourceMetadata(BaseModel):
    id: int
    domain: str
    score: float
    text: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceMetadata]

def log_interaction_to_postgres(question: str, answer: str):
    """Guarda o histórico do RAG na base de dados para auditoria."""
    try:
        conn = psycopg2.connect(
            dbname="mlflow",
            user="mlflow",
            password="mlflow",
            host="postgres", # Nome do serviço no docker-compose
            port="5432"
        )
        cur = conn.cursor()
        
        # Cria a tabela se não existir
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_audit_logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                question TEXT,
                answer TEXT
            )
        """)
        
        # Insere o registo
        cur.execute(
            "INSERT INTO rag_audit_logs (timestamp, question, answer) VALUES (%s, %s, %s)",
            (datetime.now(), question, answer)
        )
        
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Erro ao auditar no Postgres: {e}")

@app.get("/")
async def root():
    return {"message": "GenAcademy API está online. Acesse /docs para a documentação Swagger."}

@app.post("/query", response_model=QueryResponse)
async def run_rag_query(request: QueryRequest):
    """
    Endpoint principal: Recebe uma pergunta, consulta o Milvus e gera resposta via Ollama.
    """
    try:
        # 1. Gerar Embedding e Buscar no Milvus
        query_vector = embed_text(request.question)
        hits = search_milvus(query_vector, top_k=request.top_k)
        
        if not hits:
            raise HTTPException(status_code=404, detail="Nenhum contexto encontrado para esta pergunta.")

        # 2. Construir Contexto e Chamar o LLM
        prompt = build_context(request.question, hits)
        answer = ask_ollama(prompt)

        log_interaction_to_postgres(request.question, answer)

        # 3. Formatar Resposta
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=hits
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/metadata")
async def get_metadata():
    """
    Retorna informações sobre o estado da aplicação e modelos utilizados.
    """
    return {
        "embedding_model": os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        "llm_model": os.getenv("LLM_MODEL", "llama2"),
        "vector_db": "Milvus (standalone)",
        "status": "ready"
    }
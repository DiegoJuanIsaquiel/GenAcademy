from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

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
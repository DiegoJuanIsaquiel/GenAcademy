from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

# Importando a lógica da Sprint 6
from rag_milvus_query import AVAILABLE_LLM_MODELS, LLM_MODEL, answer_known_project_question, ask_ollama, build_context, embed_text, search_milvus, should_expose_sources

app = FastAPI(
    title="GenAcademy RAG API",
    description="API para consulta inteligente de logs de infraestrutura usando RAG.",
    version="1.0.0"
)

# Modelos de Validação (Input/Output)
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    llm_model: Optional[str] = None

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
        model_name = request.llm_model or LLM_MODEL
        if model_name not in AVAILABLE_LLM_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Modelo LLM invalido. Escolha entre: {', '.join(AVAILABLE_LLM_MODELS)}."
            )

        # 1. Gerar Embedding e Buscar no Milvus
        query_vector = embed_text(request.question)
        hits = search_milvus(query_vector, top_k=request.top_k, question=request.question)
        
        if not hits:
            raise HTTPException(status_code=404, detail="Nenhum contexto encontrado para esta pergunta.")

        # 2. Construir Contexto e Chamar o LLM
        prompt = build_context(request.question, hits)
        answer = answer_known_project_question(request.question) or ask_ollama(prompt, model_name=model_name)

        # 3. Formatar Resposta
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=hits if should_expose_sources(request.question) else []
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metadata")
async def get_metadata():
    """
    Retorna informações sobre o estado da aplicação e modelos utilizados.
    """
    return {
        "embedding_model": os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        "llm_model": LLM_MODEL,
        "available_llm_models": AVAILABLE_LLM_MODELS,
        "vector_db": "Milvus (standalone)",
        "status": "ready"
    }

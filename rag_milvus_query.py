import argparse
import os
import boto3
import psycopg2
from typing import Any, Dict, List, Optional

from ollama import Client, ResponseError
from pymilvus import Collection, connections, utility

# --- Configurações de Ambiente ---
# MinIO
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "data-lake")

# Milvus
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-standalone")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "GenAcademy_Gold_Data")

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

# PostgreSQL
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "prompts_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres123")

# Clientes globais
ollama_client = Client(host=OLLAMA_HOST)
s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# --- Gestão de Conexão Milvus ---
_milvus_collection = None

def get_milvus_collection() -> Collection:
    global _milvus_collection
    if _milvus_collection is None:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        if not utility.has_collection(COLLECTION_NAME):
            raise RuntimeError(f"Coleção '{COLLECTION_NAME}' não encontrada.")
        _milvus_collection = Collection(COLLECTION_NAME)
        _milvus_collection.load()
    return _milvus_collection

# --- Integração com PostgreSQL (Prompts) ---
def get_active_prompt(prompt_name: str) -> str:
    """Busca o system prompt mais recente/ativo no banco de dados."""
    fallback_prompt = (
        "Você é um assistente técnico do GenAcademy. Use o contexto fornecido "
        "para responder à pergunta. Seja técnico e preciso."
    )
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cur = conn.cursor()
        
        # Ajuste esta query de acordo com o esquema da sua tabela no banco de dados.
        # Exemplo presumido: tabela 'prompts' com colunas 'name', 'content' e 'is_active'.
        query = "SELECT content FROM prompts WHERE name = %s AND is_active = TRUE LIMIT 1;"
        cur.execute(query, (prompt_name,))
        result = cur.fetchone()
        
        cur.close()
        conn.close()

        if result:
            return result[0]
        else:
            print(f"Aviso: Prompt '{prompt_name}' não encontrado. Usando fallback.")
            return fallback_prompt
            
    except Exception as e:
        print(f"Erro ao conectar ao PostgreSQL ou buscar prompt: {e}")
        print("Usando prompt de fallback.")
        return fallback_prompt

# --- Integração com MinIO (Arquivos MD) ---
def fetch_layer_metadata() -> str:
    """Lê os arquivos .md das camadas bronze, silver e gold no MinIO."""
    layers = ["bronze", "silver", "gold"]
    context_parts = ["--- ESTRUTURA DO DATA LAKE (Documentação) ---"]
    
    for layer in layers:
        file_key = f"metadata/{layer}.md"
        try:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
            content = response['Body'].read().decode('utf-8')
            context_parts.append(f"Camada {layer.upper()}:\n{content}")
        except Exception as e:
            context_parts.append(f"Camada {layer.upper()}: Informação indisponível (Erro: {e})")
    
    return "\n\n".join(context_parts)

# --- Processamento de IA ---
def ensure_model(model_name: str):
    try:
        models = [m['name'] for m in ollama_client.list()['models']]
        if model_name not in models and f"{model_name}:latest" not in models:
            print(f"Baixando modelo {model_name}...")
            ollama_client.pull(model_name)
    except Exception as e:
        print(f"Aviso ao verificar modelo: {e}")

def embed_text(text: str) -> List[float]:
    ensure_model(EMBEDDING_MODEL)
    resp = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return resp["embedding"]

def search_context(query_vector: List[float], top_k: int = 3) -> List[Dict]:
    col = get_milvus_collection()
    results = col.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["domain", "text"]
    )
    
    hits = []
    for hit in results[0]:
        hits.append({
            "domain": hit.entity.get("domain"),
            "text": normalize_text(hit.entity.get("text")),
            "score": hit.distance
        })
    return hits

def normalize_text(text: str) -> str:
    return text.replace("False.", "Não.").replace("True.", "Sim.").strip()

# --- Lógica de Prompt e Execução ---
def run_rag_query(question: str, top_k: int):
    # 1. Recupera o System Prompt do PostgreSQL
    system_prompt = get_active_prompt("helpdesk_rag")
    
    # 2. Recupera Documentação Técnica do MinIO
    layer_metadata = fetch_layer_metadata()
    
    # 3. Recupera Dados Semânticos do Milvus
    query_vec = embed_text(question)
    data_hits = search_context(query_vec, top_k)
    
    # 4. Constrói o Contexto Híbrido
    context_str = f"{layer_metadata}\n\n--- DADOS RECUPERADOS (Registros) ---\n"
    for i, hit in enumerate(data_hits):
        context_str += f"Doc {i+1} [{hit['domain']}]: {hit['text']}\n"

    # 5. Geração com LLM
    print("Gerando resposta...")
    ensure_model(LLM_MODEL)
    response = ollama_client.chat(
        model=LLM_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Contexto:\n{context_str}\n\nPergunta: {question}"}
        ],
        options={"temperature": 0.1}
    )
    
    print("\n=== RESPOSTA DA IA ===")
    print(response['message']['content'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Sua pergunta")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    
    try:
        run_rag_query(args.question, args.top_k)
    except Exception as e:
        print(f"Erro na execução: {e}")

import io
import os
import pandas as pd
import boto3
from ollama import Client
from ollama import ResponseError
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# ── Configurações ────────────────────────────────────────────
S3_ENDPOINT_URL = "http://minio:9000" 
AWS_ACCESS_KEY_ID = "minio"
AWS_SECRET_ACCESS_KEY = "minio123"
BUCKET_NAME = "data-lake"

MILVUS_HOST = "milvus-standalone" 
MILVUS_PORT = "19530"
COLLECTION_NAME = "GenAcademy_Gold_Data"
EMBEDDING_DIM = 768

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

ollama_client = Client(host=OLLAMA_HOST)

# ── Cliente MinIO ────────────────────────────────────────────
s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name="us-east-1",
)

def get_latest_parquet(domain: str) -> pd.DataFrame:
    """Busca o arquivo parquet mais recente de um domínio na camada Gold."""
    prefix = f"gold/{domain}/"
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    
    # Pega o arquivo mais recente
    arquivos = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.parquet')]
    if not arquivos:
        print(f"Nenhum arquivo encontrado para o domínio {domain}")
        return pd.DataFrame()
        
    latest_key = sorted(arquivos)[-1]
    print(f"Lendo: {latest_key}")
    
    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=latest_key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def textify_row(row, domain: str) -> str:
    """Transforma a linha da tabela em texto natural para o RAG."""
    if domain == "cost":
        return f"No dia {row['date']} às {row['hora_completa']}, o serviço {row['service']} recebeu {row['requests']} requisições, com um custo estimado de ${row['cost']}."
    elif domain == "performance":
        return f"Em {row['date']} às {row['hora_completa']}, o sistema processou {row['requests']} eventos de {row['unique_users']} usuários únicos."
    elif domain == "security":
        return f"Alerta de segurança: Em {row['date']} às {row['hora_completa']}, o usuário {row['user_hash']} gerou {row['requests']} requisições. Nível de risco: {row['risk_level']}. Acesso fora do horário normal: {row['is_off_hours']}."
    return ""

def setup_milvus():
    """Conecta e recria a coleção no Milvus."""
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    ]
    schema = CollectionSchema(fields, "Coleção RAG GenAcademy")
    collection = Collection(COLLECTION_NAME, schema)
    
    # Cria o índice HNSW para busca rápida
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

def _model_names(model_list_response):
    """Extrai nomes de modelos lidando com formatos diferentes da API Ollama."""
    models = []
    if isinstance(model_list_response, dict):
        models = model_list_response.get("models", [])
    else:
        models = getattr(model_list_response, "models", [])

    names = set()
    for model in models:
        if isinstance(model, dict):
            name = model.get("name") or model.get("model")
        else:
            name = getattr(model, "name", None) or getattr(model, "model", None)
        if name:
            names.add(name)
            names.add(name.split(":")[0])
    return names

def ensure_embedding_model():
    """Garante que o modelo de embeddings existe no Ollama antes do loop caro."""
    try:
        available_models = _model_names(ollama_client.list())
    except Exception as exc:
        raise RuntimeError(
            f"Nao foi possivel conectar ao Ollama em {OLLAMA_HOST}. "
            "Verifique se o servico/container 'ollama' esta rodando."
        ) from exc

    if EMBEDDING_MODEL in available_models or EMBEDDING_MODEL.split(":")[0] in available_models:
        return

    print(f"Modelo Ollama '{EMBEDDING_MODEL}' nao encontrado. Baixando agora...")
    try:
        ollama_client.pull(EMBEDDING_MODEL)
    except ResponseError as exc:
        raise RuntimeError(
            f"Nao foi possivel baixar o modelo '{EMBEDDING_MODEL}' no Ollama. "
            f"Execute manualmente: docker exec -it ollama ollama pull {EMBEDDING_MODEL}"
        ) from exc

def extract_embedding(response):
    if isinstance(response, dict) and "embedding" in response:
        return response["embedding"]
    embedding = getattr(response, "embedding", None)
    if embedding:
        return embedding
    raise RuntimeError("Nao foi possivel obter embedding do Ollama.")

def main():
    print("Iniciando Pipeline de Embeddings (Sprint 5)...")
    ensure_embedding_model()
    collection = setup_milvus()
    
    domains = ["cost", "performance", "security"]
    
    for domain in domains:
        df = get_latest_parquet(domain)
        if df.empty: continue
            
        # ── OTIMIZAÇÃO AQUI ──────────────────────────────────────────
        # Vamos usar apenas os últimos 1500 registros para o RAG.
        # Assim o processo termina em poucos minutos e é suficiente para testar.
        df = df.tail(1500) 
        # ─────────────────────────────────────────────────────────────
        
        total = len(df)
        print(f"Gerando embeddings para {total} registros de {domain}...")
        
        insert_data = [[], [], []] # domain, text, embedding
        
        for i, (_, row) in enumerate(df.iterrows()):
            texto = textify_row(row, domain)
            
            # Chama o Ollama apontando para o container
            response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=texto)
            vetor = extract_embedding(response)
            
            insert_data[0].append(domain)
            insert_data[1].append(texto)
            insert_data[2].append(vetor)
            
            # Feedback no terminal a cada 100 registros
            if (i + 1) % 100 == 0:
                print(f" -> Processado: {i + 1} / {total}")
                
        # Insere no Milvus
        collection.insert([insert_data[0], insert_data[1], insert_data[2]])
        print(f"✅ Inserção do domínio {domain} concluída!\n")

    collection.flush()
    print("Índices atualizados no Milvus com sucesso!")

if __name__ == "__main__":
    main()

import io
import os
import re
from typing import Iterable, List, Optional, Tuple

import boto3
import pandas as pd
from ollama import Client, ResponseError
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "data-lake")

MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-standalone")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "GenAcademy_Gold_Data")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
MARKDOWN_CHUNK_SIZE = int(os.getenv("MARKDOWN_CHUNK_SIZE", "3000"))
PROJECT_KNOWLEDGE_PATH = os.getenv("PROJECT_KNOWLEDGE_PATH", "project_knowledge.md")

ollama_client = Client(host=OLLAMA_HOST)
s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name="us-east-1",
)


def list_object_keys(prefix: str, suffix: str) -> List[str]:
    """Lista objetos de um prefixo, incluindo resultados paginados."""
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        keys.extend(
            obj["Key"]
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(suffix)
        )
    return keys


def get_latest_parquet(domain: str) -> pd.DataFrame:
    """Busca o arquivo parquet mais recente de um dominio na camada Gold."""
    prefix = f"gold/{domain}/"
    arquivos = list_object_keys(prefix, ".parquet")
    if not arquivos:
        print(f"Nenhum arquivo encontrado para o dominio {domain}")
        return pd.DataFrame()

    latest_key = sorted(arquivos)[-1]
    print(f"Lendo: {latest_key}")
    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=latest_key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def get_latest_markdown(prefix: str) -> Tuple[Optional[str], str]:
    """Busca o markdown mais recente dentro de um prefixo do data lake."""
    arquivos = list_object_keys(prefix, ".md")
    if not arquivos:
        print(f"Nenhum markdown encontrado em {prefix}")
        return None, ""

    latest_key = sorted(arquivos)[-1]
    print(f"Lendo documentacao: {latest_key}")
    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=latest_key)
    return latest_key, obj["Body"].read().decode("utf-8")


def chunk_markdown(markdown_text: str, max_chars: int = MARKDOWN_CHUNK_SIZE) -> List[str]:
    """Divide markdown por secoes, respeitando o limite de texto do Milvus."""
    sections = re.split(r"(?=^#{1,6}\s)", markdown_text.strip(), flags=re.MULTILINE)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        while len(section) > max_chars:
            split_at = section.rfind("\n", 0, max_chars)
            if split_at <= 0:
                split_at = max_chars
            chunks.append(section[:split_at].strip())
            section = section[split_at:].strip()

        if section:
            chunks.append(section)

    return chunks


def textify_row(row, domain: str) -> str:
    """Transforma uma linha Gold em texto natural para o RAG."""
    if domain == "cost":
        return (
            f"No dia {row['date']} as {row['hora_completa']}, o servico {row['service']} "
            f"recebeu {row['requests']} requisicoes, com um custo estimado de ${row['cost']}."
        )
    if domain == "performance":
        return (
            f"Em {row['date']} as {row['hora_completa']}, o sistema processou "
            f"{row['requests']} eventos de {row['unique_users']} usuarios unicos."
        )
    if domain == "security":
        return (
            f"Alerta de seguranca: Em {row['date']} as {row['hora_completa']}, o usuario "
            f"{row['user_hash']} gerou {row['requests']} requisicoes. Nivel de risco: "
            f"{row['risk_level']}. Acesso fora do horario normal: {row['is_off_hours']}."
        )
    return ""


def setup_milvus() -> Collection:
    """Conecta e recria a colecao no Milvus."""
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    collection = Collection(
        COLLECTION_NAME,
        CollectionSchema(fields, "Colecao RAG GenAcademy"),
    )
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        },
    )
    return collection


def _model_names(model_list_response) -> set:
    """Extrai nomes de modelos lidando com formatos diferentes da API Ollama."""
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


def ensure_embedding_model() -> None:
    """Garante que o modelo de embeddings existe antes do processamento."""
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


def extract_embedding(response) -> List[float]:
    if isinstance(response, dict) and "embedding" in response:
        return response["embedding"]
    embedding = getattr(response, "embedding", None)
    if embedding:
        return embedding
    raise RuntimeError("Nao foi possivel obter embedding do Ollama.")


def insert_texts(collection: Collection, domain: str, texts: Iterable[str]) -> None:
    """Gera embeddings e insere uma sequencia de textos no Milvus."""
    texts = [text for text in texts if text]
    if not texts:
        return

    print(f"Gerando embeddings para {len(texts)} registros de {domain}...")
    insert_data = [[], [], []]  # domain, text, embedding

    for index, text in enumerate(texts, start=1):
        response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=text)
        insert_data[0].append(domain)
        insert_data[1].append(text)
        insert_data[2].append(extract_embedding(response))

        if index % 100 == 0:
            print(f" -> Processado: {index} / {len(texts)}")

    collection.insert(insert_data)
    print(f"Insercao do dominio {domain} concluida!\n")


def insert_pipeline_documentation(collection: Collection) -> None:
    """Inclui os markdowns mais recentes de Bronze, Silver e Gold no RAG."""
    documentation_sources = [
        ("documentation_bronze", "bronze/cloudtrail/"),
        ("documentation_silver", "silver/cloudtrail/"),
        ("documentation_gold_cost", "gold/cost/"),
        ("documentation_gold_performance", "gold/performance/"),
        ("documentation_gold_security", "gold/security/"),
    ]

    for domain, prefix in documentation_sources:
        source_key, markdown_text = get_latest_markdown(prefix)
        if not source_key or not markdown_text:
            continue

        insert_texts(
            collection,
            domain,
            (
                f"Fonte: {source_key}\nCamada/documentacao: {domain}\n\n{chunk}"
                for chunk in chunk_markdown(markdown_text)
            ),
        )

    if os.path.exists(PROJECT_KNOWLEDGE_PATH):
        with open(PROJECT_KNOWLEDGE_PATH, "r", encoding="utf-8") as knowledge_file:
            project_knowledge = knowledge_file.read()
        insert_texts(
            collection,
            "documentation_project",
            (
                f"Fonte: {PROJECT_KNOWLEDGE_PATH}\nDocumentacao: GenAcademy\n\n{chunk}"
                for chunk in chunk_markdown(project_knowledge)
            ),
        )
    else:
        print(f"Documento de conhecimento nao encontrado: {PROJECT_KNOWLEDGE_PATH}")


def main() -> None:
    print("Iniciando Pipeline de Embeddings (Sprint 5)...")
    ensure_embedding_model()
    collection = setup_milvus()

    for domain in ("cost", "performance", "security"):
        df = get_latest_parquet(domain)
        if df.empty:
            continue

        df = df.tail(1500)
        insert_texts(
            collection,
            domain,
            (textify_row(row, domain) for _, row in df.iterrows()),
        )

    insert_pipeline_documentation(collection)
    collection.flush()
    print("Indices atualizados no Milvus com sucesso!")


if __name__ == "__main__":
    main()

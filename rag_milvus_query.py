import argparse
import os
from typing import Any, Dict, List

import ollama
from ollama import Client
from pymilvus import Collection, connections, utility

# Configurações de ambiente
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
LLM_MODEL = os.getenv("LLM_MODEL", "llama2")

ollama_client = Client(host=OLLAMA_HOST)


def connect_milvus() -> Collection:
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    if not utility.has_collection(COLLECTION_NAME):
        raise RuntimeError(
            f"Coleção Milvus '{COLLECTION_NAME}' não encontrada. Execute primeiro create_embeddings.py para criar a coleção."
        )
    return Collection(COLLECTION_NAME)


def embed_text(text: str) -> List[float]:
    response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=text)
    if isinstance(response, dict) and "embedding" in response:
        return response["embedding"]
    raise RuntimeError("Não foi possível obter embedding do Ollama. Verifique o serviço Ollama e o modelo configurado.")


def search_milvus(query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    collection = connect_milvus()
    search_params = {"metric_type": "L2", "params": {"ef": 64}}
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["domain", "text"],
    )

    hits = []
    if not results or not results[0]:
        return hits

    for hit in results[0]:
        domain_value = None
        text_value = None
        if hasattr(hit, "entity") and hit.entity is not None:
            try:
                domain_value = hit.entity.get("domain")
                text_value = hit.entity.get("text")
            except Exception:
                pass
        hits.append(
            {
                "id": int(hit.id),
                "domain": domain_value,
                "text": text_value,
                "score": float(hit.distance) if hasattr(hit, "distance") else float(hit.score),
            }
        )
    return hits


def build_context(question: str, hits: List[Dict[str, Any]]) -> str:
    lines = [
        "Use as informações abaixo para responder à pergunta de forma objetiva e precisa.",
        "Não invente respostas além do contexto retornado.",
        "",
    ]
    for idx, hit in enumerate(hits, start=1):
        lines.append(f"Fonte {idx}: domínio={hit.get('domain')} | score={hit.get('score'):.4f}")
        lines.append(hit.get("text", ""))
        lines.append("")
    lines.append("Pergunta:")
    lines.append(question)
    return "\n".join(lines)


def ask_ollama(prompt: str) -> str:
    try:
        response = ollama_client.generate(model=LLM_MODEL, prompt=prompt, temperature=0.0)
    except Exception as exc:
        raise RuntimeError(f"Erro ao chamar Ollama LLM: {exc}")

    if isinstance(response, dict):
        return response.get("response") or response.get("text") or str(response)
    return str(response)


def format_hits(hits: List[Dict[str, Any]]) -> str:
    lines = ["=== Documentos Recuperados ==="]
    for hit in hits:
        lines.append(f"- id={hit['id']} domain={hit['domain']} score={hit['score']:.4f}")
        lines.append(f"  texto={hit['text']}")
    return "\n".join(lines)


def run_query(question: str, top_k: int, llm: bool, use_openai: bool) -> None:
    print("Gerando embedding da consulta...")
    query_vector = embed_text(question)

    print("Consultando Milvus...")
    hits = search_milvus(query_vector, top_k=top_k)
    if not hits:
        print("Nenhum resultado encontrado no Milvus.")
        return

    print(format_hits(hits))

    prompt = build_context(question, hits)
    if llm:
        print("\n=== Enviando contexto para LLM ===")
        answer = ask_ollama(prompt)
        print("\n=== Resposta LLM ===")
        print(answer)
    else:
        print("\n=== Prompt gerado ===")
        print(prompt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consulta RAG usando Milvus e LLM para o GenAcademy."
    )
    parser.add_argument("question", help="Pergunta em linguagem natural.")
    parser.add_argument("--top-k", type=int, default=5, help="Número de itens vetoriais retornados do Milvus.")
    parser.add_argument("--llm", action="store_true", help="Retorna resposta completa usando o LLM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_query(args.question, args.top_k, args.llm)


if __name__ == "__main__":
    main()

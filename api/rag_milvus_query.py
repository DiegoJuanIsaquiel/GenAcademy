import argparse
import os
from typing import Any, Dict, List

from ollama import Client
from ollama import ResponseError
from pymilvus import Collection, connections, utility

# Configurações de ambiente
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://${MINIO_ADDRESS}")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "${MINIO_ROOT_USER}")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "${MINIO_ROOT_PASSWORD}")
BUCKET_NAME = os.getenv("BUCKET_NAME", "${BUCKET_NAME}")

MILVUS_HOST = os.getenv("MILVUS_HOST", "${MILVUS_HOST}")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "GenAcademy_Gold_Data")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "${OLLAMA_HOST}")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama2")

ollama_client = Client(host=OLLAMA_HOST)

LLM_SYSTEM_PROMPT = """
Você é um analista de dados. Responda as questões feitas pelo usuário, não invente informações, se não houver dados suficientes diga claramente, seja direto e específico e monte uma reposta objetiva com dados que comprove a mesma.
""".strip()


def _model_names(model_list_response: Any) -> set:
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


def ensure_ollama_model(model_name: str) -> None:
    try:
        available_models = _model_names(ollama_client.list())
    except Exception as exc:
        raise RuntimeError(
            f"Nao foi possivel conectar ao Ollama em {OLLAMA_HOST}. "
            "Verifique se o servico/container 'ollama' esta rodando."
        ) from exc

    if model_name in available_models or model_name.split(":")[0] in available_models:
        return

    print(f"Modelo Ollama '{model_name}' nao encontrado. Baixando agora...")
    try:
        ollama_client.pull(model_name)
    except ResponseError as exc:
        raise RuntimeError(
            f"Nao foi possivel baixar o modelo '{model_name}' no Ollama. "
            f"Execute manualmente: docker exec -it ollama ollama pull {model_name}"
        ) from exc


def connect_milvus() -> Collection:
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    if not utility.has_collection(COLLECTION_NAME):
        raise RuntimeError(
            f"Coleção Milvus '{COLLECTION_NAME}' não encontrada. Execute primeiro create_embeddings.py para criar a coleção."
        )
    return Collection(COLLECTION_NAME)


def embed_text(text: str) -> List[float]:
    ensure_ollama_model(EMBEDDING_MODEL)
    response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=text)
    if isinstance(response, dict) and "embedding" in response:
        return response["embedding"]
    embedding = getattr(response, "embedding", None)
    if embedding:
        return embedding
    raise RuntimeError("Não foi possível obter embedding do Ollama. Verifique o serviço Ollama e o modelo configurado.")


def search_milvus(query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    collection = connect_milvus()
    collection.load()
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
    "Você é um analista de dados. Responda as questões feitas pelo usuário, não invente informações, se não houver dados suficientes diga claramente, seja direto e específico e monte uma reposta objetiva com dados que comprove a mesma.",
    ]
    
    for idx, hit in enumerate(hits, start=1):
        lines.append(f"Registro {idx}: dominio={hit.get('domain')}")
        lines.append(normalize_context_text(hit.get("text", "")))
        lines.append("")
        
    lines.append("Pergunta:")
    lines.append(question)
    lines.append("\nResposta final:")
    
    return "\n".join(lines)

def normalize_context_text(text: str) -> str:
    replacements = {
        "Acesso fora do horário normal: False.": "Nao houve acesso fora do horario normal.",
        "Acesso fora do horario normal: False.": "Nao houve acesso fora do horario normal.",
        "Acesso fora do horário normal: True.": "Houve acesso fora do horario normal.",
        "Acesso fora do horario normal: True.": "Houve acesso fora do horario normal.",
        "Nível de risco: normal.": "Nivel de risco: normal.",
        "NÃ­vel de risco: normal.": "Nivel de risco: normal.",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return text


def ask_ollama(prompt: str) -> str:
    ensure_ollama_model(LLM_MODEL)
    try:
        response = ollama_client.generate(
            model=LLM_MODEL,
            prompt=prompt,
            system=LLM_SYSTEM_PROMPT,
            options={
                "temperature": 0.7,
                "top_p": 0.8,
                "repeat_penalty": 1.15,
                "num_predict": 200,
                "num_ctx": 4096
            },
        )
    except Exception as exc:
        raise RuntimeError(f"Erro ao chamar Ollama LLM: {exc}")

    if isinstance(response, dict):
        return clean_llm_answer(response.get("response") or response.get("text") or str(response))
    generated_text = getattr(response, "response", None) or getattr(response, "text", None)
    if generated_text:
        return clean_llm_answer(generated_text)
    return clean_llm_answer(str(response))


def clean_llm_answer(answer: str) -> str:
    generic_phrases = [
        "Portanto, não é necessário tomar medidas de emergência ou alarmes.",
        "Portanto, nao e necessario tomar medidas de emergencia ou alarmes.",
        "Não é necessário tomar medidas de emergência ou alarmes.",
        "Nao e necessario tomar medidas de emergencia ou alarmes.",
    ]
    for phrase in generic_phrases:
        answer = answer.replace(phrase, "")
    return "\n".join(line.rstrip() for line in answer.strip().splitlines() if line.strip())


def format_hits(hits: List[Dict[str, Any]]) -> str:
    lines = ["=== Documentos Recuperados ==="]
    for hit in hits:
        lines.append(f"- id={hit['id']} domain={hit['domain']} distancia_vetorial={hit['score']:.4f}")
        lines.append(f"  texto={hit['text']}")
    return "\n".join(lines)


def run_query(question: str, top_k: int, llm: bool) -> None:
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

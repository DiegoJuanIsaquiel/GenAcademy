import argparse
import os
import unicodedata
from typing import Any, Dict, List

from ollama import Client
from ollama import ResponseError
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
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
DEFAULT_LLM_MODELS = "phi4,qwen3.5:4b,gemma3:4b,deepseek-r1:8b"
AVAILABLE_LLM_MODELS = tuple(dict.fromkeys(
    model.strip()
    for model in (LLM_MODEL, *os.getenv("LLM_MODELS", DEFAULT_LLM_MODELS).split(","))
    if model.strip()
))

ollama_client = Client(host=OLLAMA_HOST)

LLM_SYSTEM_PROMPT = """
Voce e o GenAcademy AI, assistente de dados da plataforma GenAcademy. Responda sempre em portugues do Brasil, com linguagem natural, direta e profissional.
Use apenas os dados fornecidos no contexto. Nao invente fatos, nao cite metricas tecnicas de busca e nao explique o procedimento.
Para perguntas sobre criticidade, use somente campos de negocio: nivel de risco, quantidade de requisicoes, usuario, horario e acesso fora do horario.
Se todos os registros tiverem nivel de risco normal e indicarem que nao houve acesso fora do horario normal, conclua que nao ha alerta critico no contexto.
Nao use recomendacoes genericas quando elas nao forem pedidas. Evite ingles, listas longas, texto repetitivo e conclusoes alarmistas.
Nao diga que medidas, alarmes ou acoes de emergencia sao necessarias ou desnecessarias, a menos que a pergunta solicite recomendacoes.
Para perguntas explicativas sobre o projeto, modelos, metricas ou processamento, organize a resposta em paragrafos ou listas curtas.
Para perguntas analiticas sobre alertas e dados, use o formato:
Conclusao: responda diretamente em uma frase curta.
Evidencias: resuma os principais valores observados em uma frase curta.
""".strip()


def normalize_question(question: str) -> str:
    """Normaliza acentos e espacos para reconhecer intencoes em portugues."""
    normalized = unicodedata.normalize("NFKD", question)
    without_accents = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )
    return " ".join(without_accents.lower().split())


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


def _question_intent(question: str) -> str:
    normalized_question = normalize_question(question)
    if any(term in normalized_question for term in ("quem e voce", "quem voce e")):
        return "identity"
    if any(term in normalized_question for term in ("modelo treinado", "modelos treinados", "modelos foram treinados", "metrica")):
        return "models_metrics"
    if any(term in normalized_question for term in ("pre processamento", "pre-processamento")):
        return "preprocessing"
    if "bronze" in normalized_question or "ingest" in normalized_question:
        return "documentation_bronze"
    if "silver" in normalized_question:
        return "documentation_silver"
    if "gold" in normalized_question:
        return "documentation_gold"
    if any(term in normalized_question for term in ("pipeline", "camada", "processamento", "processo")):
        return "documentation_pipeline"
    if any(term in normalized_question for term in ("seguranca", "risco", "alerta", "acesso fora")):
        return "security"
    if any(term in normalized_question for term in ("custo", "cost", "servico mais caro")):
        return "cost"
    if any(term in normalized_question for term in ("performance", "desempenho", "trafego", "usuarios unicos")):
        return "performance"
    return "general"


def _documentation_domains(question: str) -> List[str]:
    intent = _question_intent(question)
    domains_by_intent = {
        "identity": ["documentation_project"],
        "models_metrics": ["documentation_project"],
        "preprocessing": ["documentation_project"],
        "documentation_bronze": ["documentation_bronze"],
        "documentation_silver": ["documentation_silver"],
        "documentation_gold": [
            "documentation_gold_cost",
            "documentation_gold_performance",
            "documentation_gold_security",
        ],
        "documentation_pipeline": [
            "documentation_bronze",
            "documentation_silver",
            "documentation_gold_cost",
            "documentation_gold_performance",
            "documentation_gold_security",
        ],
    }
    return domains_by_intent.get(intent, [])


def _hits_from_results(results) -> List[Dict[str, Any]]:
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


def _search_domain(collection, query_vector, search_params, domain: str, limit: int):
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        expr=f'domain == "{domain}"',
        output_fields=["domain", "text"],
    )
    return _hits_from_results(results)


def _filter_project_hits(hits: List[Dict[str, Any]], intent: str) -> List[Dict[str, Any]]:
    markers_by_intent = {
        "identity": ("## Quem e o assistente",),
        "models_metrics": ("## Modelos treinados", "## Metricas utilizadas"),
        "preprocessing": ("## Pre-processamento dos dados",),
    }
    markers = markers_by_intent.get(intent, ())
    return [
        hit for hit in hits
        if any(marker in (hit.get("text") or "") for marker in markers)
    ]


def search_milvus(
    query_vector: List[float],
    top_k: int = 5,
    question: str = "",
) -> List[Dict[str, Any]]:
    collection = connect_milvus()
    collection.load()
    search_params = {"metric_type": "L2", "params": {"ef": 64}}
    intent = _question_intent(question)

    documentation_domains = _documentation_domains(question)
    if documentation_domains:
        if documentation_domains == ["documentation_project"]:
            project_hits = _search_domain(
                collection, query_vector, search_params, "documentation_project", 20
            )
            return _filter_project_hits(project_hits, intent)[:top_k]

        hits = []
        per_domain_limit = max(1, (top_k + len(documentation_domains) - 1) // len(documentation_domains))
        for domain in documentation_domains:
            hits.extend(
                _search_domain(collection, query_vector, search_params, domain, per_domain_limit)
            )
        return sorted(hits, key=lambda hit: hit["score"])[:top_k]

    if intent in {"security", "cost", "performance"}:
        return _search_domain(collection, query_vector, search_params, intent, top_k)

    general_results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["domain", "text"],
    )
    return _hits_from_results(general_results)

def build_context(question: str, hits: List[Dict[str, Any]]) -> str:
    lines = [
        "Use as informações abaixo para responder à pergunta de forma objetiva e precisa.",
        "Não invente respostas além do contexto retornado.",
        "Quando não souber a resposta, diga que não sabe.",
    ]

    if any(hit.get("domain") == "documentation_project" for hit in hits):
        lines.extend([
            "Esta e uma pergunta explicativa sobre o projeto.",
            "Responda diretamente com paragrafos ou listas curtas.",
            "Nao use os campos Conclusao e Evidencias.",
        ])
        normalized_question = normalize_question(question)
        if any(term in normalized_question for term in (
            "pre processamento",
            "pre-processamento",
            "pré processamento",
            "pré-processamento",
        )):
            lines.append(
                "Sua resposta deve conter quatro topicos obrigatorios: Bronze, Silver, Gold e Treinamento de machine learning. Nao omita nenhum deles."
            )

    lines.append("\nContexto disponivel:")

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


def answer_known_project_question(question: str) -> str:
    """Responde FAQs factuais que nao devem variar entre modelos LLM."""
    normalized_question = normalize_question(question)

    if any(term in normalized_question for term in ("quem e voce", "quem voce e", "quem é você")):
        return (
            "Eu sou o GenAcademy AI, assistente de dados da plataforma GenAcademy. "
            "Explico a arquitetura e a pipeline e respondo perguntas sobre dados, seguranca, "
            "desempenho, custos e modelos de machine learning do projeto."
        )

    if (
        any(term in normalized_question for term in (
            "modelo treinado",
            "modelos treinados",
            "modelo foi treinado",
            "modelos foram treinados",
        ))
        and any(term in normalized_question for term in ("metrica", "métrica"))
    ):
        return (
            "Foram treinados quatro modelos para deteccao de anomalias: Regressao Logistica, "
            "Random Forest, HistGradientBoosting e Isolation Forest. As metricas usadas foram "
            "accuracy, precision, recall, F1 e ROC AUC; tambem foram gerados relatorios de "
            "classificacao e matrizes de confusao. Os resultados devem ser interpretados com "
            "cautela porque o alvo usa rotulos fracos derivados de regras da camada Gold."
        )

    if any(term in normalized_question for term in (
        "pre processamento",
        "pre-processamento",
        "pré processamento",
        "pré-processamento",
    )):
        return (
            "Bronze: preserva e envia o CSV bruto ao MinIO sem transformacoes. "
            "Silver: remove duplicatas e registros criticos invalidos, padroniza colunas, "
            "converte datas, cria campos temporais, anonimiza usuarios, mascara IPs e salva "
            "Parquet. Gold: normaliza aliases, preenche ausencias, agrupa por minuto e cria "
            "os dominios Cost, Performance e Security. Treinamento de machine learning: une "
            "os dominios por timestamp, cria features, remove vazamento do alvo e linhas sem "
            "sinal, preenche numericos com mediana e categoricos com o valor mais frequente, "
            "aplica StandardScaler e OneHotEncoder."
        )

    return ""


def should_expose_sources(question: str) -> bool:
    """Oculta fontes quando a resposta e apenas institucional/conversacional."""
    normalized_question = normalize_question(question)
    return not any(term in normalized_question for term in ("quem e voce", "quem voce e"))


def ask_ollama(prompt: str, model_name: str = LLM_MODEL) -> str:
    ensure_ollama_model(model_name)
    is_explanatory = "Esta e uma pergunta explicativa sobre o projeto." in prompt
    try:
        response = ollama_client.generate(
            model=model_name,
            prompt=prompt,
            system=LLM_SYSTEM_PROMPT,
            options={
                "temperature": 0.2,
                "top_p": 0.8,
                "repeat_penalty": 1.15,
                "num_predict": 500 if is_explanatory else 240,
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


def run_query(question: str, top_k: int, llm: bool, model_name: str = LLM_MODEL) -> None:
    print("Gerando embedding da consulta...")
    query_vector = embed_text(question)

    print("Consultando Milvus...")
    hits = search_milvus(query_vector, top_k=top_k, question=question)
    if not hits:
        print("Nenhum resultado encontrado no Milvus.")
        return

    if should_expose_sources(question):
        print(format_hits(hits))

    prompt = build_context(question, hits)
    if llm:
        print("\n=== Enviando contexto para LLM ===")
        answer = answer_known_project_question(question) or ask_ollama(prompt, model_name=model_name)
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
    parser.add_argument(
        "--model",
        choices=AVAILABLE_LLM_MODELS,
        default=LLM_MODEL,
        help="Modelo Ollama usado para gerar a resposta.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_query(args.question, args.top_k, args.llm, args.model)


if __name__ == "__main__":
    main()

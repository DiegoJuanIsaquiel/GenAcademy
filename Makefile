# Variáveis de Configuração
CONTAINER_MLFLOW=mlflow-server
PYTHON=python

.PHONY: up down build ingest-bronze process-silver process-gold train-ml create-embeddings query-rag pipeline help

# Comando padrão: mostra a ajuda
help:
	@echo "🚀 GenAcademy Automation - Comandos disponíveis:"
	@echo "  infra-up           - Inicia todos os containers (Docker Compose)"
	@echo "  infra-down         - Para e remove os containers e redes"
	@echo "  infra-build        - Reconstrói as imagens e inicia os containers"
	@echo "  data-bronze        - Executa o pipeline de ingestão (CSV -> Bronze)"
	@echo "  data-silver        - Executa o processamento Silver (Limpeza/LGPD)"
	@echo "  data-gold          - Executa o processamento Gold (Domínios de negócio)"
	@echo "  ml-train           - Executa o treinamento de modelos no MLflow"
	@echo "  rag-embeddings     - Gera e insere os embeddings no Milvus"
	@echo "  pipeline-full      - Executa o fluxo completo (Bronze até Embeddings)"
	@echo "  clean              - Remove ficheiros temporários e pastas de cache"
	@echo "  test-api           - Executa testes unitários da API"
	@echo "  test-data          - Executa testes de integração e LGPD na camada Silver
	@echo "  test-ai            - Executa bateria de testes RAG"
	@echo "  audit-logs         - Consulta os logs de auditoria RAG no PostgreSQL"


# --- INFRAESTRUTURA ---
infra-up:
	docker compose up -d

infra-down:
	docker compose down

infra-build:
	docker compose up --build -d

# --- PIPELINE DE DADOS (MEDALLION) ---
data-bronze:
	docker exec -it $(CONTAINER_MLFLOW) $(PYTHON) ingestion_bronze.py

data-silver:
	docker exec -it $(CONTAINER_MLFLOW) $(PYTHON) process_silver.py

data-gold:
	docker exec -it $(CONTAINER_MLFLOW) $(PYTHON) process_gold.py

# --- MACHINE LEARNING & RAG ---
ml-train:
	docker exec -it $(CONTAINER_MLFLOW) $(PYTHON) train_ml_pipeline.py

rag-embeddings:
	docker exec -it $(CONTAINER_MLFLOW) $(PYTHON) create_embeddings.py

# --- AUTOMAÇÃO COMPLETA ---
pipeline-full: data-bronze data-silver data-gold ml-train rag-embeddings
	@echo "✅ Fluxo completo finalizado com sucesso!"

# --- TESTES E AUDITORIA (Sprint 10) ---
test-api:
	docker exec -it genacademy-api pytest test_main.py -v

test-data:
	@echo "Executando testes de integridade e LGPD na camada Silver..."
	docker exec -it mlflow-server pytest test_silver.py -v -W ignore::DeprecationWarning

test-ai:
	@echo "🧠 Executando bateria de testes RAG (Recuperação, Contexto e Alucinação)..."
	docker exec -it genacademy-api pytest test_rag.py -v

audit-logs:
	docker exec -it mlflow-postgres psql -U mlflow -d mlflow -c "SELECT * FROM rag_audit_logs;"

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
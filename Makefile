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
	@echo "  audit-logs         - Mostra os logs de auditoria"
	@echo "  test-api           - Executa os testes unitários da API"
	@echo "  clean              - Remove ficheiros temporários e pastas de cache"

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

# audit logs
audit-logs:
	docker exec -it mlflow-postgres psql -U mlflow -d mlflow -c "SELECT * FROM rag_audit_logs;"

# --- TESTES (Sprint 10) ---
test-api:
	docker exec -it genacademy-api pytest test_main.py -v

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
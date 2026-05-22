# 🚀 Arquitetura da Aplicação - Plataforma de Análise com IA

## 📌 Visão Geral

Esta aplicação implementa uma arquitetura moderna de dados e inteligência artificial, combinando:

* Data Lake (MinIO)
* Processamento de dados (pipelines em Python)
* Machine Learning (detecção de anomalias)
* LLM (Agent com ferramentas)
* Backend (FastAPI)
* Frontend (React)
* Banco relacional (PostgreSQL)
* Vector Database (RAG)

O objetivo é permitir **análise inteligente de dados**, com **detecção de anomalias** e **respostas em linguagem natural** para o usuário.

---

# 🧱 Arquitetura Geral

A arquitetura segue um fluxo dividido em três camadas principais:

```text
Usuário → Frontend → Backend → IA / Dados / ML
```

---

# 👤 Fluxo do Usuário

1. O usuário acessa o sistema via **Frontend (React)**
2. Realiza autenticação
3. Interage com o sistema (dashboard ou perguntas)
4. As requisições são enviadas para o **FastAPI**
5. O backend decide se:

   * consulta dados no PostgreSQL
   * aciona o Agent (LLM)
   * executa análise de anomalias

---

# ⚙️ Backend (FastAPI)

O backend atua como **orquestrador central**, sendo responsável por:

* Autenticação de usuários
* Exposição de APIs
* Integração com:

  * PostgreSQL
  * Data Lake (MinIO)
  * Modelo de ML
  * Agent (LLM)

---


## 🔌 Contrato da API (RAG)

A API do sistema expõe a documentação completa dos contratos de dados e rotas. Com a infraestrutura ativa, os contratos podem ser consultados em:

- **Interactive Swagger UI:** [http://localhost:4200/api/docs](http://localhost:4200/api/docs)
- **Alternative ReDoc:** [http://localhost:4200/api/redoc](http://localhost:4200/api/redoc)

O esquema estático em conformidade com o padrão OpenAPI 3.0.0 também encontra-se arquivado em `Docs/openapi.yaml`.

---

# 🗄️ PostgreSQL (Camada Operacional)

Responsável por armazenar dados de **baixa latência e uso operacional**:

* Usuários
* Configurações
* Métricas agregadas
* Anomalias detectadas
* Insights gerados pela IA

👉 Não armazena dados brutos ou datasets grandes

---

# 🪣 Data Lake (MinIO)

Estruturado em três camadas:

## 🥉 Bronze

* Dados brutos (CSV)
* Origem: dataset do Kaggle
* Sem tratamento

---

## 🥈 Silver

* Dados tratados e limpos
* Conversão para Parquet
* Aplicação de:

  * limpeza
  * filtragem
  * enriquecimento

---

## 🥇 Gold

* Dados organizados por domínio:

  * Custo
  * Performance
  * Segurança
* Prontos para análise e consumo

---

# 🔄 Pipeline de Dados

## Ingestão

```text
Kaggle → Pipeline → MinIO Bronze
```

## Processamento

```text
Bronze → Pipeline → Silver → Pipeline → Gold
```

---

# 🤖 Machine Learning (Detecção de Anomalias)

* Algoritmo utilizado: **Isolation Forest**
* Entrada: dados da camada Gold
* Saída:

  * anomaly_score
  * classificação (normal/anômalo)

## Fluxo:

```text
Gold → Modelo → Resultado → PostgreSQL
```

---

# 🧪 MLflow

Responsável por:

* Versionamento do modelo
* Registro de experimentos
* Gestão do ciclo de vida do modelo

---

# 🧠 IA (Agent com LLM)

O sistema utiliza um **Agent com LLM**, capaz de responder perguntas do usuário.

## Funcionamento:

```text
Usuário → FastAPI → Agent (LLM)
```

O Agent utiliza ferramentas (tools):

* PostgreSQL → dados estruturados
* VectorDB → contexto semântico
* Gold (via pipeline) → dados analíticos

---

# 🔍 Vector Database (RAG)

Responsável por armazenar documentos de conhecimento e permitir:

* busca semântica
* recuperação de contexto
* suporte ao Agent

## Fluxo:

```text
Gold → Pipeline de Embeddings → VectorDB
```

## Implementação disponível

O projeto agora inclui um módulo de RAG:

* `rag_core.py` — implementa o fluxo de recuperação do contexto a partir de documentos Gold existentes
* `list` — lista documentos disponíveis em `gold/` local e/ou em MinIO
* `query` — recupera os trechos mais relevantes e monta o prompt
* `--llm` — opcionalmente envia o contexto para OpenAI e recebe resposta final

### Exemplo de uso

```bash
python rag_core.py list --source local --local-base gold
python rag_core.py query "Quais são os maiores riscos de segurança?" --source local --local-base gold --top-k 5
OPENAI_API_KEY=... python rag_core.py query "O que diz a documentação do gold security?" --source local --local-base gold --llm
```

### Consulta com RAG + Milvus + LLM

O script `rag_milvus_query.py` permite fazer perguntas em linguagem natural que são respondidas com base no contexto recuperado da base vetorial (Milvus) usando o modelo de embedding `nomic-embed-text` e respondidas pelo LLM local `llama2`.

#### Passo a passo para interagir com a LLM

**1. Suba os containers do projeto.**

```bash
docker compose up --build -d
```

**2. Confirme que o Ollama está rodando e que o modelo de embedding existe.**

```bash
docker exec -it ollama ollama list
```

O modelo `nomic-embed-text` deve aparecer na lista. Se não aparecer, baixe manualmente:

```bash
docker exec -it ollama ollama pull nomic-embed-text
```

**3. Gere os dados Gold, caso ainda não tenha feito isso.**

```bash
docker exec -it mlflow-server python ingestion_bronze.py
docker exec -it mlflow-server python process_silver.py
docker exec -it mlflow-server python process_gold.py
```

**4. Crie a base vetorial no Milvus.**

```bash
docker exec -it mlflow-server python create_embeddings.py
```

Esse passo transforma os registros Gold em textos, gera embeddings com Ollama e salva os vetores na coleção `GenAcademy_Gold_Data` do Milvus.

**5. Teste primeiro a recuperação de contexto, sem chamar a LLM.**

```bash
docker exec -it mlflow-server python rag_milvus_query.py "Quais alertas de segurança são mais críticos?" --top-k 5
```

Esse comando mostra os documentos mais parecidos encontrados no Milvus e o prompt que seria enviado para a LLM.

**6. Faça a pergunta usando a LLM.**

```bash
docker exec -it mlflow-server python rag_milvus_query.py "Explique o maior risco de segurança identificado." --top-k 5 --llm
```

Com `--llm`, o script recupera contexto no Milvus, monta um prompt e envia para o modelo configurado em `LLM_MODEL`.

#### Exemplos de Consultas

Aqui estão alguns exemplos de perguntas que você pode fazer:

**Com recuperação de contexto apenas (sem LLM):**
```bash
docker exec -it mlflow-server python rag_milvus_query.py "Quais foram as últimas anomalias captadas e os motivos"
```

**Com resposta da LLM:**
```bash
docker exec -it mlflow-server python rag_milvus_query.py "Quais foram as últimas anomalias captadas e os motivos" --llm
```

**Com mais documentos para contexto:**
```bash
docker exec -it mlflow-server python rag_milvus_query.py "Quais foram as últimas anomalias captadas e os motivos" --top-k 10 --llm
```

**Outras perguntas de exemplo:**
```bash
docker exec -it mlflow-server python rag_milvus_query.py "Quais alertas de segurança são mais críticos?" --llm
docker exec -it mlflow-server python rag_milvus_query.py "Resuma os riscos de segurança encontrados." --llm
docker exec -it mlflow-server python rag_milvus_query.py "Qual foi o padrão de acesso mais incomum?" --llm
docker exec -it mlflow-server python rag_milvus_query.py "Quais usuários têm acessos fora do horário normal?" --llm
```

**7. Opcionalmente, escolha outro modelo de LLM.**

```bash
docker exec -it ollama ollama pull llama2
docker exec -it mlflow-server sh -c "LLM_MODEL=llama2 python rag_milvus_query.py 'Resuma os riscos de segurança encontrados.' --top-k 5 --llm"
```

Por padrão, o projeto usa `nomic-embed-text` para embeddings e `llama2` para respostas em linguagem natural.

#### Argumentos do Script

```bash
docker exec -it mlflow-server python rag_milvus_query.py <pergunta> [opções]
```

- `<pergunta>` (obrigatório): A pergunta em linguagem natural
- `--top-k N` (opcional): Número de documentos a recuperar do Milvus (padrão: 5)
- `--llm` (opcional): Flag para usar o LLM na resposta (sem ela, apenas recupera contexto)

> Se você quiser usar o Agent completo em produção, esse módulo é o núcleo RAG que entrega o contexto semântico ao modelo.

---

# 🔄 Integração com IA

O Agent combina múltiplas fontes:

```text
LLM
 ├── PostgreSQL (métricas e anomalias)
 ├── VectorDB (contexto)
 └── Dados analíticos (Gold)
```

---

# 📊 Domínios de Dados

## 💰 Custo

* timestamp
* requests
* cost

## ⚡ Performance

* latency
* requests
* peak usage
* status

## 🔐 Segurança

* eventos
* acessos suspeitos
* anomaly_score
* risk_level

---

# 🐳 Docker

Toda a aplicação é executada em containers:

* Frontend
* Backend (FastAPI)
* PostgreSQL
* MinIO
* MLflow
* VectorDB

---

# 🔄 Versionamento

O código e pipelines são versionados via GitHub.

---

# 🧠 Decisões de Arquitetura

### Separação de responsabilidades

* PostgreSQL → dados operacionais
* MinIO → dados analíticos
* ML → inteligência
* LLM → explicabilidade

---

### Escalabilidade

* Data Lake permite crescimento de dados
* PostgreSQL garante baixa latência
* MLflow permite evolução do modelo

---

### Flexibilidade

* Agent pode integrar novas tools facilmente
* Novos modelos podem ser adicionados

---

# ▶️ Como Rodar o Projeto
Para executar a pipeline completa de dados e machine learning localmente, certifique-se de ter o Docker e o Docker Compose instalados.

## 1. Preparação Inicial
Baixe o dataset [nineteenFeaturesDf.csv](https://www.kaggle.com/datasets/nobukim/aws-cloudtrails-dataset-from-flaws-cloud?select=nineteenFeaturesDf.csv) e coloque-o na raiz do projeto.

## 2. Subir a Infraestrutura Base
Inicie os serviços do MinIO, PostgreSQL e MLflow:

```bash
docker compose up --build -d
```

_Aguarde alguns segundos para que o container `minio-setup` crie os buckets automaticamente._

## 3. Executar o Pipeline de Dados (Medallion)
Execute sequencialmente os scripts de dados por dentro do container do MLflow:

**Ingestão na Camada Bronze:**

```bash 
docker exec -it mlflow-server python ingestion_bronze.py
```

**Processamento para a Camada Silver**
```bash 
docker exec -it mlflow-server python process_silver.py
```

**Processamento para a Camada Gold**

```bash 
docker exec -it mlflow-server python process_gold.py
```

> Se você alterar dependências Python ou instalar pacotes novos, reconstrua a imagem antes de rodar.
>
> ```bash
docker compose build mlflow
docker compose up -d
```

## 4. Executar o Pipeline de Machine Learning
Após os dados estarem consolidados na camada Gold, você pode treinar os modelos.

**Treinamento dos Modelos:**
```bash 
docker exec -it mlflow-server python train_ml_pipeline.py
```

**Realizar Previsões com o Modelo Treinado:**

```bash
docker exec -it mlflow-server python predict_anomalies.py --model-path artifacts/isolatio
n_forest/isolation_forest.pkl
```
## 5. Acessar os Dashboards Locais
- **MinIO Console:** Acesse http://localhost:9001 (Usuário: minio / Senha: minio123)
- **MLflow UI:** Acesse http://localhost:3000

---

# 🏁 Conclusão

A arquitetura combina:

* Engenharia de dados
* Machine Learning
* Inteligência Artificial
* APIs modernas

Resultando em uma plataforma capaz de:

* Detectar anomalias automaticamente
* Explicar comportamentos com IA
* Escalar para grandes volumes de dados

---

## Integrantes do Projeto

Abaixo, os integrantes listados em **ordem alfabética**:

- **Armando Bertolli** — armando.bertolli@gmail.com — RA **211192** - QA
- **Diego Juan Isaquiel Mizael** — diegoisaquiel1@gmail.com — RA **222545** - Scrum Master
- **Gabriel Henrique Domingues de Oliveira** — gabrieloliveira2758@gmail.com - RA **222398** - QA
- **Giovana Pontes Merguizo** — giovana.merguizo@outlook.com — RA **223397** - Dev
- **Guilherme Bordignon Janczak** — guijanck@gmail.com — RA **222688** - Dev
- **Gustavo Figueiredo Passos** — gustavofp111@gmail.com — RA **222560** - Dev
- **João Luiz Orlandini Alves** — joao.luiz.orlandini@gmail.com — RA **223497** - PO
- **Leonardo Barbosa Gonçalves** — leonardo.goncalves16@outlook.com — RA **211923** - Dev
- **Lucas Laureano Jorge da Silva** — lucaslaureanojorgesilva@hotmail.com —  RA **222679** - Tech Lead


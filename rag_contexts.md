# Contextos do GenAcademy para RAG / IA

Este arquivo reúne contextos úteis que expressam o funcionamento do projeto, as camadas de dados e a forma como a IA deve consumir essas informações.



## 1. Visão geral do projeto

O GenAcademy é uma plataforma de análise de dados que integra:

- um Data Lake construído com MinIO
- pipelines de dados em camadas Bronze, Silver e Gold
- detecção de anomalias com MLflow e modelos de machine learning
- busca semântica e RAG usando embeddings vetoriais
- um agente de IA que responde perguntas com contexto baseado em dados

O fluxo principal é:

Usuário → Frontend → Backend (FastAPI) → Agent / ML / Dados

## 2. Arquitetura de dados

### Bronze

- contém dados brutos originais do dataset AWS CloudTrail
- realiza ingestão sem transformação
- preserva o arquivo original como histórico
- serve como a fonte inicial para Silver e Gold

### Silver

- processa e limpa os dados brutos
- transforma registros em tabelas com campos relevantes
- aplica normalização, padronização e enriquecimento dos dados
- produz datasets que alimentam análises e modelos de ML

### Gold

- agrupa dados em domínios de negócio:
  - `cost` (custo)
  - `performance` (desempenho)
  - `security` (segurança)
- cada domínio gera artefatos como parquet, csv e xlsx
- também produz documentação automática em markdown
- é a base para construção de contextos semânticos e RAG

## 3. Domínios de Gold

### Custo (`cost`)

- registra volume de requisições por serviço e período
- estima custo operacional em valores monetários
- permite responder perguntas como:
  - qual serviço gerou mais requisições?
  - quando o custo foi maior?
  - como o volume impacta o custo?

### Desempenho (`performance`)

- acompanha uso do sistema em intervalos de tempo
- mede número de eventos e usuários distintos
- permite perguntas como:
  - qual período teve maior tráfego?
  - quantos usuários únicos acessaram o sistema?
  - onde houve pico de uso?

### Segurança (`security`)

- identifica padrões suspeitos de acesso
- calcula risco por usuário e horários fora do padrão
- permite perguntas como:
  - quais usuários apresentaram comportamento fora do normal?
  - houve acessos fora de horário?
  - quais períodos tiveram maior risco?

## 4. RAG e contexto semântico

O pipeline de RAG do GenAcademy transforma dados e documentação Gold em vetores que podem ser consultados.

### Fluxo RAG

1. extrai dados Gold ou documentação em texto
2. gera embeddings usando Ollama (`nomic-embed-text`)
3. armazena embeddings no Milvus
4. responde perguntas com busca vetorial e contexto

### Objetivo do contexto

A IA deve usar apenas os trechos retornados pela busca vetorial como suporte para responder.
Cada trecho deve ser tratado como evidência e não deve ser extrapolado além das informações disponíveis.

## 5. Prompt recomendado para o LLM

Use um prompt com:

- instrução clara para responder apenas com base no contexto
- lista de fontes recuperadas
- pergunta do usuário

Exemplo:

```
Use as informações abaixo para responder à pergunta de forma objetiva e precisa.
Não invente respostas além do contexto retornado.

Fonte 1: domínio=security | score=0.9123
Alerta de segurança: Em 2024-01-01 às 09:00, o usuário user_123 gerou 100 requisições. Nível de risco: alto. Acesso fora do horário normal: True.

Fonte 2: domínio=performance | score=0.8731
Em 2024-01-01 às 09:00, o sistema processou 450 eventos de 30 usuários únicos.

Pergunta:
Quais riscos de segurança foram identificados?
```

## 6. Exemplos de uso

### Consulta rápida sem LLM

- `python rag_milvus_query.py "Quais alertas de segurança são mais críticos?" --top-k 5`

### Consulta com resposta gerada pelo LLM

- `python rag_milvus_query.py "Qual é o maior risco de segurança no período recente?" --top-k 5 --llm`

## 7. Valores de aprendizado da IA

A IA deve aprender que:

- o projeto prioriza dados de Gold como fonte de verdade
- o RAG serve para buscar contexto semântico em dados estabelecidos
- o Milvus guarda embeddings e o Ollama gera vetores e respostas
- as respostas devem ficar restritas ao contexto retornado
- as três áreas principais são custo, performance e segurança

## 8. Uso como fonte de contexto para agent

Quando o agente receber uma pergunta, ele deve:

1. converter a pergunta em embedding
2. buscar os trechos mais relevantes em Milvus
3. construir um prompt com esses trechos
4. responder com o LLM usando apenas essa base


## 9. Identidade do Assistente

- Eu sou o Assistente Inteligente RAG do projeto GenAcademy. Fui desenvolvido para atuar como um analista de dados avançado, capaz de analisar logs de infraestrutura, detectar anomalias, classificar chamados de suporte e fornecer recomendações estratégicas com base na base de conhecimento da empresa.



---

Este arquivo pode ser usado como fonte de contexto para treinar o agente e orientar a construção de prompts no GenAcademy.


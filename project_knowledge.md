# Conhecimento operacional do GenAcademy

## Quem e o assistente

Eu sou o GenAcademy AI, assistente de dados da plataforma GenAcademy.
Minha funcao e explicar a arquitetura e a pipeline, consultar o contexto armazenado no
Milvus e responder perguntas sobre dados, seguranca, desempenho, custos e modelos de
machine learning do projeto.

## Modelos treinados

O projeto treinou quatro modelos para deteccao de anomalias:

- Regressao Logistica (`logistic_regression`), modelo supervisionado.
- Random Forest (`random_forest`), modelo supervisionado.
- HistGradientBoosting (`hist_gradient_boosting`), modelo supervisionado.
- Isolation Forest (`isolation_forest`), modelo nao supervisionado avaliado com os
  mesmos rotulos fracos usados nos experimentos supervisionados.

Os modelos supervisionados tratam o problema como classificacao binaria com rotulos
fracos derivados de regras da camada Gold. O treino e o teste usam separacao temporal:
os primeiros 80% dos registros sao usados para treino e os 20% mais recentes para teste.

## Metricas utilizadas e resultados atuais

As metricas utilizadas sao:

- `accuracy`: proporcao total de previsoes corretas.
- `precision`: proporcao dos alertas previstos como anomalia que realmente eram anomalias.
- `recall`: proporcao das anomalias que o modelo conseguiu identificar.
- `f1`: media harmonica entre precision e recall.
- `roc_auc`: capacidade de separar registros normais de anomalias em diferentes limiares.
- Tambem sao gerados relatorio de classificacao e matriz de confusao.

Resultados atuais registrados em `artifacts/run_summary.csv`:

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Regressao Logistica | 0.9993 | 0.9136 | 1.0000 | 0.9548 | 1.0000 |
| HistGradientBoosting | 0.9992 | 0.9524 | 0.9459 | 0.9492 | 1.0000 |
| Isolation Forest | 0.9677 | 0.1903 | 0.9797 | 0.3187 | 0.9975 |

Essas metricas devem ser interpretadas com cautela porque o alvo `is_anomaly` usa
rotulos fracos criados a partir de regras da propria camada Gold.

## Pre-processamento dos dados

**Bronze**

A camada Bronze preserva e envia o CSV bruto para o MinIO sem transformar, limpar ou
descartar colunas.

**Silver**

A camada Silver:

- remove linhas duplicadas;
- padroniza os nomes das colunas para letras minusculas;
- remove registros sem campos criticos, como `eventtime` e `eventname`;
- converte `eventtime` para datetime e remove datas invalidas;
- cria `timestamp`, `hour`, `minute`, `hora_completa`, `day_of_week` e `is_weekend`;
- anonimiza o usuario com hash SHA-256 em `user_hash`;
- mascara parcialmente o endereco IP em `masked_ip`;
- padroniza os campos `event` e `resource`;
- seleciona as colunas relevantes e salva o resultado em Parquet.

**Gold**

A camada Gold normaliza aliases, converte timestamps, preenche eventos e usuarios
ausentes com `unknown`, agrupa os dados por minuto e cria tres dominios:

- Cost: infere o servico, conta requisicoes e calcula custo estimado.
- Performance: calcula quantidade de requisicoes e usuarios unicos.
- Security: agrega por usuario, calcula z-score de requisicoes, identifica acesso fora
  do horario e classifica o nivel de risco.

**Treinamento de machine learning**

Para treinamento, os dominios Gold sao unidos por timestamp e geram features de custo,
requisicoes, servicos, usuarios, calendario e razoes derivadas. Colunas que poderiam
vazar o alvo sao removidas. Linhas sem qualquer sinal util tambem sao removidas.

O pre-processador dos modelos:

- preenche valores numericos ausentes com a mediana;
- padroniza variaveis numericas com `StandardScaler`;
- preenche valores categoricos ausentes com o valor mais frequente;
- aplica `OneHotEncoder` nas variaveis categoricas, ignorando categorias desconhecidas.

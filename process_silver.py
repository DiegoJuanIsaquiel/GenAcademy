import boto3
import pandas as pd
import io
import hashlib
from datetime import datetime


# Configuração de conexão com o MinIO
s3_client = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123",
)


def hash_user_id(valor):
    # Aplica um hash SHA-256 no ID do usuário (LGPD)
    if pd.isna(valor):
        return None
    return hashlib.sha256(str(valor).encode("utf-8")).hexdigest()


def mask_ip(ip):
    # Mascara os últimos octetos do IP (LGPD)
    if pd.isna(ip):
        return None

    partes = str(ip).split(".")
    if len(partes) == 4:
        return f"{partes[0]}.{partes[1]}.***.***"

    return "***"  # Fallback para IPv6 ou formatos inesperados


def upload_buffer_to_minio(buffer: io.BytesIO, bucket_name: str, object_key: str, content_type: str):
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=buffer.getvalue(),
        ContentType=content_type,
    )


def get_silver_column_descriptions():
    return {
        "timestamp": "Data e hora do evento já convertida para tipo datetime. É a principal referência temporal do dataset.",
        "user_hash": "Identificador anonimizado do usuário, gerado com hash SHA-256 para atender LGPD.",
        "event": "Nome padronizado do evento realizado no log.",
        "resource": "Recurso associado ao evento, quando disponível.",
        "hour": "Hora do dia extraída do timestamp.",
        "day_of_week": "Dia da semana numérico extraído do timestamp. Exemplo: 0 = segunda-feira.",
        "is_weekend": "Indicador se o evento ocorreu em final de semana. 1 = sim, 0 = não.",
        "masked_ip": "Endereço IP com mascaramento parcial para atender requisitos de privacidade.",
        "errorcode": "Código de erro retornado pelo evento, quando existir.",
        "errormessage": "Mensagem de erro associada ao evento, quando existir.",
        "useragent": "Identificação do agente ou aplicação cliente que gerou o evento.",
        "requestparameters": "Parâmetros enviados na requisição original.",
        "responseelements": "Elementos retornados na resposta do evento.",
        "eventsource": "Serviço ou origem do evento no CloudTrail.",
    }


def build_silver_markdown(
    linhas_originais: int,
    linhas_finais: int,
    source_key: str,
    destination_key: str,
    colunas_finais: list[str],
) -> str:
    descriptions = get_silver_column_descriptions()

    lines = []
    lines.append("# Documentação do Process Silver")
    lines.append("")
    lines.append("## 1. Para que serve")
    lines.append("")
    lines.append(
        "O processo **Silver** tem como objetivo transformar os dados brutos da camada Bronze "
        "em um dataset mais confiável, limpo, padronizado e pronto para análises posteriores."
    )
    lines.append("")
    lines.append(
        "Ele prepara os logs para consumo pela camada Gold, reduzindo inconsistências, aplicando "
        "regras de privacidade e estruturando as colunas principais."
    )
    lines.append("")
    lines.append("## 2. O que este processo faz")
    lines.append("")
    lines.append("- Lê o arquivo bruto da camada Bronze.")
    lines.append("- Remove linhas duplicadas.")
    lines.append("- Padroniza nomes de colunas para minúsculo.")
    lines.append("- Remove registros com campos críticos ausentes.")
    lines.append("- Converte datas para datetime.")
    lines.append("- Cria colunas temporais derivadas.")
    lines.append("- Anonimiza identificadores de usuário.")
    lines.append("- Mascara IPs para atender LGPD.")
    lines.append("- Seleciona apenas colunas relevantes para a Silver.")
    lines.append("- Salva o resultado em formato Parquet no MinIO.")
    lines.append("")
    lines.append("## 3. Origem e destino")
    lines.append("")
    lines.append(f"- **Origem Bronze:** `{source_key}`")
    lines.append(f"- **Destino Silver:** `{destination_key}`")
    lines.append("")
    lines.append("## 4. Resumo do processamento")
    lines.append("")
    lines.append(f"- **Linhas originais:** {linhas_originais}")
    lines.append(f"- **Linhas finais:** {linhas_finais}")
    lines.append(f"- **Quantidade de colunas finais:** {len(colunas_finais)}")
    lines.append(f"- **Data de geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 5. Regras de tratamento aplicadas")
    lines.append("")
    lines.append("### Limpeza")
    lines.append("- Remove registros duplicados.")
    lines.append("- Remove registros sem `eventtime` e `eventname`, quando essas colunas existirem.")
    lines.append("- Remove falhas de conversão de data.")
    lines.append("")
    lines.append("### Padronização")
    lines.append("- Converte nomes de colunas para lowercase.")
    lines.append("- Cria a coluna `timestamp` a partir de `eventtime`.")
    lines.append("- Cria colunas temporais como `hour`, `day_of_week` e `is_weekend`.")
    lines.append("- Padroniza `event` e `resource` para consumo posterior.")
    lines.append("")
    lines.append("### Privacidade e LGPD")
    lines.append("- Gera `user_hash` com SHA-256 para anonimizar o usuário.")
    lines.append("- Mascara parcialmente `sourceipaddress`, gerando `masked_ip`.")
    lines.append("")
    lines.append("## 6. Colunas finais geradas")
    lines.append("")

    for col in colunas_finais:
        lines.append(f"- **{col}**")

    lines.append("")
    lines.append("## 7. Descrição das colunas")
    lines.append("")

    for col in colunas_finais:
        lines.append(f"### {col}")
        lines.append("")
        lines.append(descriptions.get(col, "Coluna preservada pelo pipeline para uso analítico posterior."))
        lines.append("")

    lines.append("## 8. Observação")
    lines.append("")
    lines.append(
        "Este markdown é gerado automaticamente durante a execução do `process_silver.py`, "
        "junto com o arquivo parquet tratado da camada Silver."
    )
    lines.append("")

    return "\n".join(lines)


def process_bronze_to_silver():
    print("Iniciando pipeline: Bronze -> Silver...")

    data_atual = datetime.now().strftime("%Y-%m-%d")
    bucket_name = "data-lake"

    # Novos caminhos particionados
    source_key = f"bronze/cloudtrail/dt={data_atual}/raw_logs.csv"
    destination_key = f"silver/cloudtrail/dt={data_atual}/cleaned_logs.parquet"
    markdown_key = f"silver/cloudtrail/dt={data_atual}/silver_documentation.md"

    # 1. Lendo o CSV bruto particionado
    print(f"Buscando arquivo em '{bucket_name}/{source_key}'...")
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=source_key)
        df = pd.read_csv(response["Body"])
    except Exception as e:
        print(f"Erro ao buscar o arquivo na Bronze. Verifique se a ingestão rodou hoje. Detalhes: {e}")
        return

    linhas_originais = len(df)

    # 2. Transformação: Limpeza e Estruturação
    print("Limpando e processando os dados...")

    # Remover linhas totalmente duplicadas
    df_limpo = df.drop_duplicates()

    # Padronizar nomes de colunas para lowercase
    df_limpo.columns = [col.strip().lower() for col in df_limpo.columns]

    # Removendo nulos críticos
    colunas_base = ["eventtime", "eventname"]
    df_limpo = df_limpo.dropna(subset=[c for c in colunas_base if c in df_limpo.columns])

    # Tipagem e mapeamento
    if "eventtime" in df_limpo.columns:
        df_limpo["eventtime"] = pd.to_datetime(df_limpo["eventtime"], utc=True, errors="coerce")
        df_limpo = df_limpo.dropna(subset=["eventtime"])

        df_limpo["timestamp"] = df_limpo["eventtime"]
        df_limpo["hour"] = df_limpo["timestamp"].dt.hour
        df_limpo["minute"] = df_limpo["timestamp"].dt.minute
        df_limpo["hora_completa"] = df_limpo["timestamp"].dt.strftime("%H:%M")
        df_limpo["day_of_week"] = df_limpo["timestamp"].dt.dayofweek
        df_limpo["is_weekend"] = df_limpo["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # Aplicar LGPD: Hash no userId
    if "useridentityaccountid" in df_limpo.columns:
        df_limpo["user_hash"] = df_limpo["useridentityaccountid"].apply(hash_user_id)
    elif "useridentityusername" in df_limpo.columns:
        df_limpo["user_hash"] = df_limpo["useridentityusername"].apply(hash_user_id)
    else:
        df_limpo["user_hash"] = "unknown"

    # Mascaramento de IP
    if "sourceipaddress" in df_limpo.columns:
        df_limpo["masked_ip"] = df_limpo["sourceipaddress"].apply(mask_ip)

    # Padronização de nomes exigida pelo requisito
    df_limpo["event"] = df_limpo["eventname"] if "eventname" in df_limpo.columns else "unknown"
    df_limpo["resource"] = df_limpo["resources"] if "resources" in df_limpo.columns else "unknown"

    # Definição de colunas finais
    colunas_obrigatorias = [
        "timestamp",
        "user_hash",
        "event",
        "resource",
        "hour",
        "day_of_week",
        "is_weekend",
    ]

    if "masked_ip" in df_limpo.columns:
        colunas_obrigatorias.append("masked_ip")

    colunas_extras_finops = [
        "errorcode",
        "errormessage",
        "useragent",
        "requestparameters",
        "responseelements",
        "eventsource",
    ]

    colunas_finais = colunas_obrigatorias + [col for col in colunas_extras_finops if col in df_limpo.columns]
    df_limpo = df_limpo[colunas_finais]

    linhas_finais = len(df_limpo)

    # Salva parquet na Silver
    print(f"Convertendo para Parquet e enviando para '{bucket_name}/{destination_key}'...")
    parquet_buffer = io.BytesIO()
    df_limpo.to_parquet(parquet_buffer, index=False, engine="pyarrow")
    parquet_buffer.seek(0)

    s3_client.put_object(
        Bucket=bucket_name,
        Key=destination_key,
        Body=parquet_buffer.getvalue(),
        ContentType="application/octet-stream",
    )

    # Gera markdown automático
    markdown_text = build_silver_markdown(
        linhas_originais=linhas_originais,
        linhas_finais=linhas_finais,
        source_key=source_key,
        destination_key=destination_key,
        colunas_finais=colunas_finais,
    )

    markdown_buffer = io.BytesIO(markdown_text.encode("utf-8"))
    upload_buffer_to_minio(
        markdown_buffer,
        bucket_name,
        markdown_key,
        "text/markdown; charset=utf-8",
    )

    print(f"Markdown gerado e enviado para '{bucket_name}/{markdown_key}'...")
    print("Pipeline concluído com sucesso! Os dados tratados estão na camada Silver.")


if __name__ == "__main__":
    process_bronze_to_silver()
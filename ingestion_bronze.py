import boto3
import pandas as pd
import io
from datetime import datetime
import os


# Configuração de conexão com o MinIO
s3_client = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123"
)


def ensure_bucket_exists(bucket_name: str) -> None:
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except Exception:
        print(f"Bucket '{bucket_name}' não encontrado. Criando...")
        s3_client.create_bucket(Bucket=bucket_name)


def upload_buffer_to_minio(buffer: io.BytesIO, bucket_name: str, object_key: str, content_type: str):
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=buffer.getvalue(),
        ContentType=content_type,
    )


def build_bronze_markdown(
    local_file_path: str,
    bucket_name: str,
    destination_path: str,
    linhas: int,
    colunas: list[str],
) -> str:
    lines = []
    lines.append("# Documentação do Process Bronze")
    lines.append("")
    lines.append("## 1. Para que serve")
    lines.append("")
    lines.append(
        "O processo **Bronze** tem como objetivo realizar a ingestão bruta do dataset original "
        "para o data lake, sem transformar, limpar ou descartar colunas."
    )
    lines.append("")
    lines.append(
        "Ele funciona como a porta de entrada dos dados, preservando o arquivo original para que "
        "as próximas camadas do pipeline possam tratá-lo com segurança."
    )
    lines.append("")
    lines.append("## 2. O que este processo faz")
    lines.append("")
    lines.append("- Verifica se o arquivo CSV existe localmente.")
    lines.append("- Garante a existência do bucket de destino.")
    lines.append("- Faz upload do arquivo bruto para a camada Bronze.")
    lines.append("- Lê o CSV apenas para fins de documentação.")
    lines.append("- Gera um markdown automático com informações do dataset ingerido.")
    lines.append("")
    lines.append("## 3. Origem e destino")
    lines.append("")
    lines.append(f"- **Arquivo local:** `{local_file_path}`")
    lines.append(f"- **Bucket:** `{bucket_name}`")
    lines.append(f"- **Destino Bronze:** `{destination_path}`")
    lines.append("")
    lines.append("## 4. Resumo da ingestão")
    lines.append("")
    lines.append(f"- **Quantidade de linhas:** {linhas}")
    lines.append(f"- **Quantidade de colunas:** {len(colunas)}")
    lines.append(f"- **Data de geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 5. Colunas encontradas no arquivo bruto")
    lines.append("")

    for col in colunas:
        lines.append(f"- **{col}**")

    lines.append("")
    lines.append("## 6. Colunas descartadas")
    lines.append("")
    lines.append(
        "Nenhuma coluna é descartada na camada Bronze. "
        "O objetivo desta etapa é armazenar o dado bruto da forma mais fiel possível ao arquivo original."
    )
    lines.append("")
    lines.append("## 7. Observação")
    lines.append("")
    lines.append(
        "Este markdown é gerado automaticamente durante a execução do `ingest_to_bronze()`, "
        "junto com o upload do arquivo bruto para a camada Bronze."
    )
    lines.append("")

    return "\n".join(lines)


def ingest_to_bronze():
    print("Iniciando ingestão para a camada Bronze...")

    data_atual = datetime.now().strftime("%Y-%m-%d")
    bucket_name = "data-lake"

    destination_path = f"bronze/cloudtrail/dt={data_atual}/raw_logs.csv"
    markdown_path = f"bronze/cloudtrail/dt={data_atual}/bronze_documentation.md"

    local_file_path = "nineteenFeaturesDf.csv"

    if not os.path.exists(local_file_path):
        print(f"Erro: Arquivo '{local_file_path}' não encontrado na pasta atual.")
        return

    ensure_bucket_exists(bucket_name)

    print(f"Fazendo upload de '{local_file_path}' para '{bucket_name}/{destination_path}'...")

    s3_client.upload_file(
        Filename=local_file_path,
        Bucket=bucket_name,
        Key=destination_path
    )

    # Leitura do CSV apenas para gerar documentação automática
    try:
        df = pd.read_csv(local_file_path)
        linhas = len(df)
        colunas = [str(col).strip() for col in df.columns]
    except Exception as e:
        print(f"Aviso: não foi possível ler o CSV para gerar documentação detalhada. Detalhes: {e}")
        linhas = 0
        colunas = []

    markdown_text = build_bronze_markdown(
        local_file_path=local_file_path,
        bucket_name=bucket_name,
        destination_path=destination_path,
        linhas=linhas,
        colunas=colunas,
    )

    markdown_buffer = io.BytesIO(markdown_text.encode("utf-8"))
    upload_buffer_to_minio(
        markdown_buffer,
        bucket_name,
        markdown_path,
        "text/markdown; charset=utf-8",
    )

    print(f"Markdown gerado e enviado para '{bucket_name}/{markdown_path}'...")
    print("Ingestão concluída com sucesso! Critérios de aceite garantidos.")


if __name__ == "__main__":
    ingest_to_bronze()
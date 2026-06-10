import os
from ollama import Client
from pymilvus import connections, Collection

# Configurações iguais às do seu create_embeddings.py
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-standalone")
MILVUS_PORT = "19530"
COLLECTION_NAME = "GenAcademy_Gold_Data"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

ollama_client = Client(host=OLLAMA_HOST)

def main():
    print("Conectando ao Milvus para indexar a documentação do projeto...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    
    # Carrega a coleção existente
    collection = Collection(COLLECTION_NAME)
    
    # Lê o arquivo de contexto
    with open("rag_contexts.md", "r", encoding="utf-8") as f:
        texto_completo = f.read()

    # Divide o texto pelos parágrafos que criamos
    chunks = [c.strip() for c in texto_completo.split("\n\n") if c.strip()]
    
    print(f"Gerando embeddings para {len(chunks)} tópicos da documentação...")
    
    insert_data = [[], [], []] # domain, text, embedding
    
    for chunk in chunks:
        # Chama o Ollama para gerar o vetor do texto
        response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=chunk)
        
        # Extrai o vetor
        embedding = response.get("embedding") or getattr(response, "embedding", None)
        
        insert_data[0].append("documentacao") # Novo domínio
        insert_data[1].append(chunk)
        insert_data[2].append(embedding)

    # Insere no Milvus
    collection.insert(insert_data)
    collection.flush()
    print("✅ Documentação indexada com sucesso! O RAG agora sabe quem é e como foi feito.")

if __name__ == "__main__":
    main()
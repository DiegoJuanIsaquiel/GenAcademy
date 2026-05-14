import psycopg2
import os
from datetime import datetime

# Configurações de ligação ao PostgreSQL (tenta ir buscar ao .env, ou usa os valores por defeito)
DB_HOST = os.getenv("POSTGRES_HOST", "postgres") # Nome do container do postgres no docker-compose
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "mlflowdb")
DB_USER = os.getenv("POSTGRES_USER", "mlflowuser")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "mlflowpass")

def get_db_connection():
    """Cria a ligação à base de dados PostgreSQL."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

def init_prompt_table():
    """Cria a tabela de versionamento de prompts se não existir e insere um prompt inicial."""
    print("A verificar/inicializar a tabela de versionamento de Prompts...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # 1. Criar a tabela
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_prompts (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                prompt_text TEXT NOT NULL,
                version INT NOT NULL,
                is_active BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 2. Verificar se já existe algum prompt para o RAG
        cursor.execute("SELECT COUNT(*) FROM system_prompts WHERE name = 'helpdesk_rag';")
        count = cursor.fetchone()[0]
        
        # 3. Se estiver vazia, inserir a versão 1 (O seu prompt original)
        if count == 0:
            default_prompt = (
                "És um assistente de suporte ao cliente especializado em resolver problemas técnicos. "
                "Utiliza o contexto fornecido para responder à questão de forma clara, educada e direta. "
                "Se não souberes a resposta baseada no contexto, diz que não tens a informação necessária."
            )
            
            cursor.execute("""
                INSERT INTO system_prompts (name, prompt_text, version, is_active)
                VALUES (%s, %s, %s, %s)
            """, ('helpdesk_rag', default_prompt, 1, True))
            
            print("Prompt inicial v1 inserido com sucesso na base de dados.")
            
        conn.commit()
    except Exception as e:
        print(f"Erro ao inicializar base de dados de prompts: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def get_active_prompt(prompt_name="helpdesk_rag"):
    """Vai buscar o prompt ativo mais recente à base de dados."""
    conn = get_db_connection()
    cursor = conn.cursor()
    prompt_text = ""
    
    try:
        # Traz o texto onde o nome coincide e is_active é TRUE (ordena pela versão mais alta por segurança)
        cursor.execute("""
            SELECT prompt_text FROM system_prompts 
            WHERE name = %s AND is_active = TRUE 
            ORDER BY version DESC 
            LIMIT 1;
        """, (prompt_name,))
        
        result = cursor.fetchone()
        
        if result:
            prompt_text = result[0]
        else:
            # Fallback de segurança caso a tabela esteja vazia ou nada esteja ativo
            prompt_text = "És um assistente virtual útil."
            print("Aviso: Nenhum prompt ativo encontrado. A usar fallback.")
            
    except Exception as e:
        print(f"Erro ao obter o prompt: {e}")
        prompt_text = "És um assistente virtual útil." # Fallback
    finally:
        cursor.close()
        conn.close()
        
    return prompt_text
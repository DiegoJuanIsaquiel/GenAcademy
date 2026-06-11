import pytest
import os
from ollama import Client
from rag_milvus_query import embed_text, search_milvus, build_context, LLM_MODEL

# Instancia o cliente do Ollama para o teste de juiz
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
ollama_client = Client(host=OLLAMA_HOST)

class TestRAGPipeline:
    
    def test_1_milvus_retrieval_relevance(self):
        """
        [Teste de Recuperação] 
        Verifica se a busca semântica consegue rotear a intenção corretamente.
        Ao perguntar sobre dinheiro/AWS, deve recuperar o domínio 'cost'.
        """
        query = "Qual foi o pico de custo da nossa infraestrutura?"
        query_vector = embed_text(query)
        
        # Realiza a busca no banco vetorial
        hits = search_milvus(query_vector, top_k=3, question=query)
        
        # Extrai os domínios retornados
        domains_returned = [hit.domain for hit in hits]
        
        # Pelo menos um dos documentos recuperados DEVE ser do domínio 'cost'
        assert "cost" in domains_returned, (
            f"Falha de Busca Semântica: O Milvus não associou a pergunta de custos "
            f"ao domínio correto. Domínios retornados: {domains_returned}"
        )

    def test_2_context_builder_injection(self):
        """
        [Teste de Formatação]
        Garante que o prompt final repassa a informação do banco para o LLM.
        """
        # Simulando um retorno do Milvus
        class FakeHit:
            def __init__(self):
                self.text = "Em 2026-05-14 o servidor AWS-01 teve 500 requisições."
                self.score = 0.95

        fake_hits = [FakeHit()]
        question = "Quantas requisições o servidor teve?"
        
        prompt = build_context(question, fake_hits)
        
        # O texto do banco DEVE estar contido no prompt final
        assert "AWS-01 teve 500 requisições" in prompt, "O contexto vetorial não foi injetado no prompt."
        assert question in prompt, "A pergunta original não foi injetada no prompt."

    def test_3_llm_faithfulness_judge(self):
        """
        [Teste de Alucinação / LLM-as-a-judge]
        Usa o próprio modelo de IA para julgar se uma resposta inventou dados.
        """
        contexto_real = "O erro crítico ocorreu às 14h00 devido a falta de memória RAM no servidor Web."
        
        # Uma resposta fiel e uma resposta que contém alucinação
        resposta_fiel = "O erro aconteceu às 14h00 por falta de memória RAM."
        resposta_alucinada = "O erro aconteceu às 14h00 por um ataque hacker de ransomware."
        
        # Prompt de avaliação estrita para o modelo agir como juiz
        judge_prompt = (
            "Você é um juiz de qualidade de IA. Sua tarefa é verificar se a RESPOSTA "
            "é totalmente baseada no CONTEXTO. Você NÃO PODE dizer mais nada além de 'SIM' ou 'NAO'.\n\n"
            f"CONTEXTO: {contexto_real}\n\n"
            f"RESPOSTA A AVALIAR: {resposta_alucinada}\n\n"
            "A resposta inventa informações que não estão no contexto? (Responda SIM ou NAO)"
        )
        
        response = ollama_client.generate(
            model=LLM_MODEL,
            prompt=judge_prompt,
            options={"temperature": 0.0} # Queremos o modo mais analítico e determinístico possível
        )
        
        avaliacao = response.get("response", "").strip().upper()
        
        # O modelo juiz deve identificar que a resposta_alucinada inventou dados (deve responder SIM para a invenção)
        assert "SIM" in avaliacao, f"O LLM falhou em detectar uma alucinação óbvia. Ele respondeu: {avaliacao}"
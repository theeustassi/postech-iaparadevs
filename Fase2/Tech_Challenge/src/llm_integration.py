"""
Integração com LLMs (Large Language Models)
Geração de relatórios e instruções usando GPT-4

Este módulo implementa:
- Geração de instruções para motoristas
- Criação de relatórios executivos
- Análise de eficiência em linguagem natural
- Sistema de perguntas e respostas
"""

import os
from typing import List, Dict
from datetime import datetime
import json
from pathlib import Path

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Aviso: Biblioteca google-generativeai não instalada. Funcionalidade LLM limitada.")
    print("Instale com: pip install google-generativeai")

from dotenv import load_dotenv

# Carregar variáveis de ambiente
# Buscar .env no diretório raiz do projeto
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)


class LLMReportGenerator:
    """
    Gerador de relatórios e instruções usando LLMs (Google Gemini)
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Args:
            api_key: Chave da API Gemini (opcional, usa variável de ambiente)
            model: Modelo a ser usado (opcional, testa vários automaticamente)
        """
        # Recarregar .env para garantir que temos a chave mais recente
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = None
        self.client = None
        
        if not GEMINI_AVAILABLE:
            print("Biblioteca google-generativeai não está instalada.")
            print("Execute: pip install google-generativeai")
            return
            
        if not self.api_key:
            print("GEMINI_API_KEY não encontrada no arquivo .env")
            print("Obtenha gratuitamente em: https://makersuite.google.com/app/apikey")
            return
        
        # Configurar Gemini
        genai.configure(api_key=self.api_key)
        
        # Lista de modelos para testar (ordem de preferência)
        models_to_try = [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro", 
            "gemini-pro",
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash-latest"
        ] if not model else [model]
        
        # Testar modelos até encontrar um que funciona
        print("Testando modelos Gemini disponíveis...")
        for test_model in models_to_try:
            try:
                print(f"  Tentando {test_model}...", end=" ")
                test_client = genai.GenerativeModel(test_model)
                
                # Fazer um teste simples
                response = test_client.generate_content(
                    "Responda apenas 'OK'",
                    generation_config={"max_output_tokens": 10}
                )
                
                # Se chegou aqui, o modelo funciona
                self.client = test_client
                self.model = test_model
                print(f"✓ Sucesso!")
                print(f"\nGemini inicializado: {test_model}")
                break
                
            except Exception as e:
                print(f"✗ Falhou ({str(e)[:50]}...)")
                continue
        
        if not self.client:
            print("\nNenhum modelo Gemini disponível funcionou.")
            print("Verifique sua API key e conectividade.")
    
    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Chama o LLM com um prompt
        
        Args:
            prompt: Texto do prompt
            temperature: Criatividade da resposta (0-1)
            
        Returns:
            Resposta do LLM
        """
        if not self.client:
            raise RuntimeError("Cliente Gemini não inicializado. Verifique GEMINI_API_KEY e instalação da biblioteca.")
        
        try:
            # Adicionar contexto ao prompt
            full_prompt = f"""Você é um assistente especializado em logística hospitalar e otimização de rotas.

{prompt}"""
            
            # Configurar parâmetros de geração
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": 8192,
            }
            
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            return response.text.strip()
        
        except Exception as e:
            raise RuntimeError(f"Erro ao chamar Gemini: {e}") from e
    
    def generate_driver_instructions(
        self,
        route: List[int],
        route_info: Dict,
        delivery_points: List
    ) -> str:
        """
        Gera instruções detalhadas para o motorista
        
        Args:
            route: Lista de IDs dos pontos na rota
            route_info: Informações da rota
            delivery_points: Lista de DeliveryPoints
            
        Returns:
            Texto com instruções
        """
        # Preparar dados da rota
        route_details = []
        for idx, point_id in enumerate(route[1:-1], start=1):
            point = delivery_points[point_id]
            route_details.append({
                'ordem': idx,
                'nome': point.name,
                'endereco': f"Lat: {point.lat}, Lon: {point.lon}",
                'demanda': point.demand,
                'prioridade': point.priority.name,
                'tempo_servico': point.service_time
            })
        
        # Criar prompt
        prompt = f"""
        Gere instruções claras e profissionais para um motorista de entrega médica.
        
        INFORMAÇÕES DA ROTA:
        - Veículo: {route_info['vehicle']}
        - Distância total: {route_info['distance_km']} km
        - Tempo estimado: {route_info['total_time_hours']:.1f} horas
        - Número de entregas: {route_info['num_deliveries']}
        - Capacidade utilizada: {route_info['capacity_usage_percent']}%
        
        PONTOS DE ENTREGA (em ordem):
        {json.dumps(route_details, indent=2, ensure_ascii=False)}
        
        As instruções devem incluir:
        1. Resumo da missão
        2. Checklist pré-saída
        3. Sequência detalhada de entregas
        4. Orientações de segurança
        5. Contatos de emergência
        
        Use linguagem clara, direta e profissional.
        """
        
        instructions = self._call_llm(prompt, temperature=0.5)
        
        # Adicionar cabeçalho
        header = f"""
        ════════════════════════════════════════════════════════════
        INSTRUÇÕES DE ROTA - SISTEMA HOSPITALAR
        Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        Veículo: {route_info['vehicle']}
        ════════════════════════════════════════════════════════════
        
        """
        
        return header + instructions
    
    def generate_executive_report(
        self,
        route_infos: List[Dict],
        optimization_stats: Dict,
        delivery_points: List
    ) -> str:
        """
        Gera relatório executivo sobre as rotas otimizadas
        
        Args:
            route_infos: Informações de todas as rotas
            optimization_stats: Estatísticas da otimização
            delivery_points: Lista de pontos
            
        Returns:
            Relatório em texto
        """
        # Calcular estatísticas gerais
        total_distance = sum(info['distance_km'] for info in route_infos)
        total_cost = sum(info['cost_reais'] for info in route_infos)
        total_time = sum(info['total_time_hours'] for info in route_infos)
        total_deliveries = sum(info['num_deliveries'] for info in route_infos)
        
        # Análise de prioridades
        priority_counts = {}
        for point in delivery_points[1:]:  # Excluir depósito
            priority_name = point.priority.name
            priority_counts[priority_name] = priority_counts.get(priority_name, 0) + 1
        
        # Criar prompt
        prompt = f"""
        Crie um relatório executivo profissional sobre a otimização de rotas de entregas médicas.
        
        ESTATÍSTICAS GERAIS:
        - Número de rotas: {len(route_infos)}
        - Distância total: {total_distance:.2f} km
        - Custo total: R$ {total_cost:.2f}
        - Tempo total: {total_time:.1f} horas
        - Total de entregas: {total_deliveries}
        
        DISTRIBUIÇÃO DE PRIORIDADES:
        {json.dumps(priority_counts, indent=2, ensure_ascii=False)}
        
        DESEMPENHO DA OTIMIZAÇÃO:
        - Gerações do AG: {optimization_stats.get('generations', 0)}
        - Melhoria obtida: {optimization_stats.get('improvement_percentage', 0):.1f}%
        - Fitness inicial: {optimization_stats.get('best_fitness_initial', 0):.2f}
        - Fitness final: {optimization_stats.get('best_fitness_final', 0):.2f}
        
        DETALHES DAS ROTAS:
        {json.dumps(route_infos, indent=2, ensure_ascii=False)}
        
        O relatório deve incluir:
        1. Sumário Executivo
        2. Análise de Eficiência
        3. Distribuição de Recursos
        4. Indicadores de Desempenho (KPIs)
        5. Recomendações e Oportunidades de Melhoria
        
        Use linguagem executiva, focada em resultados e métricas.
        """
        
        report = self._call_llm(prompt, temperature=0.6)
        
        # Adicionar cabeçalho
        header = f"""
        ════════════════════════════════════════════════════════════
        RELATÓRIO DE OTIMIZAÇÃO DE ROTAS
        Sistema de Distribuição de Medicamentos e Insumos Médicos
        Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        ════════════════════════════════════════════════════════════
        
        """
        
        return header + report
    
    def generate_daily_summary(
        self,
        route_infos: List[Dict],
        issues: List[str] = None
    ) -> str:
        """
        Gera resumo diário das operações
        
        Args:
            route_infos: Informações das rotas executadas
            issues: Lista de problemas/alertas (opcional)
            
        Returns:
            Resumo em texto
        """
        total_distance = sum(info['distance_km'] for info in route_infos)
        total_deliveries = sum(info['num_deliveries'] for info in route_infos)
        avg_capacity = np.mean([info['capacity_usage_percent'] for info in route_infos])
        
        prompt = f"""
        Crie um resumo diário conciso das operações de entrega.
        
        RESUMO DO DIA:
        - Rotas realizadas: {len(route_infos)}
        - Total de entregas: {total_deliveries}
        - Distância percorrida: {total_distance:.1f} km
        - Uso médio de capacidade: {avg_capacity:.1f}%
        
        {"PROBLEMAS IDENTIFICADOS: " + str(issues) if issues else "Nenhum problema relatado"}
        
        Gere um resumo de 2-3 parágrafos destacando:
        - Principais conquistas do dia
        - Eficiência operacional
        - Pontos de atenção (se houver)
        - Perspectiva para o próximo dia
        
        Tom: positivo, mas realista e profissional.
        """
        
        return self._call_llm(prompt, temperature=0.7)
    
    def answer_question(self, question: str, context: Dict) -> str:
        """
        Responde perguntas sobre rotas e entregas em linguagem natural
        
        Args:
            question: Pergunta do usuário
            context: Contexto com informações das rotas
            
        Returns:
            Resposta
        """
        prompt = f"""
        Você é um assistente de logística hospitalar. Responda à seguinte pergunta
        com base nas informações fornecidas.
        
        PERGUNTA: {question}
        
        CONTEXTO:
        {json.dumps(context, indent=2, ensure_ascii=False)}
        
        Forneça uma resposta clara, objetiva e útil.
        """
        
        return self._call_llm(prompt, temperature=0.7)
    
    def suggest_improvements(
        self,
        route_infos: List[Dict],
        historical_data: Dict = None
    ) -> str:
        """
        Sugere melhorias no processo de entregas
        
        Args:
            route_infos: Informações das rotas
            historical_data: Dados históricos (opcional)
            
        Returns:
            Sugestões de melhoria
        """
        # Identificar pontos de atenção
        overloaded_routes = [
            info for info in route_infos 
            if info['capacity_usage_percent'] > 90
        ]
        
        long_routes = [
            info for info in route_infos
            if info['total_time_hours'] > 6
        ]
        
        prompt = f"""
        Analise as rotas de entrega e sugira melhorias operacionais.
        
        DADOS DAS ROTAS:
        {json.dumps(route_infos, indent=2, ensure_ascii=False)}
        
        OBSERVAÇÕES:
        - Rotas com sobrecarga (>90%): {len(overloaded_routes)}
        - Rotas muito longas (>6h): {len(long_routes)}
        
        Forneça sugestões práticas para:
        1. Otimizar uso de recursos
        2. Reduzir tempo de entregas
        3. Melhorar distribuição de carga
        4. Aumentar eficiência geral
        
        Seja específico e prático.
        """
        
        return self._call_llm(prompt, temperature=0.8)
    
    def save_report(self, content: str, filename: str, output_dir: str = "results/relatorios"):
        """
        Salva relatório em arquivo
        
        Args:
            content: Conteúdo do relatório
            filename: Nome do arquivo
            output_dir: Diretório de saída
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Relatório salvo em: {filepath}")


# Importar numpy se necessário
try:
    import numpy as np
except ImportError:
    import statistics
    class np:
        @staticmethod
        def mean(lst):
            return statistics.mean(lst)

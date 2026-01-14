"""
Sistema de Otimizacao de Rotas para Distribuicao de Medicamentos
Script Principal

Este script executa o sistema completo de otimizacao de rotas.
"""

import sys
from pathlib import Path

# Adicionar diretorio src ao path
sys.path.append(str(Path(__file__).parent))

from genetic_algorithm import GeneticAlgorithm
from routing import RouteOptimizer, create_sample_data
from visualization import RouteVisualizer
from llm_integration import LLMReportGenerator


def main():
    """
    Funcao principal do sistema
    """
    print("="*70)
    print("SISTEMA DE OTIMIZACAO DE ROTAS MEDICAS")
    print("Algoritmos Geneticos + LLMs")
    print("="*70)
    print()
    
    # 1. Criar dados de exemplo
    print("Carregando pontos de entrega e veiculos...")
    delivery_points, vehicles = create_sample_data()
    print(f"   - {len(delivery_points)} pontos de entrega")
    print(f"   - {len(vehicles)} veiculos disponiveis")
    print()
    
    # 2. Inicializar otimizador
    print("Inicializando otimizador de rotas...")
    optimizer = RouteOptimizer(delivery_points, vehicles, depot_id=0)
    print("   - Matriz de distancias calculada")
    print()
    
    # 3. Configurar Algoritmo Genetico
    print("Configurando Algoritmo Genetico...")
    ga = GeneticAlgorithm(
        population_size=100,
        generations=300,
        mutation_rate=0.2,
        crossover_rate=0.8,
        elite_size=5,
        tournament_size=5,
        random_seed=42
    )
    print("   - Parametros configurados")
    print()
    
    # 4. Executar otimizacao
    print("Executando otimizacao...")
    print("   Aguarde enquanto o algoritmo genetico evolui...")
    print()
    
    # Criar funcao fitness parcial
    fitness_func = lambda route: optimizer.fitness_function(route, vehicle_id=0)
    
    # Evoluir
    best_solution = ga.evolve(
        num_points=len(delivery_points),
        fitness_function=fitness_func,
        depot=0,
        verbose=True
    )
    
    print()
    print("Otimizacao concluida!")
    print()
    
    # 5. Analisar resultados
    print("Resultados da Otimizacao:")
    print("-" * 70)
    
    stats = ga.get_statistics()
    route_info = optimizer.get_route_info(best_solution.genes, vehicle_id=0)
    
    print(f"  Melhor rota encontrada:")
    print(f"  - Distancia: {route_info['distance_km']} km")
    print(f"  - Tempo total: {route_info['total_time_hours']:.1f} horas")
    print(f"  - Custo: R$ {route_info['cost_reais']:.2f}")
    print(f"  - Entregas: {route_info['num_deliveries']}")
    print(f"  - Uso de capacidade: {route_info['capacity_usage_percent']}%")
    print(f"  - Uso de autonomia: {route_info['autonomy_usage_percent']}%")
    print()
    print(f"  Desempenho do Algoritmo Genetico:")
    print(f"  - Melhoria: {stats['improvement_percentage']:.1f}%")
    print(f"  - Fitness inicial: {stats['best_fitness_initial']:.2f}")
    print(f"  - Fitness final: {stats['best_fitness_final']:.2f}")
    print()
    
    # 6. Dividir em multiplas rotas se necessario
    print("Verificando necessidade de multiplos veiculos...")
    
    if route_info['capacity_usage_percent'] > 100 or route_info['autonomy_usage_percent'] > 100:
        print("   ! Rota unica excede limites. Dividindo em multiplas rotas...")
        sub_routes = optimizer.split_route_for_multiple_vehicles(best_solution.genes)
        print(f"   - Dividido em {len(sub_routes)} rotas")
        
        # Obter informacoes de cada sub-rota
        route_infos = [
            optimizer.get_route_info(route, vehicle_id=i % len(vehicles))
            for i, route in enumerate(sub_routes)
        ]
        routes_to_visualize = sub_routes
    else:
        print("   - Rota unica e viavel")
        route_infos = [route_info]
        routes_to_visualize = [best_solution.genes]
    
    print()
    
    # 7. Visualizar resultados
    print("Gerando visualizacoes...")
    visualizer = RouteVisualizer(delivery_points)
    
    # Mapa de rotas
    visualizer.create_route_map(
        routes_to_visualize,
        route_infos,
        filename="mapa_rotas_otimizadas.html"
    )
    
    # Grafico de evolucao
    visualizer.plot_evolution(
        ga.best_fitness_history,
        ga.avg_fitness_history,
        filename="evolucao_algoritmo_genetico.png"
    )
    
    # Comparacao de rotas (se multiplas)
    if len(route_infos) > 1:
        visualizer.plot_route_comparison(
            route_infos,
            filename="comparacao_rotas.png"
        )
        
        visualizer.plot_capacity_usage(
            route_infos,
            filename="uso_recursos.png"
        )
    
    # Dashboard interativo
    visualizer.create_interactive_dashboard(
        route_infos,
        filename="dashboard_interativo.html"
    )
    
    print()
    
    # 8. Gerar relatorios com LLM
    print("Gerando relatorios com IA...")
    llm_generator = LLMReportGenerator()
    
    # Instrucoes para motorista
    for i, (route, info) in enumerate(zip(routes_to_visualize, route_infos)):
        instructions = llm_generator.generate_driver_instructions(
            route, info, delivery_points
        )
        llm_generator.save_report(
            instructions,
            f"instrucoes_motorista_rota_{i+1}.txt"
        )
    
    # Relatorio executivo
    executive_report = llm_generator.generate_executive_report(
        route_infos,
        stats,
        delivery_points
    )
    llm_generator.save_report(
        executive_report,
        "relatorio_executivo.txt"
    )
    
    # Resumo diario
    daily_summary = llm_generator.generate_daily_summary(route_infos)
    llm_generator.save_report(
        daily_summary,
        "resumo_diario.txt"
    )
    
    # Sugestoes de melhoria
    improvements = llm_generator.suggest_improvements(route_infos)
    llm_generator.save_report(
        improvements,
        "sugestoes_melhoria.txt"
    )
    
    print()
    
    # 9. Finalizacao
    print("="*70)
    print("PROCESSAMENTO CONCLUIDO!")
    print()
    print("Arquivos gerados:")
    print("  - results/graficos/")
    print("    * mapa_rotas_otimizadas.html")
    print("    * evolucao_algoritmo_genetico.png")
    print("    * dashboard_interativo.html")
    if len(route_infos) > 1:
        print("    * comparacao_rotas.png")
        print("    * uso_recursos.png")
    print()
    print("  - results/relatorios/")
    print("    * instrucoes_motorista_rota_*.txt")
    print("    * relatorio_executivo.txt")
    print("    * resumo_diario.txt")
    print("    * sugestoes_melhoria.txt")
    print()
    print("Sistema executado com sucesso!")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecucao interrompida pelo usuario")
    except Exception as e:
        print(f"\n\nErro durante execucao: {e}")
        import traceback
        traceback.print_exc()

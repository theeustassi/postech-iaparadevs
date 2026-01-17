"""
Testes para o módulo de Algoritmos Genéticos
"""

import pytest
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from genetic_algorithm import GeneticAlgorithm, Individual
from routing import RouteOptimizer, create_sample_data, Priority, DeliveryPoint, Vehicle


class TestGeneticAlgorithm:
    """Testes para a classe GeneticAlgorithm"""
    
    def test_create_individual(self):
        """Testa criação de indivíduo"""
        ga = GeneticAlgorithm(random_seed=42)
        individual = ga.create_individual(num_points=11, depot=0)
        
        assert len(individual.genes) == 12  # 11 pontos totais (10 entregas + 1 depósito) -> depósito no início e fim = 12 genes
        assert individual.genes[0] == 0  # Depósito no início
        assert individual.genes[-1] == 0  # Depósito no fim
        assert individual.fitness == float('inf')  # Não avaliado ainda
    
    def test_create_population(self):
        """Testa criação de população"""
        ga = GeneticAlgorithm(population_size=50, random_seed=42)
        population = ga.create_population(num_points=11, depot=0)
        
        assert len(population) == 50
        assert all(isinstance(ind, Individual) for ind in population)
    
    def test_crossover_order(self):
        """Testa operador de crossover"""
        ga = GeneticAlgorithm(random_seed=42)
        
        parent1 = Individual(genes=[0, 1, 2, 3, 4, 5, 0])
        parent2 = Individual(genes=[0, 5, 4, 3, 2, 1, 0])
        
        child1, child2 = ga.crossover_order(parent1, parent2)
        
        # Verificar estrutura
        assert child1.genes[0] == 0
        assert child1.genes[-1] == 0
        assert len(child1.genes) == len(parent1.genes)
        
        # Verificar que todos os genes estão presentes (sem duplicatas)
        assert set(child1.genes[1:-1]) == set(parent1.genes[1:-1])
    
    def test_mutation_swap(self):
        """Testa mutação por troca"""
        ga = GeneticAlgorithm(random_seed=42)
        
        original = Individual(genes=[0, 1, 2, 3, 4, 5, 0])
        mutated = ga.mutation_swap(original)
        
        # Depósitos não devem mudar
        assert mutated.genes[0] == 0
        assert mutated.genes[-1] == 0
        
        # Genes intermediários devem estar presentes
        assert set(mutated.genes) == set(original.genes)
        
        # Fitness deve ser resetado
        assert mutated.fitness == float('inf')


class TestRouteOptimizer:
    """Testes para a classe RouteOptimizer"""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture com dados de exemplo"""
        return create_sample_data()
    
    def test_distance_matrix(self, sample_data):
        """Testa cálculo da matriz de distâncias"""
        delivery_points, vehicles = sample_data
        optimizer = RouteOptimizer(delivery_points, vehicles, depot_id=0)
        
        # Verificar dimensões
        n = len(delivery_points)
        assert optimizer.distance_matrix.shape == (n, n)
        
        # Diagonal deve ser zero
        for i in range(n):
            assert optimizer.distance_matrix[i][i] == 0
        
        # Matriz deve ser simétrica
        assert (optimizer.distance_matrix == optimizer.distance_matrix.T).all()
    
    def test_calculate_route_distance(self, sample_data):
        """Testa cálculo de distância de rota"""
        delivery_points, vehicles = sample_data
        optimizer = RouteOptimizer(delivery_points, vehicles, depot_id=0)
        
        route = [0, 1, 2, 3, 0]
        distance = optimizer.calculate_route_distance(route)
        
        assert distance > 0
        assert isinstance(distance, float)
    
    def test_calculate_route_demand(self, sample_data):
        """Testa cálculo de demanda de rota"""
        delivery_points, vehicles = sample_data
        optimizer = RouteOptimizer(delivery_points, vehicles, depot_id=0)
        
        route = [0, 1, 2, 0]
        demand = optimizer.calculate_route_demand(route)
        
        expected_demand = delivery_points[1].demand + delivery_points[2].demand
        assert demand == expected_demand
    
    def test_fitness_function(self, sample_data):
        """Testa função fitness"""
        delivery_points, vehicles = sample_data
        optimizer = RouteOptimizer(delivery_points, vehicles, depot_id=0)
        
        route = [0, 1, 2, 3, 0]
        fitness, distance, penalty = optimizer.fitness_function(route, vehicle_id=0)
        
        assert fitness > 0
        assert distance > 0
        assert penalty >= 0
        assert fitness >= distance  # Fitness inclui distância + penalidades
    
    def test_check_capacity_constraint(self, sample_data):
        """Testa verificação de restrição de capacidade"""
        delivery_points, vehicles = sample_data
        optimizer = RouteOptimizer(delivery_points, vehicles, depot_id=0)
        
        # Rota pequena que não deve violar capacidade
        route = [0, 1, 2, 0]
        violated, excess = optimizer.check_capacity_constraint(route, vehicles[0])
        
        assert isinstance(violated, bool)
        assert excess >= 0
    
    def test_split_route_multiple_vehicles(self, sample_data):
        """Testa divisão de rota em múltiplas sub-rotas"""
        delivery_points, vehicles = sample_data
        optimizer = RouteOptimizer(delivery_points, vehicles, depot_id=0)
        
        # Rota grande
        route = [0] + list(range(1, len(delivery_points))) + [0]
        sub_routes = optimizer.split_route_for_multiple_vehicles(route)
        
        assert len(sub_routes) > 0
        
        # Todas as sub-rotas devem começar e terminar no depósito
        for sub_route in sub_routes:
            assert sub_route[0] == 0
            assert sub_route[-1] == 0


class TestIntegration:
    """Testes de integração"""
    
    def test_full_optimization(self):
        """Testa otimização completa"""
        # Criar dados
        delivery_points, vehicles = create_sample_data()
        
        # Criar otimizador
        optimizer = RouteOptimizer(delivery_points, vehicles, depot_id=0)
        
        # Criar AG
        ga = GeneticAlgorithm(
            population_size=20,
            generations=10,
            random_seed=42
        )
        
        # Executar otimização
        fitness_func = lambda route: optimizer.fitness_function(route, vehicle_id=0)
        
        best_solution = ga.evolve(
            num_points=len(delivery_points),
            fitness_function=fitness_func,
            depot=0,
            verbose=False
        )
        
        # Verificar resultado
        assert best_solution is not None
        assert len(best_solution.genes) == len(delivery_points) + 1
        assert best_solution.fitness < float('inf')
        
        # Verificar que melhorou ao longo das gerações
        assert ga.best_fitness_history[0] > ga.best_fitness_history[-1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Implementacao do Algoritmo Genetico para Otimizacao de Rotas
Problema do Caixeiro Viajante (TSP) adaptado para o contexto medico

CODIGO BASE: Este modulo extende o codigo base fornecido pela FIAP
Repository: https://github.com/FIAP/genetic_algorithm_tsp
Principais extensoes:
- Estrutura OOP com classes Individual e GeneticAlgorithm
- Suporte a restricoes multiplas (capacidade, autonomia, prioridades)
- Funcao fitness multi-objetivo
- Operadores geneticos avancados (inversao, PMX)
- Visualizacao detalhada da evolucao

Este modulo implementa um Algoritmo Genetico completo com:
- Representacao de individuos (rotas)
- Operadores geneticos (selecao, crossover, mutacao)
- Funcao fitness considerando multiplas restricoes
"""

import numpy as np
import random
from typing import List, Tuple, Callable
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Individual:
    """
    Representa um indivíduo na população (uma possível solução/rota)
    
    Attributes:
        genes: Lista de inteiros representando a ordem de visita dos pontos
        fitness: Valor de aptidão (quanto menor, melhor para TSP)
        distance: Distância total da rota
        penalty: Penalidade por violação de restrições
    """
    genes: List[int]
    fitness: float = float('inf')
    distance: float = 0.0
    penalty: float = 0.0
    
    def __lt__(self, other):
        """Permite ordenar indivíduos por fitness"""
        return self.fitness < other.fitness
    
    def copy(self):
        """Cria uma cópia do indivíduo"""
        return Individual(
            genes=self.genes.copy(),
            fitness=self.fitness,
            distance=self.distance,
            penalty=self.penalty
        )


class GeneticAlgorithm:
    """
    Implementação do Algoritmo Genético para otimização de rotas
    
    Parâmetros principais:
    - population_size: Tamanho da população
    - generations: Número de gerações
    - mutation_rate: Taxa de mutação (0 a 1)
    - crossover_rate: Taxa de cruzamento (0 a 1)
    - elite_size: Número de melhores indivíduos preservados (elitismo)
    - tournament_size: Tamanho do torneio para seleção
    """
    
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 500,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.8,
        elite_size: int = 5,
        tournament_size: int = 5,
        random_seed: int = None
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        # Histórico da evolução
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        
        # Configurar seed para reprodutibilidade
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def create_individual(self, num_points: int, depot: int = 0) -> Individual:
        """
        Cria um indivíduo aleatório (rota aleatória)
        
        Args:
            num_points: Número total de pontos de entrega
            depot: Índice do depósito (ponto de partida e chegada)
            
        Returns:
            Individual com genes aleatórios
        """
        # Criar lista de pontos excluindo o depósito
        points = [i for i in range(num_points) if i != depot]
        # Embaralhar aleatoriamente
        random.shuffle(points)
        # Adicionar depósito no início e fim
        genes = [depot] + points + [depot]
        
        return Individual(genes=genes)
    
    def create_population(self, num_points: int, depot: int = 0) -> List[Individual]:
        """
        Cria a população inicial
        
        Args:
            num_points: Número total de pontos
            depot: Índice do depósito
            
        Returns:
            Lista de indivíduos
        """
        return [
            self.create_individual(num_points, depot)
            for _ in range(self.population_size)
        ]
    
    def evaluate_population(
        self,
        population: List[Individual],
        fitness_function: Callable
    ) -> List[Individual]:
        """
        Avalia todos os indivíduos da população
        
        Args:
            population: Lista de indivíduos
            fitness_function: Função que calcula o fitness
            
        Returns:
            População avaliada
        """
        for individual in population:
            if individual.fitness == float('inf'):
                individual.fitness, individual.distance, individual.penalty = \
                    fitness_function(individual.genes)
        
        return population
    
    def selection_tournament(
        self,
        population: List[Individual],
        k: int = None
    ) -> Individual:
        """
        Seleção por torneio
        Escolhe k indivíduos aleatórios e retorna o melhor
        
        Args:
            population: População atual
            k: Tamanho do torneio (padrão: self.tournament_size)
            
        Returns:
            Melhor indivíduo do torneio
        """
        if k is None:
            k = self.tournament_size
        
        tournament = random.sample(population, k)
        return min(tournament, key=lambda ind: ind.fitness)
    
    def crossover_order(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Crossover de Ordem (OX - Order Crossover)
        Operador especializado para problemas de permutação como o TSP
        
        Mantém a ordem relativa dos genes de um dos pais
        e preenche as lacunas com genes do outro pai
        
        Args:
            parent1: Primeiro pai
            parent2: Segundo pai
            
        Returns:
            Tupla com dois filhos
        """
        size = len(parent1.genes)
        
        # Não fazer crossover nos depósitos (primeiro e último)
        # Trabalhar apenas com a parte intermediária
        genes1 = parent1.genes[1:-1]
        genes2 = parent2.genes[1:-1]
        
        # Selecionar dois pontos de corte aleatórios
        point1, point2 = sorted(random.sample(range(len(genes1)), 2))
        
        # Criar filhos
        child1_genes = [None] * len(genes1)
        child2_genes = [None] * len(genes2)
        
        # Copiar segmento entre os pontos de corte
        child1_genes[point1:point2] = genes1[point1:point2]
        child2_genes[point1:point2] = genes2[point1:point2]
        
        # Preencher as lacunas com genes do outro pai (mantendo a ordem)
        def fill_genes(child, parent):
            pos = point2
            for gene in parent[point2:] + parent[:point2]:
                if gene not in child:
                    while child[pos % len(child)] is not None:
                        pos += 1
                    child[pos % len(child)] = gene
        
        fill_genes(child1_genes, genes2)
        fill_genes(child2_genes, genes1)
        
        # Adicionar depósito no início e fim
        depot = parent1.genes[0]
        child1 = Individual(genes=[depot] + child1_genes + [depot])
        child2 = Individual(genes=[depot] + child2_genes + [depot])
        
        return child1, child2
    
    def mutation_swap(self, individual: Individual) -> Individual:
        """
        Mutação por troca (swap)
        Troca a posição de dois genes aleatórios
        
        Args:
            individual: Indivíduo a ser mutado
            
        Returns:
            Indivíduo mutado
        """
        mutated = individual.copy()
        
        # Não mutar os depósitos (primeiro e último)
        # Mutar apenas a parte intermediária
        if len(mutated.genes) > 3:
            idx1, idx2 = random.sample(range(1, len(mutated.genes) - 1), 2)
            mutated.genes[idx1], mutated.genes[idx2] = \
                mutated.genes[idx2], mutated.genes[idx1]
        
        # Resetar fitness para forçar reavaliação
        mutated.fitness = float('inf')
        
        return mutated
    
    def mutation_inversion(self, individual: Individual) -> Individual:
        """
        Mutação por inversão
        Inverte a ordem de um segmento da rota
        
        Args:
            individual: Indivíduo a ser mutado
            
        Returns:
            Indivíduo mutado
        """
        mutated = individual.copy()
        
        # Não mutar os depósitos
        if len(mutated.genes) > 3:
            idx1, idx2 = sorted(random.sample(range(1, len(mutated.genes) - 1), 2))
            mutated.genes[idx1:idx2] = reversed(mutated.genes[idx1:idx2])
        
        mutated.fitness = float('inf')
        
        return mutated
    
    def evolve(
        self,
        num_points: int,
        fitness_function: Callable,
        depot: int = 0,
        verbose: bool = True
    ) -> Individual:
        """
        Executa o algoritmo genético completo
        
        Args:
            num_points: Número de pontos de entrega
            fitness_function: Função de avaliação
            depot: Índice do depósito
            verbose: Se True, mostra barra de progresso
            
        Returns:
            Melhor indivíduo encontrado
        """
        # Criar população inicial
        population = self.create_population(num_points, depot)
        population = self.evaluate_population(population, fitness_function)
        
        # Configurar barra de progresso
        pbar = tqdm(range(self.generations), disable=not verbose, 
                    desc="Evolução do AG")
        
        for generation in pbar:
            # Ordenar população por fitness
            population.sort()
            
            # Guardar estatísticas
            best_fitness = population[0].fitness
            avg_fitness = np.mean([ind.fitness for ind in population])
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            # Atualizar melhor indivíduo
            self.best_individual = population[0].copy()
            
            # Atualizar barra de progresso
            pbar.set_postfix({
                'Melhor': f'{best_fitness:.2f}',
                'Média': f'{avg_fitness:.2f}'
            })
            
            # Criar nova população
            new_population = []
            
            # Elitismo: preservar os melhores indivíduos
            new_population.extend([ind.copy() for ind in population[:self.elite_size]])
            
            # Gerar o restante da população
            while len(new_population) < self.population_size:
                # Seleção
                parent1 = self.selection_tournament(population)
                parent2 = self.selection_tournament(population)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover_order(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutação
                if random.random() < self.mutation_rate:
                    # Alternar entre swap e inversion
                    if random.random() < 0.5:
                        child1 = self.mutation_swap(child1)
                    else:
                        child1 = self.mutation_inversion(child1)
                
                if random.random() < self.mutation_rate:
                    if random.random() < 0.5:
                        child2 = self.mutation_swap(child2)
                    else:
                        child2 = self.mutation_inversion(child2)
                
                new_population.extend([child1, child2])
            
            # Limitar ao tamanho da população
            population = new_population[:self.population_size]
            
            # Avaliar novos indivíduos
            population = self.evaluate_population(population, fitness_function)
        
        # Retornar o melhor indivíduo final
        population.sort()
        self.best_individual = population[0].copy()
        
        return self.best_individual
    
    def get_statistics(self) -> dict:
        """
        Retorna estatísticas da evolução
        
        Returns:
            Dicionário com estatísticas
        """
        return {
            'best_fitness_final': self.best_fitness_history[-1] if self.best_fitness_history else None,
            'best_fitness_initial': self.best_fitness_history[0] if self.best_fitness_history else None,
            'improvement': self.best_fitness_history[0] - self.best_fitness_history[-1] if self.best_fitness_history else 0,
            'improvement_percentage': ((self.best_fitness_history[0] - self.best_fitness_history[-1]) / 
                                      self.best_fitness_history[0] * 100) if self.best_fitness_history else 0,
            'generations': len(self.best_fitness_history),
            'best_individual': self.best_individual
        }

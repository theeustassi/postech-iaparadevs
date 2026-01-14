"""
Sistema de Otimização de Rotas para Distribuição de Medicamentos
Usando Algoritmos Genéticos

Este módulo contém as classes e funções principais para o sistema.
"""

__version__ = "1.0.0"
__author__ = "FIAP Pós-Tech IA para Devs - Fase 2"

from .genetic_algorithm import GeneticAlgorithm, Individual
from .routing import RouteOptimizer, DeliveryPoint, Vehicle
from .visualization import RouteVisualizer
from .llm_integration import LLMReportGenerator

__all__ = [
    'GeneticAlgorithm',
    'Individual',
    'RouteOptimizer',
    'DeliveryPoint',
    'Vehicle',
    'RouteVisualizer',
    'LLMReportGenerator'
]

"""
Sistema de Roteamento com Restrições Realistas
Problema de Roteamento de Veículos (VRP) para contexto hospitalar

Este módulo implementa:
- Pontos de entrega com prioridades
- Veículos com capacidade e autonomia limitadas
- Cálculo de distâncias
- Função fitness considerando múltiplas restrições
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class Priority(Enum):
    """Níveis de prioridade para entregas"""
    CRITICAL = 1    # Medicamentos críticos/urgentes
    HIGH = 2        # Medicamentos importantes
    MEDIUM = 3      # Insumos regulares
    LOW = 4         # Suprimentos gerais


@dataclass
class Medication:
    """Representa um medicamento ou insumo a ser entregue"""
    id: int
    name: str
    type: str  # Medicamento, Insumo, EPI, Equipamento
    priority: Priority
    quantity: float = 1.0  # Quantidade solicitada
    weight_per_unit: float = 1.0  # Peso unitário (kg)
    
    @property
    def total_weight(self) -> float:
        """Peso total do item"""
        return self.quantity * self.weight_per_unit
    
    def __repr__(self):
        return f"Medication({self.name}: {self.quantity} un)"


@dataclass
class DeliveryPoint:
    """
    Representa um ponto de entrega
    
    Attributes:
        id: Identificador único
        name: Nome do local
        lat: Latitude
        lon: Longitude
        demand: Quantidade/peso a ser entregue (kg)
        priority: Prioridade da entrega
        time_window: Janela de tempo (inicio, fim) em horas
        service_time: Tempo de serviço no local (minutos)
        medications: Lista de medicamentos/insumos solicitados
    """
    id: int
    name: str
    lat: float
    lon: float
    demand: float = 0.0  # em kg
    priority: Priority = Priority.MEDIUM
    time_window: Tuple[float, float] = (0, 24)  # horário de abertura
    service_time: float = 15.0  # minutos
    medications: List[Medication] = field(default_factory=list)
    
    def calculate_demand(self) -> float:
        """Calcula demanda total baseada nos medicamentos"""
        if self.medications:
            return sum(med.total_weight for med in self.medications)
        return self.demand
    
    def get_highest_priority(self) -> Priority:
        """Retorna a prioridade mais alta entre os medicamentos"""
        if self.medications:
            priorities = [med.priority.value for med in self.medications]
            return Priority(min(priorities))  # Menor valor = maior prioridade
        return self.priority
    
    def __repr__(self):
        return f"DeliveryPoint({self.id}: {self.name})"


@dataclass
class Vehicle:
    """
    Representa um veículo de entrega
    
    Attributes:
        id: Identificador único
        name: Nome/placa do veículo
        capacity: Capacidade de carga (kg)
        max_distance: Distância máxima que pode percorrer (km)
        avg_speed: Velocidade média (km/h)
        cost_per_km: Custo por km rodado
    """
    id: int
    name: str
    capacity: float = 100.0  # kg
    max_distance: float = 150.0  # km
    avg_speed: float = 40.0  # km/h
    cost_per_km: float = 2.5  # R$
    
    def __repr__(self):
        return f"Vehicle({self.id}: {self.name})"


class RouteOptimizer:
    """
    Otimizador de rotas com restrições realistas
    """
    
    def __init__(
        self,
        delivery_points: List[DeliveryPoint],
        vehicles: List[Vehicle],
        depot_id: int = 0
    ):
        """
        Args:
            delivery_points: Lista de pontos de entrega (incluindo o depósito)
            vehicles: Lista de veículos disponíveis
            depot_id: ID do depósito (ponto de partida e chegada)
        """
        self.delivery_points = delivery_points
        self.vehicles = vehicles
        self.depot_id = depot_id
        
        # Criar matriz de distâncias
        self.distance_matrix = self._calculate_distance_matrix()
        
        # Pesos para função fitness
        self.weights = {
            'distance': 1.0,
            'priority_penalty': 100.0,
            'capacity_penalty': 500.0,
            'autonomy_penalty': 500.0,
            'time_window_penalty': 200.0
        }
    
    def _calculate_distance_matrix(self) -> np.ndarray:
        """
        Calcula matriz de distâncias entre todos os pontos
        Usa distância euclidiana simplificada
        
        Returns:
            Matriz NxN de distâncias (simétrica)
        """
        n = len(self.delivery_points)
        matrix = np.zeros((n, n))
        
        # Calcular apenas triângulo superior para garantir simetria
        for i in range(n):
            for j in range(i + 1, n):
                p1 = self.delivery_points[i]
                p2 = self.delivery_points[j]
                
                # Distância euclidiana em coordenadas geográficas
                # Aproximação: 1 grau ≈ 111 km
                lat_diff = (p1.lat - p2.lat) * 111
                lon_diff = (p1.lon - p2.lon) * 111 * np.cos(np.radians(p1.lat))
                
                distance = np.sqrt(lat_diff**2 + lon_diff**2)
                matrix[i][j] = distance
                matrix[j][i] = distance  # Garantir simetria
        
        return matrix
    
    def calculate_route_distance(self, route: List[int]) -> float:
        """
        Calcula distância total de uma rota
        
        Args:
            route: Lista de IDs dos pontos na ordem de visita
            
        Returns:
            Distância total em km
        """
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += self.distance_matrix[route[i]][route[i+1]]
        
        return total_distance
    
    def calculate_route_demand(self, route: List[int]) -> float:
        """
        Calcula demanda total de uma rota
        
        Args:
            route: Lista de IDs dos pontos
            
        Returns:
            Demanda total em kg
        """
        total_demand = 0.0
        for point_id in route[1:-1]:  # Excluir depósito
            total_demand += self.delivery_points[point_id].demand
        
        return total_demand
    
    def check_capacity_constraint(
        self,
        route: List[int],
        vehicle: Vehicle
    ) -> Tuple[bool, float]:
        """
        Verifica se a rota respeita a capacidade do veículo
        
        Args:
            route: Lista de IDs dos pontos
            vehicle: Veículo a ser usado
            
        Returns:
            (violou_restricao, excesso)
        """
        demand = self.calculate_route_demand(route)
        excess = max(0, demand - vehicle.capacity)
        
        return excess > 0, excess
    
    def check_autonomy_constraint(
        self,
        route: List[int],
        vehicle: Vehicle
    ) -> Tuple[bool, float]:
        """
        Verifica se a rota respeita a autonomia do veículo
        
        Args:
            route: Lista de IDs dos pontos
            vehicle: Veículo a ser usado
            
        Returns:
            (violou_restricao, excesso_distancia)
        """
        distance = self.calculate_route_distance(route)
        excess = max(0, distance - vehicle.max_distance)
        
        return excess > 0, excess
    
    def calculate_priority_score(self, route: List[int]) -> float:
        """
        Calcula score de prioridade da rota
        Rotas que atendem prioridades críticas primeiro têm score menor (melhor)
        
        Args:
            route: Lista de IDs dos pontos
            
        Returns:
            Score de prioridade (menor é melhor)
        """
        score = 0.0
        
        for position, point_id in enumerate(route[1:-1], start=1):
            point = self.delivery_points[point_id]
            # Quanto mais crítica a prioridade e mais tarde na rota, maior a penalidade
            priority_weight = point.priority.value
            position_weight = position / len(route)
            
            score += priority_weight * position_weight
        
        return score
    
    def fitness_function(
        self,
        route: List[int],
        vehicle_id: int = 0
    ) -> Tuple[float, float, float]:
        """
        Função de fitness para avaliação de rotas
        Considera distância e múltiplas restrições
        
        Args:
            route: Lista de IDs dos pontos na ordem de visita
            vehicle_id: ID do veículo a ser usado
            
        Returns:
            (fitness_total, distancia, penalidade)
        """
        # Selecionar veículo
        vehicle = self.vehicles[vehicle_id] if vehicle_id < len(self.vehicles) else self.vehicles[0]
        
        # Calcular distância base
        distance = self.calculate_route_distance(route)
        
        # Inicializar penalidades
        penalty = 0.0
        
        # Penalidade por violação de capacidade
        violated_capacity, excess_capacity = self.check_capacity_constraint(route, vehicle)
        if violated_capacity:
            penalty += self.weights['capacity_penalty'] * excess_capacity
        
        # Penalidade por violação de autonomia
        violated_autonomy, excess_distance = self.check_autonomy_constraint(route, vehicle)
        if violated_autonomy:
            penalty += self.weights['autonomy_penalty'] * excess_distance
        
        # Penalidade por prioridades
        priority_score = self.calculate_priority_score(route)
        penalty += self.weights['priority_penalty'] * priority_score
        
        # Fitness total = distância + penalidades
        fitness = self.weights['distance'] * distance + penalty
        
        return fitness, distance, penalty
    
    def split_route_for_multiple_vehicles(
        self,
        route: List[int]
    ) -> List[List[int]]:
        """
        Divide uma rota grande em múltiplas rotas menores
        para múltiplos veículos
        
        Args:
            route: Rota completa
            
        Returns:
            Lista de sub-rotas
        """
        # Remover depósitos do início e fim
        points = route[1:-1]
        
        # Ordenar por prioridade
        points_with_priority = [
            (p, self.delivery_points[p].priority.value)
            for p in points
        ]
        points_with_priority.sort(key=lambda x: x[1])
        
        # Dividir em sub-rotas baseado na capacidade
        sub_routes = []
        current_route = [self.depot_id]
        current_capacity = 0
        
        for point_id, _ in points_with_priority:
            point = self.delivery_points[point_id]
            
            # Se adicionar este ponto exceder a capacidade, iniciar nova rota
            if current_capacity + point.demand > self.vehicles[0].capacity and len(current_route) > 1:
                current_route.append(self.depot_id)
                sub_routes.append(current_route)
                current_route = [self.depot_id]
                current_capacity = 0
            
            current_route.append(point_id)
            current_capacity += point.demand
        
        # Adicionar última rota
        if len(current_route) > 1:
            current_route.append(self.depot_id)
            sub_routes.append(current_route)
        
        return sub_routes
    
    def get_route_info(self, route: List[int], vehicle_id: int = 0) -> Dict:
        """
        Obtém informações detalhadas sobre uma rota
        
        Args:
            route: Lista de IDs dos pontos
            vehicle_id: ID do veículo
            
        Returns:
            Dicionário com informações da rota
        """
        vehicle = self.vehicles[vehicle_id] if vehicle_id < len(self.vehicles) else self.vehicles[0]
        
        distance = self.calculate_route_distance(route)
        demand = self.calculate_route_demand(route)
        fitness, _, penalty = self.fitness_function(route, vehicle_id)
        
        # Calcular tempo estimado
        travel_time = distance / vehicle.avg_speed  # horas
        service_time = sum(
            self.delivery_points[p].service_time / 60  # converter para horas
            for p in route[1:-1]
        )
        total_time = travel_time + service_time
        
        # Calcular custo
        cost = distance * vehicle.cost_per_km
        
        return {
            'route': route,
            'vehicle': vehicle.name,
            'distance_km': round(distance, 2),
            'demand_kg': round(demand, 2),
            'fitness': round(fitness, 2),
            'penalty': round(penalty, 2),
            'travel_time_hours': round(travel_time, 2),
            'service_time_hours': round(service_time, 2),
            'total_time_hours': round(total_time, 2),
            'cost_reais': round(cost, 2),
            'num_deliveries': len(route) - 2,
            'capacity_usage_percent': round((demand / vehicle.capacity) * 100, 1),
            'autonomy_usage_percent': round((distance / vehicle.max_distance) * 100, 1)
        }


def load_medications_from_csv(medications_file: str = '../data/medicamentos.csv') -> List[Dict]:
    """
    Carrega catálogo de medicamentos do CSV
    
    Args:
        medications_file: Caminho para o CSV de medicamentos
    
    Returns:
        Lista de dicionários com dados dos medicamentos
    """
    import pandas as pd
    import os
    
    if not os.path.exists(medications_file):
        medications_file = os.path.join(os.path.dirname(__file__), medications_file)
    
    if not os.path.exists(medications_file):
        return []
    
    df = pd.read_csv(medications_file)
    return df.to_dict('records')


def load_data_from_csv(
    locations_file: str = '../data/locais_entrega.csv',
    medications_file: str = '../data/medicamentos.csv',
    vehicles_file: str = None,
    assign_medications: bool = True
) -> Tuple[List[DeliveryPoint], List[Vehicle]]:
    """
    Carrega dados de entrega a partir de arquivos CSV
    
    Args:
        locations_file: Caminho para o CSV de locais de entrega
        medications_file: Caminho para o CSV de medicamentos
        vehicles_file: Caminho para o CSV de veículos (opcional)
        assign_medications: Se True, associa medicamentos aleatórios aos locais
    
    Returns:
        (delivery_points, vehicles)
    """
    import pandas as pd
    import os
    import random
    
    # Carregar locais de entrega
    if not os.path.exists(locations_file):
        locations_file = os.path.join(os.path.dirname(__file__), locations_file)
    
    df_locations = pd.read_csv(locations_file)
    
    # Carregar medicamentos
    medications_catalog = load_medications_from_csv(medications_file)
    
    delivery_points = []
    for _, row in df_locations.iterrows():
        point = DeliveryPoint(
            id=int(row['id']),
            name=row['name'],
            lat=float(row['lat']),
            lon=float(row['lon']),
            demand=float(row['demand']),
            priority=Priority[row['priority']],
            service_time=float(row['service_time'])
        )
        
        # Associar medicamentos aos pontos (exceto depósito)
        if assign_medications and medications_catalog and point.id != 0:
            # Número aleatório de medicamentos (2 a 5 itens por local)
            num_items = random.randint(2, 5)
            selected_meds = random.sample(medications_catalog, min(num_items, len(medications_catalog)))
            
            for med_data in selected_meds:
                # Quantidade aleatória baseada na demanda típica
                typical_demand = float(med_data.get('typical_demand', 1.0))
                quantity = random.uniform(0.5, 2.0) * typical_demand
                
                medication = Medication(
                    id=int(med_data['id']),
                    name=med_data['name'],
                    type=med_data['type'],
                    priority=Priority[med_data['priority']],
                    quantity=quantity,
                    weight_per_unit=1.0  # Assumindo 1kg por unidade
                )
                point.medications.append(medication)
            
            # Recalcular demanda baseada nos medicamentos
            point.demand = point.calculate_demand()
            # Atualizar prioridade para a mais alta dos medicamentos
            point.priority = point.get_highest_priority()
        
        delivery_points.append(point)
    
    # Carregar veículos (se arquivo fornecido) ou usar dados padrão
    if vehicles_file and os.path.exists(vehicles_file):
        df_vehicles = pd.read_csv(vehicles_file)
        vehicles = []
        for _, row in df_vehicles.iterrows():
            vehicle = Vehicle(
                id=int(row['id']),
                name=row['name'],
                capacity=float(row['capacity']),
                max_distance=float(row['max_distance']),
                avg_speed=float(row['avg_speed']),
                cost_per_km=float(row['cost_per_km'])
            )
            vehicles.append(vehicle)
    else:
        # Veículos padrão
        vehicles = [
            Vehicle(0, "Van 001", capacity=50, max_distance=120, avg_speed=35, cost_per_km=2.5),
            Vehicle(1, "Van 002", capacity=50, max_distance=120, avg_speed=35, cost_per_km=2.5),
            Vehicle(2, "Moto 001", capacity=20, max_distance=80, avg_speed=45, cost_per_km=1.5),
        ]
    
    return delivery_points, vehicles


def create_sample_data() -> Tuple[List[DeliveryPoint], List[Vehicle]]:
    """
    Cria dados de exemplo para testes
    Agora carrega dados do CSV se disponível
    
    Returns:
        (delivery_points, vehicles)
    """
    try:
        # Tentar carregar do CSV
        return load_data_from_csv()
    except Exception as e:
        print(f"Aviso: Não foi possível carregar CSV ({e}). Usando dados fixos.")
        
        # Fallback para dados fixos
        delivery_points = [
            DeliveryPoint(0, "Hospital Central (Depósito)", -23.5505, -46.6333, 0, Priority.CRITICAL, service_time=0),
            DeliveryPoint(1, "UBS Vila Mariana", -23.5880, -46.6380, 15, Priority.CRITICAL, service_time=20),
            DeliveryPoint(2, "UBS Mooca", -23.5500, -46.5950, 10, Priority.HIGH, service_time=15),
            DeliveryPoint(3, "UBS Pinheiros", -23.5640, -46.6820, 12, Priority.HIGH, service_time=15),
            DeliveryPoint(4, "UBS Tatuapé", -23.5400, -46.5750, 8, Priority.MEDIUM, service_time=15),
            DeliveryPoint(5, "Atendimento Domiciliar Sr. Silva", -23.5650, -46.6450, 5, Priority.CRITICAL, service_time=30),
            DeliveryPoint(6, "Atendimento Domiciliar Sra. Santos", -23.5750, -46.6150, 3, Priority.HIGH, service_time=25),
            DeliveryPoint(7, "UBS Ipiranga", -23.5920, -46.6100, 10, Priority.MEDIUM, service_time=15),
            DeliveryPoint(8, "UBS Santana", -23.5150, -46.6250, 7, Priority.LOW, service_time=15),
            DeliveryPoint(9, "Clínica Butantã", -23.5690, -46.7230, 12, Priority.MEDIUM, service_time=20),
            DeliveryPoint(10, "Atendimento Domiciliar Sr. Oliveira", -23.5450, -46.6550, 4, Priority.HIGH, service_time=30),
        ]
        
        vehicles = [
            Vehicle(0, "Van 001", capacity=50, max_distance=120, avg_speed=35, cost_per_km=2.5),
            Vehicle(1, "Van 002", capacity=50, max_distance=120, avg_speed=35, cost_per_km=2.5),
            Vehicle(2, "Moto 001", capacity=20, max_distance=80, avg_speed=45, cost_per_km=1.5),
        ]
        
        return delivery_points, vehicles

"""
Visualização de Rotas Otimizadas
Gera mapas interativos e gráficos de análise

Este módulo cria:
- Mapas interativos com folium
- Gráficos de evolução do algoritmo genético
- Visualizações comparativas
"""

import folium
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


class RouteVisualizer:
    """
    Classe para visualização de rotas otimizadas
    """
    
    def __init__(self, delivery_points: List, output_dir: str = "results/graficos"):
        """
        Args:
            delivery_points: Lista de DeliveryPoints
            output_dir: Diretório para salvar visualizações
        """
        self.delivery_points = delivery_points
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar estilo dos gráficos
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def create_route_map(
        self,
        routes: List[List[int]],
        route_infos: List[Dict] = None,
        filename: str = "mapa_rotas.html"
    ) -> folium.Map:
        """
        Cria mapa interativo com as rotas
        
        Args:
            routes: Lista de rotas (cada rota é uma lista de IDs)
            route_infos: Informações detalhadas das rotas (opcional)
            filename: Nome do arquivo HTML a salvar
            
        Returns:
            Objeto folium.Map
        """
        # Calcular centro do mapa
        lats = [p.lat for p in self.delivery_points]
        lons = [p.lon for p in self.delivery_points]
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        
        # Criar mapa
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Cores para diferentes rotas
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                  'lightblue', 'darkgreen', 'cadetblue', 'pink']
        
        # Desenhar cada rota
        for route_idx, route in enumerate(routes):
            color = colors[route_idx % len(colors)]
            
            # Informações da rota
            if route_infos and route_idx < len(route_infos):
                info = route_infos[route_idx]
                route_label = f"Rota {route_idx + 1} - {info.get('vehicle', 'N/A')}"
                route_desc = f"""
                <b>{route_label}</b><br>
                Distância: {info.get('distance_km', 0)} km<br>
                Entregas: {info.get('num_deliveries', 0)}<br>
                Tempo total: {info.get('total_time_hours', 0):.1f}h<br>
                Custo: R$ {info.get('cost_reais', 0):.2f}<br>
                Capacidade usada: {info.get('capacity_usage_percent', 0)}%
                """
            else:
                route_label = f"Rota {route_idx + 1}"
                route_desc = route_label
            
            # Desenhar linha da rota
            route_coords = [
                [self.delivery_points[point_id].lat, 
                 self.delivery_points[point_id].lon]
                for point_id in route
            ]
            
            folium.PolyLine(
                route_coords,
                color=color,
                weight=3,
                opacity=0.7,
                popup=route_desc
            ).add_to(m)
            
            # Adicionar marcadores para cada ponto
            for idx, point_id in enumerate(route):
                point = self.delivery_points[point_id]
                
                # Definir ícone baseado no tipo de ponto
                if point_id == 0:  # Depósito
                    icon = folium.Icon(color='black', icon='home', prefix='fa')
                    popup_text = f"<b>{point.name}</b><br>DEPÓSITO"
                else:
                    # Cor do ícone baseada na prioridade
                    priority_colors = {
                        1: 'red',      # CRITICAL
                        2: 'orange',   # HIGH
                        3: 'blue',     # MEDIUM
                        4: 'green'     # LOW
                    }
                    icon_color = priority_colors.get(point.priority.value, 'gray')
                    icon = folium.Icon(color=icon_color, icon='medkit', prefix='fa')
                    
                    popup_text = f"""
                    <b>{point.name}</b><br>
                    Posição na rota: {idx}<br>
                    Demanda: {point.demand} kg<br>
                    Prioridade: {point.priority.name}<br>
                    Tempo de serviço: {point.service_time} min
                    """
                
                folium.Marker(
                    location=[point.lat, point.lon],
                    popup=folium.Popup(popup_text, max_width=250),
                    tooltip=point.name,
                    icon=icon
                ).add_to(m)
        
        # Adicionar legenda
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 220px; height: auto; 
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
        <p><b>Legenda</b></p>
        <p><i class="fa fa-home" style="color:black"></i> Depósito/Hospital</p>
        <p><i class="fa fa-medkit" style="color:red"></i> Prioridade CRÍTICA</p>
        <p><i class="fa fa-medkit" style="color:orange"></i> Prioridade ALTA</p>
        <p><i class="fa fa-medkit" style="color:blue"></i> Prioridade MÉDIA</p>
        <p><i class="fa fa-medkit" style="color:green"></i> Prioridade BAIXA</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Salvar mapa
        output_path = self.output_dir / filename
        m.save(str(output_path))
        print(f"Mapa salvo em: {output_path}")
        
        return m
    
    def plot_evolution(
        self,
        best_fitness_history: List[float],
        avg_fitness_history: List[float],
        filename: str = "evolucao_ag.png"
    ):
        """
        Plota evolução do algoritmo genético
        
        Args:
            best_fitness_history: Histórico do melhor fitness
            avg_fitness_history: Histórico do fitness médio
            filename: Nome do arquivo a salvar
        """
        plt.figure(figsize=(12, 6))
        
        generations = range(len(best_fitness_history))
        
        plt.plot(generations, best_fitness_history, 'b-', linewidth=2, label='Melhor Fitness')
        plt.plot(generations, avg_fitness_history, 'r--', linewidth=1.5, label='Fitness Médio')
        
        plt.xlabel('Geração', fontsize=12)
        plt.ylabel('Fitness (menor é melhor)', fontsize=12)
        plt.title('Evolução do Algoritmo Genético', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Adicionar anotações
        improvement = best_fitness_history[0] - best_fitness_history[-1]
        improvement_pct = (improvement / best_fitness_history[0]) * 100
        
        plt.text(
            0.02, 0.98,
            f'Melhoria: {improvement:.1f} ({improvement_pct:.1f}%)',
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top',
            fontsize=10
        )
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico de evolução salvo em: {output_path}")
    
    def plot_route_comparison(
        self,
        route_infos: List[Dict],
        filename: str = "comparacao_rotas.png"
    ):
        """
        Plota comparação entre diferentes rotas
        
        Args:
            route_infos: Lista de informações das rotas
            filename: Nome do arquivo a salvar
        """
        if not route_infos:
            print("Nenhuma informação de rota fornecida")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparação de Rotas Otimizadas', fontsize=16, fontweight='bold')
        
        # Preparar dados
        route_names = [f"Rota {i+1}\n{info['vehicle']}" for i, info in enumerate(route_infos)]
        distances = [info['distance_km'] for info in route_infos]
        times = [info['total_time_hours'] for info in route_infos]
        costs = [info['cost_reais'] for info in route_infos]
        deliveries = [info['num_deliveries'] for info in route_infos]
        
        # 1. Distâncias
        axes[0, 0].bar(route_names, distances, color='skyblue', edgecolor='navy')
        axes[0, 0].set_ylabel('Distância (km)', fontsize=11)
        axes[0, 0].set_title('Distância Total por Rota', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Tempos
        axes[0, 1].bar(route_names, times, color='lightcoral', edgecolor='darkred')
        axes[0, 1].set_ylabel('Tempo (horas)', fontsize=11)
        axes[0, 1].set_title('Tempo Total por Rota', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Custos
        axes[1, 0].bar(route_names, costs, color='lightgreen', edgecolor='darkgreen')
        axes[1, 0].set_ylabel('Custo (R$)', fontsize=11)
        axes[1, 0].set_title('Custo Total por Rota', fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Número de entregas
        axes[1, 1].bar(route_names, deliveries, color='plum', edgecolor='purple')
        axes[1, 1].set_ylabel('Número de Entregas', fontsize=11)
        axes[1, 1].set_title('Entregas por Rota', fontsize=12)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico de comparação salvo em: {output_path}")
    
    def plot_capacity_usage(
        self,
        route_infos: List[Dict],
        filename: str = "uso_capacidade.png"
    ):
        """
        Plota uso de capacidade e autonomia dos veículos
        
        Args:
            route_infos: Lista de informações das rotas
            filename: Nome do arquivo a salvar
        """
        if not route_infos:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Utilização de Recursos dos Veículos', fontsize=16, fontweight='bold')
        
        route_names = [f"Rota {i+1}" for i in range(len(route_infos))]
        capacity_usage = [info['capacity_usage_percent'] for info in route_infos]
        autonomy_usage = [info['autonomy_usage_percent'] for info in route_infos]
        
        # Uso de capacidade
        bars1 = ax1.barh(route_names, capacity_usage, color='steelblue', edgecolor='navy')
        ax1.axvline(x=100, color='red', linestyle='--', linewidth=2, label='Limite (100%)')
        ax1.set_xlabel('Uso de Capacidade (%)', fontsize=11)
        ax1.set_title('Uso de Capacidade de Carga', fontsize=12)
        ax1.legend()
        
        # Colorir barras que excedem 100%
        for i, (bar, value) in enumerate(zip(bars1, capacity_usage)):
            if value > 100:
                bar.set_color('red')
        
        # Uso de autonomia
        bars2 = ax2.barh(route_names, autonomy_usage, color='darkorange', edgecolor='darkred')
        ax2.axvline(x=100, color='red', linestyle='--', linewidth=2, label='Limite (100%)')
        ax2.set_xlabel('Uso de Autonomia (%)', fontsize=11)
        ax2.set_title('Uso de Autonomia/Combustível', fontsize=12)
        ax2.legend()
        
        # Colorir barras que excedem 100%
        for i, (bar, value) in enumerate(zip(bars2, autonomy_usage)):
            if value > 100:
                bar.set_color('red')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico de uso de recursos salvo em: {output_path}")
    
    def create_interactive_dashboard(
        self,
        route_infos: List[Dict],
        filename: str = "dashboard_rotas.html"
    ):
        """
        Cria dashboard interativo com Plotly
        
        Args:
            route_infos: Lista de informações das rotas
            filename: Nome do arquivo HTML a salvar
        """
        from plotly.subplots import make_subplots
        
        # Criar subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distância por Rota', 'Tempo Total', 
                           'Custo Operacional', 'Eficiência de Recursos'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        route_names = [f"Rota {i+1}" for i in range(len(route_infos))]
        
        # 1. Distância
        fig.add_trace(
            go.Bar(x=route_names, y=[info['distance_km'] for info in route_infos],
                   name='Distância (km)', marker_color='skyblue'),
            row=1, col=1
        )
        
        # 2. Tempo
        fig.add_trace(
            go.Bar(x=route_names, y=[info['total_time_hours'] for info in route_infos],
                   name='Tempo (h)', marker_color='lightcoral'),
            row=1, col=2
        )
        
        # 3. Custo
        fig.add_trace(
            go.Bar(x=route_names, y=[info['cost_reais'] for info in route_infos],
                   name='Custo (R$)', marker_color='lightgreen'),
            row=2, col=1
        )
        
        # 4. Eficiência (scatter plot)
        fig.add_trace(
            go.Scatter(
                x=[info['capacity_usage_percent'] for info in route_infos],
                y=[info['autonomy_usage_percent'] for info in route_infos],
                mode='markers+text',
                text=route_names,
                textposition='top center',
                marker=dict(size=12, color='purple'),
                name='Uso de Recursos'
            ),
            row=2, col=2
        )
        
        # Atualizar layout
        fig.update_xaxes(title_text="Rota", row=1, col=1)
        fig.update_xaxes(title_text="Rota", row=1, col=2)
        fig.update_xaxes(title_text="Rota", row=2, col=1)
        fig.update_xaxes(title_text="Uso Capacidade (%)", row=2, col=2)
        
        fig.update_yaxes(title_text="km", row=1, col=1)
        fig.update_yaxes(title_text="horas", row=1, col=2)
        fig.update_yaxes(title_text="R$", row=2, col=1)
        fig.update_yaxes(title_text="Uso Autonomia (%)", row=2, col=2)
        
        fig.update_layout(
            title_text="Dashboard de Análise de Rotas",
            showlegend=False,
            height=800
        )
        
        # Salvar
        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        
        print(f"Dashboard interativo salvo em: {output_path}")

"""
Módulo de Avaliação de Modelos
Tech Challenge - Fase 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)


class ModelEvaluator:
    """Classe para avaliação de modelos de classificação"""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model_name, y_true, y_pred, y_proba=None):
        """
        Avalia um modelo com diversas métricas
        
        Args:
            model_name (str): Nome do modelo
            y_true: Valores verdadeiros
            y_pred: Predições
            y_proba: Probabilidades (opcional)
            
        Returns:
            dict: Dicionário com as métricas
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Calcula ROC-AUC se probabilidades estiverem disponíveis
        if y_proba is not None:
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        # Matriz de confusão
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, 
            target_names=['Benigno', 'Maligno'],
            output_dict=True
        )
        
        self.results[model_name] = metrics
        
        return metrics
    
    def print_metrics(self, model_name):
        """
        Imprime as métricas de um modelo
        
        Args:
            model_name (str): Nome do modelo
        """
        metrics = self.results[model_name]
        
        print(f"\n{'='*50}")
        print(f"MÉTRICAS - {model_name}")
        print('='*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print("\nMatriz de Confusão:")
        print(metrics['confusion_matrix'])
        
    def compare_models(self):
        """
        Compara todos os modelos avaliados
        
        Returns:
            pd.DataFrame: DataFrame com comparação das métricas
        """
        comparison = []
        
        for model_name, metrics in self.results.items():
            row = {
                'Modelo': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            }
            
            if 'roc_auc' in metrics:
                row['ROC-AUC'] = metrics['roc_auc']
            
            comparison.append(row)
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
        
        return df_comparison
    
    def plot_confusion_matrix(self, model_name, figsize=(8, 6)):
        """
        Plota a matriz de confusão
        
        Args:
            model_name (str): Nome do modelo
            figsize (tuple): Tamanho da figura
        """
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benigno', 'Maligno'],
            yticklabels=['Benigno', 'Maligno']
        )
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.ylabel('Valor Real')
        plt.xlabel('Predição')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_roc_curve(self, model_name, y_true, y_proba):
        """
        Plota a curva ROC
        
        Args:
            model_name (str): Nome do modelo
            y_true: Valores verdadeiros
            y_proba: Probabilidades
        """
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title(f'Curva ROC - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_comparison(self, metric='f1_score'):
        """
        Plota comparação entre modelos
        
        Args:
            metric (str): Métrica para comparar
        """
        df_comparison = self.compare_models()
        
        metric_map = {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1-Score',
            'roc_auc': 'ROC-AUC'
        }
        
        metric_col = metric_map.get(metric, 'F1-Score')
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_comparison, x='Modelo', y=metric_col, palette='viridis')
        plt.title(f'Comparação de Modelos - {metric_col}')
        plt.ylabel(metric_col)
        plt.xlabel('Modelo')
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_feature_importance(self, feature_names, importance_values, model_name, top_n=20):
        """
        Plota importância das features
        
        Args:
            feature_names (list): Nomes das features
            importance_values (np.array): Valores de importância
            model_name (str): Nome do modelo
            top_n (int): Número de features mais importantes a mostrar
        """
        # Cria DataFrame com importâncias
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        })
        
        # Ordena por importância
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Seleciona top_n
        importance_df = importance_df.head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='Feature', x='Importance', palette='viridis')
        plt.title(f'Top {top_n} Features Mais Importantes - {model_name}')
        plt.xlabel('Importância')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        return plt.gcf()

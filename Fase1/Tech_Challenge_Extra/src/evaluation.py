"""
Módulo de avaliação de modelos CNN.
Contém funções para calcular métricas, gerar visualizações e interpretar resultados.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, roc_auc_score, precision_recall_curve,
    f1_score, accuracy_score
)
import pandas as pd


def evaluate_model(model, test_generator, class_names=['NORMAL', 'PNEUMONIA']):
    """
    Avalia o modelo no conjunto de teste e retorna métricas.
    
    Args:
        model: Modelo treinado
        test_generator: Gerador de dados de teste
        class_names: Nomes das classes
        
    Returns:
        dict com métricas e predições
    """
    print("🔍 Avaliando modelo no conjunto de teste...\n")
    
    # Reset do gerador
    test_generator.reset()
    
    # Fazer predições
    predictions_prob = model.predict(test_generator, verbose=1)
    predictions = (predictions_prob > 0.5).astype(int).flatten()
    
    # Labels verdadeiros
    y_true = test_generator.classes
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    auc = roc_auc_score(y_true, predictions_prob)
    
    print(f"\n📊 Métricas Gerais:")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"{'='*50}\n")
    
    # Relatório de classificação
    print("📋 Relatório Detalhado por Classe:\n")
    print(classification_report(y_true, predictions, 
                                target_names=class_names,
                                digits=4))
    
    return {
        'predictions': predictions,
        'predictions_prob': predictions_prob,
        'y_true': y_true,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc
    }


def plot_confusion_matrix(y_true, predictions, class_names=['NORMAL', 'PNEUMONIA'], 
                         save_path=None):
    """
    Plota matriz de confusão.
    
    Args:
        y_true: Labels verdadeiros
        predictions: Predições do modelo
        class_names: Nomes das classes
        save_path: Caminho para salvar figura (opcional)
    """
    cm = confusion_matrix(y_true, predictions)
    
    # Calcular percentuais
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Criar figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Matriz de confusão (valores absolutos)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Contagem'})
    ax1.set_title('Matriz de Confusão (Contagem)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Valor Real', fontsize=12)
    ax1.set_xlabel('Valor Predito', fontsize=12)
    
    # Matriz de confusão (percentuais)
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Percentual (%)'})
    ax2.set_title('Matriz de Confusão (Percentual)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Valor Real', fontsize=12)
    ax2.set_xlabel('Valor Predito', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figura salva em: {save_path}")
    
    plt.show()
    
    # Análise textual
    tn, fp, fn, tp = cm.ravel()
    print(f"\n📊 Análise da Matriz de Confusão:")
    print(f"{'='*50}")
    print(f"Verdadeiros Negativos (TN): {tn:4d}  - Correto: NORMAL como NORMAL")
    print(f"Falsos Positivos (FP):      {fp:4d}  - Erro: NORMAL como PNEUMONIA")
    print(f"Falsos Negativos (FN):      {fn:4d}  - Erro: PNEUMONIA como NORMAL ⚠️")
    print(f"Verdadeiros Positivos (TP): {tp:4d}  - Correto: PNEUMONIA como PNEUMONIA")
    print(f"{'='*50}")
    print(f"\n⚠️ IMPORTANTE: Falsos Negativos são críticos em medicina!")
    print(f"   Não detectar uma pneumonia real pode ter consequências graves.\n")


def plot_roc_curve(y_true, predictions_prob, save_path=None):
    """
    Plota curva ROC.
    
    Args:
        y_true: Labels verdadeiros
        predictions_prob: Probabilidades preditas
        save_path: Caminho para salvar figura (opcional)
    """
    fpr, tpr, thresholds = roc_curve(y_true, predictions_prob)
    auc = roc_auc_score(y_true, predictions_prob)
    
    plt.figure(figsize=(10, 7))
    
    # Linha do modelo
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {auc:.4f})')
    
    # Linha de referência (classificador aleatório)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier (AUC = 0.50)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
    plt.title('Curva ROC - Receiver Operating Characteristic', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figura salva em: {save_path}")
    
    plt.show()
    
    print(f"\n📈 Interpretação da Curva ROC:")
    print(f"{'='*50}")
    print(f"AUC = {auc:.4f}")
    if auc >= 0.9:
        print("   Excelente capacidade de discriminação! 🌟")
    elif auc >= 0.8:
        print("   Boa capacidade de discriminação! ✅")
    elif auc >= 0.7:
        print("   Capacidade razoável de discriminação. 👍")
    else:
        print("   Capacidade limitada de discriminação. ⚠️")
    print(f"{'='*50}\n")


def plot_precision_recall_curve(y_true, predictions_prob, save_path=None):
    """
    Plota curva Precision-Recall.
    
    Args:
        y_true: Labels verdadeiros
        predictions_prob: Probabilidades preditas
        save_path: Caminho para salvar figura (opcional)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, predictions_prob)
    
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall (Sensibilidade)', fontsize=12)
    plt.ylabel('Precision (Precisão)', fontsize=12)
    plt.title('Curva Precision-Recall', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    # Adicionar ponto do threshold 0.5
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    plt.plot(recall[idx_05], precision[idx_05], 'ro', markersize=10,
             label=f'Threshold=0.5 (P={precision[idx_05]:.2f}, R={recall[idx_05]:.2f})')
    plt.legend(fontsize=11)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figura salva em: {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plota histórico de treinamento (loss e métricas).
    
    Args:
        history: Objeto History do Keras
        save_path: Caminho para salvar figura (opcional)
    """
    # Extrair métricas
    metrics = [m for m in history.history.keys() if not m.startswith('val_')]
    
    # Criar subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics[:4]):  # Máximo 4 métricas
        ax = axes[idx]
        
        train_values = history.history[metric]
        val_metric = f'val_{metric}'
        
        if val_metric in history.history:
            val_values = history.history[val_metric]
            
            ax.plot(train_values, label=f'Treino', linewidth=2)
            ax.plot(val_values, label=f'Validação', linewidth=2)
            
            # Destacar melhor época
            best_epoch = np.argmin(val_values) if metric == 'loss' else np.argmax(val_values)
            best_value = val_values[best_epoch]
            ax.plot(best_epoch, best_value, 'r*', markersize=15,
                   label=f'Melhor (época {best_epoch+1})')
        else:
            ax.plot(train_values, label=f'Treino', linewidth=2)
        
        ax.set_xlabel('Época', fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(f'{metric.upper()} ao longo do treinamento', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figura salva em: {save_path}")
    
    plt.show()


def compare_models_results(results_dict):
    """
    Compara resultados de múltiplos modelos.
    
    Args:
        results_dict: Dict com {nome_modelo: dict_metricas}
    """
    # Preparar dados
    data = []
    for model_name, metrics in results_dict.items():
        data.append({
            'Modelo': model_name,
            'Accuracy': metrics['accuracy'],
            'F1-Score': metrics['f1_score'],
            'AUC-ROC': metrics['auc']
        })
    
    df = pd.DataFrame(data)
    
    # Plotar comparação
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.25
    
    ax.bar(x - width, df['Accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x, df['F1-Score'], width, label='F1-Score', alpha=0.8)
    ax.bar(x + width, df['AUC-ROC'], width, label='AUC-ROC', alpha=0.8)
    
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparação de Desempenho dos Modelos', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Modelo'], rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Tabela
    print("\n📊 Comparação de Modelos:")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)


if __name__ == "__main__":
    print("🧪 Módulo de avaliação carregado com sucesso!")
    print("✅ Funções disponíveis:")
    print("   - evaluate_model()")
    print("   - plot_confusion_matrix()")
    print("   - plot_roc_curve()")
    print("   - plot_precision_recall_curve()")
    print("   - plot_training_history()")
    print("   - compare_models_results()")

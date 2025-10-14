"""
Script Principal - Tech Challenge Fase 1
Diagnóstico de Câncer de Mama
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from preprocessing import DataPreprocessor
from models import ModelTrainer
from evaluation import ModelEvaluator


def main():
    """Função principal do pipeline de ML"""
    
    print("="*70)
    print("TECH CHALLENGE - FASE 1")
    print("Sistema de Diagnóstico de Câncer de Mama")
    print("="*70)
    
    # 1. Carregamento dos dados
    print("\n1. Carregando dados...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target
    df['diagnosis_label'] = df['diagnosis'].map({0: 'M', 1: 'B'})
    
    print(f"   ✓ Dataset carregado: {df.shape[0]} amostras")
    
    # 2. Pré-processamento
    print("\n2. Pré-processamento dos dados...")
    preprocessor = DataPreprocessor()
    
    X = df.drop(['diagnosis', 'diagnosis_label'], axis=1)
    y = df['diagnosis_label']
    
    y_encoded = preprocessor.encode_target(y)
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y_encoded, test_size=0.2, val_size=0.2, random_state=42
    )
    
    X_train_scaled, X_val_scaled = preprocessor.scale_features(X_train, X_val)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    print("   ✓ Pré-processamento concluído")
    
    # 3. Treinamento dos modelos
    print("\n3. Treinamento dos modelos...")
    trainer = ModelTrainer()
    trainer.initialize_models()
    models = trainer.train_all_models(X_train_scaled, y_train, tune_hyperparameters=False)
    print("   ✓ Modelos treinados")
    
    # 4. Avaliação
    print("\n4. Avaliação dos modelos...")
    evaluator = ModelEvaluator()
    
    for model_name in models.keys():
        y_pred = trainer.predict(model_name, X_test_scaled)
        y_proba = trainer.predict_proba(model_name, X_test_scaled)
        evaluator.evaluate_model(model_name, y_test, y_pred, y_proba)
    
    # 5. Resultados
    print("\n5. Resultados:")
    print("="*70)
    comparison_df = evaluator.compare_models()
    print(comparison_df.to_string(index=False))
    
    best_model_name = comparison_df.iloc[0]['Modelo']
    print(f"\n🏆 Melhor modelo: {best_model_name}")
    print(f"   F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
    
    # 6. Salvar modelo
    print("\n6. Salvando modelo...")
    import joblib
    model = trainer.best_models[best_model_name]
    joblib.dump(model, f'../results/{best_model_name.replace(" ", "_").lower()}_model.pkl')
    joblib.dump(preprocessor.scaler, '../results/scaler.pkl')
    print("   ✓ Modelo salvo")
    
    print("\n" + "="*70)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*70)


if __name__ == "__main__":
    main()

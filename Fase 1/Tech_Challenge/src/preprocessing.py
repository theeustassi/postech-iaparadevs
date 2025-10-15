"""
Módulo de Pré-processamento de Dados
Tech Challenge - Fase 1
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Classe para pré-processamento de dados médicos"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_data(self, filepath):
        """
        Carrega os dados do arquivo CSV
        
        Args:
            filepath (str): Caminho do arquivo CSV
            
        Returns:
            pd.DataFrame: Dados carregados
        """
        df = pd.read_csv(filepath)
        print(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
        return df
    
    def explore_data(self, df):
        """
        Realiza exploração básica dos dados
        
        Args:
            df (pd.DataFrame): DataFrame a ser explorado
            
        Returns:
            dict: Dicionário com informações sobre os dados
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'describe': df.describe()
        }
        
        print("=" * 50)
        print("EXPLORAÇÃO DOS DADOS")
        print("=" * 50)
        print(f"Dimensões: {info['shape']}")
        print(f"Duplicados: {info['duplicates']}")
        print(f"\nValores ausentes por coluna:")
        for col, missing in info['missing_values'].items():
            if missing > 0:
                print(f"  {col}: {missing}")
        
        return info
    
    def clean_data(self, df):
        """
        Limpa os dados removendo valores ausentes e duplicados
        
        Args:
            df (pd.DataFrame): DataFrame a ser limpo
            
        Returns:
            pd.DataFrame: DataFrame limpo
        """
        df_clean = df.copy()
        
        # Remove colunas com muitos valores ausentes (>50%)
        threshold = len(df_clean) * 0.5
        df_clean = df_clean.dropna(thresh=threshold, axis=1)
        
        # Remove linhas com valores ausentes
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        removed_rows = initial_rows - len(df_clean)
        
        # Remove duplicados
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        
        print(f"\nLimpeza de dados:")
        print(f"  Linhas removidas (valores ausentes): {removed_rows}")
        print(f"  Linhas removidas (duplicados): {removed_duplicates}")
        print(f"  Total de linhas após limpeza: {len(df_clean)}")
        
        return df_clean
    
    def encode_target(self, y):
        """
        Codifica a variável target (M/B para 1/0)
        
        Args:
            y (pd.Series): Variável target
            
        Returns:
            np.array: Target codificada
        """
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"\nClasses codificadas:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {label} -> {i}")
        return y_encoded
    
    def prepare_features(self, df, target_column):
        """
        Prepara features e target para modelagem
        
        Args:
            df (pd.DataFrame): DataFrame completo
            target_column (str): Nome da coluna target
            
        Returns:
            tuple: (X, y) features e target
        """
        # Separa features e target
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Remove colunas não numéricas ou identificadores
        id_columns = ['id', 'ID', 'Id']
        X = X.select_dtypes(include=[np.number])
        X = X.drop(columns=[col for col in id_columns if col in X.columns], errors='ignore')
        
        self.feature_names = X.columns.tolist()
        
        print(f"\nFeatures selecionadas: {len(self.feature_names)}")
        
        return X, y
    
    def scale_features(self, X_train, X_test):
        """
        Aplica escalonamento nas features
        
        Args:
            X_train (pd.DataFrame): Features de treino
            X_test (pd.DataFrame): Features de teste
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Converte de volta para DataFrame
        X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, 
            columns=X_test.columns,
            index=X_test.index
        )
        
        print("\nFeatures escalonadas (StandardScaler)")
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """
        Divide os dados em treino, validação e teste
        
        Args:
            X: Features
            y: Target
            test_size (float): Proporção do conjunto de teste
            val_size (float): Proporção do conjunto de validação
            random_state (int): Seed para reprodutibilidade
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Primeiro split: treino+val vs teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Segundo split: treino vs validação
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"\nDivisão dos dados:")
        print(f"  Treino: {len(X_train)} amostras ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validação: {len(X_val)} amostras ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Teste: {len(X_test)} amostras ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

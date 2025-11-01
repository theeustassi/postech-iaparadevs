"""
Módulo de Modelos de Machine Learning
Tech Challenge - Fase 1
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import numpy as np


class ModelTrainer:
    """Classe para treinamento de múltiplos modelos"""
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.training_history = {}
        
    def initialize_models(self):
        """Inicializa os modelos de classificação"""
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(
                random_state=42,
                probability=True
            )
        }
        
        print(f"Modelos inicializados: {list(self.models.keys())}")
        
    def get_hyperparameter_grids(self):
        """Define os grids de hiperparâmetros para cada modelo"""
        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            'Decision Tree': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        return param_grids
    
    def train_model(self, model_name, X_train, y_train, tune_hyperparameters=False):
        """
        Treina um modelo específico
        
        Args:
            model_name (str): Nome do modelo
            X_train: Features de treino
            y_train: Target de treino
            tune_hyperparameters (bool): Se True, realiza busca de hiperparâmetros
            
        Returns:
            model: Modelo treinado
        """
        model = self.models[model_name]
        
        if tune_hyperparameters:
            print(f"\nTreinando {model_name} com GridSearchCV...")
            param_grids = self.get_hyperparameter_grids()
            
            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            print(f"Melhores parâmetros: {grid_search.best_params_}")
            print(f"Melhor score (CV): {grid_search.best_score_:.4f}")
            
            self.best_models[model_name] = best_model
            self.training_history[model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            return best_model
        else:
            print(f"\nTreinando {model_name}...")
            model.fit(X_train, y_train)
            self.best_models[model_name] = model
            return model
    
    def train_all_models(self, X_train, y_train, tune_hyperparameters=False):
        """
        Treina todos os modelos
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            tune_hyperparameters (bool): Se True, realiza busca de hiperparâmetros
            
        Returns:
            dict: Dicionário com todos os modelos treinados
        """
        print("\n" + "="*50)
        print("TREINAMENTO DOS MODELOS")
        print("="*50)
        
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train, tune_hyperparameters)
        
        print("\nTodos os modelos foram treinados!")
        return self.best_models
    
    def predict(self, model_name, X):
        """
        Realiza predições com um modelo específico
        
        Args:
            model_name (str): Nome do modelo
            X: Features para predição
            
        Returns:
            np.array: Predições
        """
        model = self.best_models[model_name]
        return model.predict(X)
    
    def predict_proba(self, model_name, X):
        """
        Retorna probabilidades de predição
        
        Args:
            model_name (str): Nome do modelo
            X: Features para predição
            
        Returns:
            np.array: Probabilidades
        """
        model = self.best_models[model_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            # Para modelos sem predict_proba, usa decision_function
            decision = model.decision_function(X)
            # Converte para probabilidades usando sigmoid
            proba = 1 / (1 + np.exp(-decision))
            return np.vstack([1 - proba, proba]).T
    
    def get_feature_importance(self, model_name):
        """
        Obtém a importância das features para modelos que suportam
        
        Args:
            model_name (str): Nome do modelo
            
        Returns:
            np.array or None: Importância das features
        """
        model = self.best_models[model_name]
        
        # Para modelos com feature_importances_ (Random Forest, Decision Tree)
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        
        # Para modelos com coeficientes (Logistic Regression, SVM Linear)
        elif hasattr(model, 'coef_'):
            # Retorna o valor absoluto dos coeficientes
            if len(model.coef_.shape) > 1:
                return np.abs(model.coef_[0])
            else:
                return np.abs(model.coef_)
        
        # Se nenhum dos anteriores, retorna None (ex: SVM RBF, KNN)
        else:
            return None
    
    def get_permutation_importance(self, model_name, X_test, y_test, n_repeats=10):
        """
        Calcula Permutation Importance (funciona com QUALQUER modelo)
        Ideal para SVM, KNN e outros que não têm feature importance nativa
        
        Args:
            model_name (str): Nome do modelo
            X_test: Features de teste
            y_test: Target de teste
            n_repeats (int): Número de repetições (padrão: 10)
            
        Returns:
            np.array: Importância das features por permutação
        """
        model = self.best_models[model_name]
        
        perm_importance = permutation_importance(
            model, X_test, y_test, 
            n_repeats=n_repeats, 
            random_state=42,
            n_jobs=-1
        )
        
        return perm_importance.importances_mean

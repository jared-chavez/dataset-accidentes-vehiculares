#!/usr/bin/env python3
"""
Modelos de Clasificación para Predicción de Severidad de Accidentes
=====================================================================

Este módulo implementa varios algoritmos de clasificación para predecir
la severidad de accidentes (Minor, Serious, Critical).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# XGBoost se importará dinámicamente cuando se necesite
XGBOOST_AVAILABLE = None  # Se verificará cuando se llame

class ClassificationModels:
    """
    Clase para entrenar y evaluar modelos de clasificación
    """
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.predictions = {}
        
    def train_logistic_regression(self, X_train, y_train, X_val, y_val, optimize=True):
        """
        Entrena un modelo de Regresión Logística
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            optimize: Si True, optimiza hiperparámetros
            
        Returns:
            LogisticRegression: Modelo entrenado
        """
        print("\nEntrenando Logistic Regression...")
        
        if optimize:
            # Grid de hiperparámetros
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [100, 200, 500]
            }
            
            # Usar RandomizedSearchCV para ser más eficiente
            base_model = LogisticRegression(random_state=42, multi_class='multinomial')
            search = RandomizedSearchCV(
                base_model, param_grid, 
                n_iter=20, cv=5, 
                scoring='f1_macro',
                n_jobs=-1, random_state=42
            )
            search.fit(X_train, y_train)
            
            model = search.best_estimator_
            self.best_params['LogisticRegression'] = search.best_params_
            print(f"Mejores parámetros: {search.best_params_}")
        else:
            model = LogisticRegression(
                random_state=42, 
                multi_class='multinomial',
                max_iter=500
            )
            model.fit(X_train, y_train)
        
        # Evaluar en validación
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Accuracy en validación: {accuracy:.4f}")
        
        self.models['LogisticRegression'] = model
        return model
    
    def train_random_forest(self, X_train, y_train, X_val, y_val, optimize=True):
        """
        Entrena un modelo de Random Forest
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            optimize: Si True, optimiza hiperparámetros
            
        Returns:
            RandomForestClassifier: Modelo entrenado
        """
        print("\nEntrenando Random Forest Classifier...")
        
        if optimize:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=30, cv=5,
                scoring='f1_macro',
                n_jobs=-1, random_state=42
            )
            search.fit(X_train, y_train)
            
            model = search.best_estimator_
            self.best_params['RandomForest'] = search.best_params_
            print(f"Mejores parámetros: {search.best_params_}")
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        # Evaluar en validación
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Accuracy en validación: {accuracy:.4f}")
        
        self.models['RandomForest'] = model
        return model
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val, optimize=True):
        """
        Entrena un modelo de Gradient Boosting (XGBoost)
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            optimize: Si True, optimiza hiperparámetros
            
        Returns:
            xgb.XGBClassifier: Modelo entrenado
        """
        # Intentar importar XGBoost dinámicamente
        global XGBOOST_AVAILABLE
        if XGBOOST_AVAILABLE is None:
            try:
                import xgboost as xgb
                XGBOOST_AVAILABLE = True
            except Exception as e:
                XGBOOST_AVAILABLE = False
                print(f"\nXGBoost no disponible ({type(e).__name__}), omitiendo Gradient Boosting")
                return None
        
        if not XGBOOST_AVAILABLE:
            print("\nXGBoost no disponible, omitiendo Gradient Boosting")
            return None
        
        # Importar ahora que sabemos que está disponible
        import xgboost as xgb
        
        print("\nEntrenando Gradient Boosting (XGBoost)...")
        
        if optimize:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            base_model = xgb.XGBClassifier(
                random_state=42,
                objective='multi:softprob',
                eval_metric='mlogloss',
                n_jobs=-1
            )
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=30, cv=5,
                scoring='f1_macro',
                n_jobs=-1, random_state=42
            )
            search.fit(X_train, y_train)
            
            model = search.best_estimator_
            self.best_params['XGBoost'] = search.best_params_
            print(f"Mejores parámetros: {search.best_params_}")
        else:
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                objective='multi:softprob',
                eval_metric='mlogloss',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        # Evaluar en validación
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Accuracy en validación: {accuracy:.4f}")
        
        self.models['XGBoost'] = model
        return model
    
    def train_svm(self, X_train, y_train, X_val, y_val, optimize=True):
        """
        Entrena un modelo de Support Vector Machine
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            optimize: Si True, optimiza hiperparámetros
            
        Returns:
            SVC: Modelo entrenado
        """
        print("\nEntrenando Support Vector Machine...")
        
        if optimize:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            }
            
            base_model = SVC(random_state=42, probability=True)
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=20, cv=3,  # Reducir CV porque SVM es lento
                scoring='f1_macro',
                n_jobs=-1, random_state=42
            )
            search.fit(X_train, y_train)
            
            model = search.best_estimator_
            self.best_params['SVM'] = search.best_params_
            print(f"Mejores parámetros: {search.best_params_}")
        else:
            model = SVC(
                C=1.0,
                kernel='rbf',
                random_state=42,
                probability=True
            )
            model.fit(X_train, y_train)
        
        # Evaluar en validación
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Accuracy en validación: {accuracy:.4f}")
        
        self.models['SVM'] = model
        return model
    
    def train_all_models(self, X_train, y_train, X_val, y_val, optimize=True):
        """
        Entrena todos los modelos de clasificación
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            optimize: Si True, optimiza hiperparámetros
        """
        print("="*60)
        print("ENTRENANDO TODOS LOS MODELOS DE CLASIFICACIÓN")
        print("="*60)
        
        self.train_logistic_regression(X_train, y_train, X_val, y_val, optimize)
        self.train_random_forest(X_train, y_train, X_val, y_val, optimize)
        
        # Intentar entrenar Gradient Boosting (puede no estar disponible)
        gb_model = self.train_gradient_boosting(X_train, y_train, X_val, y_val, optimize)
        if gb_model is None:
            print("Gradient Boosting omitido")
        
        # SVM es opcional (puede ser muy lento)
        # self.train_svm(X_train, y_train, X_val, y_val, optimize)
        
        print("\nModelos disponibles entrenados")
        print("="*60)
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evalúa un modelo en el conjunto de test
        
        Args:
            model: Modelo entrenado
            X_test, y_test: Datos de prueba
            model_name: Nombre del modelo
            
        Returns:
            dict: Diccionario con métricas
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'predictions_proba': y_pred_proba
        }
        
        self.predictions[model_name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        return metrics
    
    def get_feature_importance(self, model, feature_names, model_name):
        """
        Obtiene la importancia de features para modelos que la soportan
        
        Args:
            model: Modelo entrenado
            feature_names: Lista de nombres de features
            model_name: Nombre del modelo
            
        Returns:
            pd.DataFrame: DataFrame con importancia de features
        """
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        elif hasattr(model, 'coef_'):
            # Para Logistic Regression, usar coeficientes promedio
            coef_mean = np.mean(np.abs(model.coef_), axis=0)
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': coef_mean
            }).sort_values('importance', ascending=False)
            return importance
        else:
            return None


#!/usr/bin/env python3
"""
Modelos de Regresión para Predicción de Fatalidades
====================================================

Este módulo implementa varios algoritmos de regresión para predecir
el número de fatalidades en accidentes vehiculares.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')

# XGBoost se importará dinámicamente cuando se necesite
XGBOOST_AVAILABLE = None  # Se verificará cuando se llame

class RegressionModels:
    """
    Clase para entrenar y evaluar modelos de regresión
    """
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.predictions = {}
        
    def train_linear_regression(self, X_train, y_train, X_val, y_val):
        """
        Entrena un modelo de Regresión Lineal
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            
        Returns:
            LinearRegression: Modelo entrenado
        """
        print("\nEntrenando Linear Regression...")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluar en validación
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"MAE en validación: {mae:.4f}")
        
        self.models['LinearRegression'] = model
        return model
    
    def train_ridge_regression(self, X_train, y_train, X_val, y_val, optimize=True):
        """
        Entrena un modelo de Ridge Regression
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            optimize: Si True, optimiza hiperparámetros
            
        Returns:
            Ridge: Modelo entrenado
        """
        print("\nEntrenando Ridge Regression...")
        
        if optimize:
            param_grid = {
                'alpha': [0.1, 1, 10, 100, 1000]
            }
            
            base_model = Ridge(random_state=42)
            search = GridSearchCV(
                base_model, param_grid,
                cv=5, scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            
            model = search.best_estimator_
            self.best_params['Ridge'] = search.best_params_
            print(f"Mejor alpha: {search.best_params_['alpha']}")
        else:
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_train, y_train)
        
        # Evaluar en validación
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"MAE en validación: {mae:.4f}")
        
        self.models['Ridge'] = model
        return model
    
    def train_random_forest_regressor(self, X_train, y_train, X_val, y_val, optimize=True):
        """
        Entrena un modelo de Random Forest Regressor
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            optimize: Si True, optimiza hiperparámetros
            
        Returns:
            RandomForestRegressor: Modelo entrenado
        """
        print("\nEntrenando Random Forest Regressor...")
        
        if optimize:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=30, cv=5,
                scoring='neg_mean_absolute_error',
                n_jobs=-1, random_state=42
            )
            search.fit(X_train, y_train)
            
            model = search.best_estimator_
            self.best_params['RandomForest'] = search.best_params_
            print(f"Mejores parámetros: {search.best_params_}")
        else:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        # Evaluar en validación
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"MAE en validación: {mae:.4f}")
        
        self.models['RandomForest'] = model
        return model
    
    def train_gradient_boosting_regressor(self, X_train, y_train, X_val, y_val, optimize=True):
        """
        Entrena un modelo de Gradient Boosting Regressor (XGBoost)
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            optimize: Si True, optimiza hiperparámetros
            
        Returns:
            xgb.XGBRegressor: Modelo entrenado
        """
        # Intentar importar XGBoost dinámicamente
        global XGBOOST_AVAILABLE
        if XGBOOST_AVAILABLE is None:
            try:
                import xgboost as xgb
                XGBOOST_AVAILABLE = True
            except Exception as e:
                XGBOOST_AVAILABLE = False
                print(f"\nXGBoost no disponible ({type(e).__name__}), omitiendo Gradient Boosting Regressor")
                return None
        
        if not XGBOOST_AVAILABLE:
            print("\nXGBoost no disponible, omitiendo Gradient Boosting Regressor")
            return None
        
        # Importar ahora que sabemos que está disponible
        import xgboost as xgb
        
        print("\nEntrenando Gradient Boosting Regressor (XGBoost)...")
        
        if optimize:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            base_model = xgb.XGBRegressor(
                random_state=42,
                n_jobs=-1
            )
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=30, cv=5,
                scoring='neg_mean_absolute_error',
                n_jobs=-1, random_state=42
            )
            search.fit(X_train, y_train)
            
            model = search.best_estimator_
            self.best_params['XGBoost'] = search.best_params_
            print(f"Mejores parámetros: {search.best_params_}")
        else:
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        # Evaluar en validación
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"MAE en validación: {mae:.4f}")
        
        self.models['XGBoost'] = model
        return model
    
    def train_all_models(self, X_train, y_train, X_val, y_val, optimize=True):
        """
        Entrena todos los modelos de regresión
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            optimize: Si True, optimiza hiperparámetros
        """
        print("="*60)
        print("ENTRENANDO TODOS LOS MODELOS DE REGRESIÓN")
        print("="*60)
        
        self.train_linear_regression(X_train, y_train, X_val, y_val)
        self.train_ridge_regression(X_train, y_train, X_val, y_val, optimize)
        self.train_random_forest_regressor(X_train, y_train, X_val, y_val, optimize)
        
        # Intentar entrenar Gradient Boosting (puede no estar disponible)
        gb_model = self.train_gradient_boosting_regressor(X_train, y_train, X_val, y_val, optimize)
        if gb_model is None:
            print("Gradient Boosting Regressor omitido")
        
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
        
        # Asegurar que las predicciones no sean negativas
        y_pred = np.maximum(y_pred, 0)
        
        # Calcular todas las métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # MAPE
        try:
            mape = mean_absolute_percentage_error(y_test, y_pred)
        except:
            mask = y_test != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            else:
                mape = np.nan
        
        # Error mediano
        median_ae = np.median(np.abs(y_test - y_pred))
        
        # Error medio y desviación estándar
        errors = y_pred - y_test
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        metrics = {
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'median_ae': median_ae,
            'mean_error': mean_error,
            'std_error': std_error,
            'predictions': y_pred
        }
        
        self.predictions[model_name] = {
            'y_pred': y_pred,
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
            # Para modelos lineales, usar coeficientes
            coef_abs = np.abs(model.coef_)
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': coef_abs
            }).sort_values('importance', ascending=False)
            return importance
        else:
            return None


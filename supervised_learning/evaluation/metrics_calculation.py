#!/usr/bin/env python3
"""
Cálculo de Métricas para Evaluación de Modelos
===============================================

Este módulo calcula y organiza todas las métricas de evaluación
para modelos de clasificación y regresión.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')

def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calcula todas las métricas de clasificación
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        y_pred_proba: Probabilidades predichas (opcional)
        
    Returns:
        dict: Diccionario con todas las métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    # Calcular métricas por clase
    classes = np.unique(y_true)
    for cls in classes:
        cls_mask = (y_true == cls)
        cls_pred = (y_pred == cls)
        
        metrics[f'precision_{cls}'] = precision_score(
            cls_mask, cls_pred, zero_division=0
        )
        metrics[f'recall_{cls}'] = recall_score(
            cls_mask, cls_pred, zero_division=0
        )
        metrics[f'f1_{cls}'] = f1_score(
            cls_mask, cls_pred, zero_division=0
        )
    
    return metrics

def calculate_regression_metrics(y_true, y_pred):
    """
    Calcula todas las métricas de regresión
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        
    Returns:
        dict: Diccionario con todas las métricas
    """
    # Asegurar que las predicciones no sean negativas
    y_pred = np.maximum(y_pred, 0)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE puede ser problemático si hay ceros
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except:
        # Calcular MAPE manualmente, evitando división por cero
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
    
    # Error mediano (robusto a outliers)
    median_ae = np.median(np.abs(y_true - y_pred))
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'median_ae': median_ae,
        'mean_error': np.mean(y_pred - y_true),
        'std_error': np.std(y_pred - y_true)
    }
    
    return metrics

def create_classification_metrics_table(all_metrics):
    """
    Crea una tabla comparativa de métricas de clasificación
    
    Args:
        all_metrics: Lista de diccionarios de métricas por modelo
        
    Returns:
        pd.DataFrame: DataFrame con métricas comparativas
    """
    rows = []
    for metrics in all_metrics:
        row = {
            'Modelo': metrics['model'],
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision (Macro)': f"{metrics['precision_macro']:.4f}",
            'Recall (Macro)': f"{metrics['recall_macro']:.4f}",
            'F1-Score (Macro)': f"{metrics['f1_macro']:.4f}",
            'Precision (Weighted)': f"{metrics['precision_weighted']:.4f}",
            'Recall (Weighted)': f"{metrics['recall_weighted']:.4f}",
            'F1-Score (Weighted)': f"{metrics['f1_weighted']:.4f}"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def create_regression_metrics_table(all_metrics):
    """
    Crea una tabla comparativa de métricas de regresión
    
    Args:
        all_metrics: Lista de diccionarios de métricas por modelo
        
    Returns:
        pd.DataFrame: DataFrame con métricas comparativas
    """
    rows = []
    for metrics in all_metrics:
        row = {
            'Modelo': metrics['model'],
            'MAE': f"{metrics['mae']:.4f}",
            'RMSE': f"{metrics['rmse']:.4f}",
            'R²': f"{metrics['r2']:.4f}",
            'MAPE (%)': f"{metrics['mape']:.2f}" if not np.isnan(metrics['mape']) else 'N/A',
            'Median AE': f"{metrics['median_ae']:.4f}",
            'Mean Error': f"{metrics['mean_error']:.4f}",
            'Std Error': f"{metrics['std_error']:.4f}"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def get_best_model_classification(all_metrics):
    """
    Identifica el mejor modelo de clasificación basado en F1-Score macro
    
    Args:
        all_metrics: Lista de diccionarios de métricas
        
    Returns:
        dict: Información del mejor modelo
    """
    best_model = max(all_metrics, key=lambda x: x['f1_macro'])
    return {
        'model_name': best_model['model'],
        'f1_macro': best_model['f1_macro'],
        'accuracy': best_model['accuracy'],
        'all_metrics': best_model
    }

def get_best_model_regression(all_metrics):
    """
    Identifica el mejor modelo de regresión basado en MAE
    
    Args:
        all_metrics: Lista de diccionarios de métricas
        
    Returns:
        dict: Información del mejor modelo
    """
    best_model = min(all_metrics, key=lambda x: x['mae'])
    return {
        'model_name': best_model['model'],
        'mae': best_model['mae'],
        'rmse': best_model['rmse'],
        'r2': best_model['r2'],
        'all_metrics': best_model
    }


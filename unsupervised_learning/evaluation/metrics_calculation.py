#!/usr/bin/env python3
"""
Cálculo de Métricas para Análisis No Supervisado
================================================

Este módulo calcula las métricas de evaluación para modelos no supervisados:
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- Inertia (para K-Means)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
import warnings
warnings.filterwarnings('ignore')

def calculate_clustering_metrics(X, labels):
    """
    Calcula todas las métricas de clustering para un conjunto de labels
    
    Args:
        X (pd.DataFrame o np.array): Features normalizadas
        labels (np.array): Etiquetas de clústeres
        
    Returns:
        dict: Diccionario con todas las métricas
    """
    # Filtrar outliers (labels == -1)
    mask = labels != -1
    if mask.sum() < len(labels):
        X_filtered = X[mask] if hasattr(X, 'iloc') else X[mask]
        labels_filtered = labels[mask]
    else:
        X_filtered = X
        labels_filtered = labels
    
    # Verificar que hay al menos 2 clústeres
    n_clusters = len(set(labels_filtered))
    if n_clusters < 2:
        return {
            'silhouette_score': np.nan,
            'calinski_harabasz_score': np.nan,
            'davies_bouldin_score': np.nan,
            'n_clusters': n_clusters,
            'n_samples': len(labels_filtered),
            'n_outliers': len(labels) - len(labels_filtered)
        }
    
    # Calcular métricas
    try:
        silhouette = silhouette_score(X_filtered, labels_filtered)
    except Exception as e:
        print(f"  Error calculando Silhouette Score: {e}")
        silhouette = np.nan
    
    try:
        calinski_harabasz = calinski_harabasz_score(X_filtered, labels_filtered)
    except Exception as e:
        print(f"  Error calculando Calinski-Harabasz Score: {e}")
        calinski_harabasz = np.nan
    
    try:
        davies_bouldin = davies_bouldin_score(X_filtered, labels_filtered)
    except Exception as e:
        print(f"  Error calculando Davies-Bouldin Score: {e}")
        davies_bouldin = np.nan
    
    metrics = {
        'silhouette_score': silhouette,
        'calinski_harabasz_score': calinski_harabasz,
        'davies_bouldin_score': davies_bouldin,
        'n_clusters': n_clusters,
        'n_samples': len(labels_filtered),
        'n_outliers': len(labels) - len(labels_filtered)
    }
    
    return metrics

def create_metrics_comparison_table(metrics_list):
    """
    Crea una tabla comparativa de métricas de diferentes modelos
    
    Args:
        metrics_list (list): Lista de diccionarios con métricas de cada modelo
        
    Returns:
        pd.DataFrame: Tabla comparativa
    """
    df = pd.DataFrame(metrics_list)
    
    # Ordenar por Silhouette Score (mayor es mejor)
    if 'silhouette_score' in df.columns:
        df = df.sort_values('silhouette_score', ascending=False, na_position='last')
    
    return df

def get_best_model(metrics_list, metric='silhouette_score', higher_is_better=True):
    """
    Identifica el mejor modelo basado en una métrica
    
    Args:
        metrics_list (list): Lista de diccionarios con métricas
        metric (str): Nombre de la métrica a usar
        higher_is_better (bool): Si True, mayor es mejor; si False, menor es mejor
        
    Returns:
        dict: Métricas del mejor modelo
    """
    # Filtrar modelos con métricas válidas
    valid_metrics = [m for m in metrics_list if metric in m and not np.isnan(m[metric])]
    
    if not valid_metrics:
        print(f"  Advertencia: No hay modelos con métrica {metric} válida")
        return metrics_list[0] if metrics_list else None
    
    if higher_is_better:
        best = max(valid_metrics, key=lambda x: x[metric])
    else:
        best = min(valid_metrics, key=lambda x: x[metric])
    
    return best

def calculate_optimal_k_metrics(X, k_range=range(2, 11)):
    """
    Calcula métricas para diferentes valores de k (para K-Means)
    
    Args:
        X (pd.DataFrame o np.array): Features normalizadas
        k_range (range): Rango de valores de k a probar
        
    Returns:
        pd.DataFrame: DataFrame con métricas por k
    """
    from sklearn.cluster import KMeans
    
    results = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        metrics = calculate_clustering_metrics(X, labels)
        metrics['k'] = k
        metrics['inertia'] = kmeans.inertia_
        
        results.append(metrics)
    
    return pd.DataFrame(results)

def calculate_dbscan_parameter_sensitivity(X, eps_range, min_samples_range):
    """
    Calcula métricas para diferentes combinaciones de parámetros DBSCAN
    
    Args:
        X (pd.DataFrame o np.array): Features normalizadas
        eps_range (list): Lista de valores de eps a probar
        min_samples_range (list): Lista de valores de min_samples a probar
        
    Returns:
        pd.DataFrame: DataFrame con métricas por combinación de parámetros
    """
    from sklearn.cluster import DBSCAN
    
    results = []
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            metrics = calculate_clustering_metrics(X, labels)
            metrics['eps'] = eps
            metrics['min_samples'] = min_samples
            
            results.append(metrics)
    
    return pd.DataFrame(results)


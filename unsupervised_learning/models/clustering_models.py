#!/usr/bin/env python3
"""
Modelos de Clustering
====================

Este módulo implementa los algoritmos de clustering:
- K-Means
- DBSCAN
- Clustering Jerárquico (Agglomerative)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

class ClusteringModels:
    """
    Clase que encapsula los modelos de clustering y sus métodos
    """
    
    def __init__(self, random_state=42):
        """
        Inicializa la clase de modelos de clustering
        
        Args:
            random_state (int): Semilla aleatoria para reproducibilidad
        """
        self.random_state = random_state
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def fit_kmeans(self, X, n_clusters=3, init='k-means++', n_init=10, max_iter=300):
        """
        Entrena un modelo K-Means
        
        Args:
            X (pd.DataFrame o np.array): Features normalizadas
            n_clusters (int): Número de clústeres
            init (str): Método de inicialización
            n_init (int): Número de inicializaciones
            max_iter (int): Máximo número de iteraciones
            
        Returns:
            KMeans: Modelo entrenado
        """
        print(f"Entrenando K-Means con {n_clusters} clústeres...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=self.random_state
        )
        
        kmeans.fit(X)
        
        # Guardar modelo y predicciones
        self.models['K-Means'] = kmeans
        labels = kmeans.predict(X)
        self.predictions['K-Means'] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_,
            'centroids': kmeans.cluster_centers_
        }
        
        print(f"  K-Means entrenado. Inertia: {kmeans.inertia_:.2f}")
        
        return kmeans
    
    def fit_dbscan(self, X, eps=0.5, min_samples=5):
        """
        Entrena un modelo DBSCAN
        
        Args:
            X (pd.DataFrame o np.array): Features normalizadas
            eps (float): Distancia máxima entre muestras en el mismo clúster
            min_samples (int): Número mínimo de muestras en un clúster
            
        Returns:
            DBSCAN: Modelo entrenado
        """
        print(f"Entrenando DBSCAN (eps={eps}, min_samples={min_samples})...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Guardar modelo y predicciones
        self.models['DBSCAN'] = dbscan
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        self.predictions['DBSCAN'] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps,
            'min_samples': min_samples
        }
        
        print(f"  DBSCAN entrenado. Clústeres encontrados: {n_clusters}, Outliers: {n_noise}")
        
        return dbscan
    
    def fit_hierarchical(self, X, n_clusters=3, linkage='ward', affinity='euclidean'):
        """
        Entrena un modelo de Clustering Jerárquico
        
        Args:
            X (pd.DataFrame o np.array): Features normalizadas
            n_clusters (int): Número de clústeres
            linkage (str): Criterio de linkage ('ward', 'complete', 'average', 'single')
            affinity (str): Métrica de distancia ('euclidean', 'manhattan', 'cosine')
            
        Returns:
            AgglomerativeClustering: Modelo entrenado
        """
        print(f"Entrenando Clustering Jerárquico ({linkage}, {affinity}) con {n_clusters} clústeres...")
        
        if linkage == 'ward':
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage
            )
        else:
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric=affinity
            )
        
        labels = hierarchical.fit_predict(X)
        
        # Guardar modelo y predicciones
        self.models['Hierarchical'] = hierarchical
        self.predictions['Hierarchical'] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'linkage': linkage,
            'affinity': affinity
        }
        
        print(f"  Clustering Jerárquico entrenado")
        
        return hierarchical
    
    def evaluate_model(self, model_name, X):
        """
        Evalúa un modelo de clustering usando métricas no supervisadas
        
        Args:
            model_name (str): Nombre del modelo ('K-Means', 'DBSCAN', 'Hierarchical')
            X (pd.DataFrame o np.array): Features normalizadas
            
        Returns:
            dict: Diccionario con métricas de evaluación
        """
        if model_name not in self.predictions:
            raise ValueError(f"Modelo {model_name} no ha sido entrenado")
        
        labels = self.predictions[model_name]['labels']
        
        # Filtrar outliers para métricas (DBSCAN puede tener -1)
        mask = labels != -1
        if mask.sum() < len(labels):
            # Hay outliers, evaluar solo con puntos no-outlier
            X_filtered = X[mask] if hasattr(X, 'iloc') else X[mask]
            labels_filtered = labels[mask]
        else:
            X_filtered = X
            labels_filtered = labels
        
        # Solo calcular métricas si hay al menos 2 clústeres y más de 1 muestra por clúster
        n_clusters = len(set(labels_filtered))
        if n_clusters < 2:
            print(f"  Advertencia: {model_name} tiene menos de 2 clústeres, no se pueden calcular métricas")
            return {
                'model_name': model_name,
                'silhouette_score': np.nan,
                'calinski_harabasz_score': np.nan,
                'davies_bouldin_score': np.nan,
                'n_clusters': n_clusters,
                'n_samples': len(labels_filtered)
            }
        
        # Calcular métricas
        try:
            silhouette = silhouette_score(X_filtered, labels_filtered)
        except:
            silhouette = np.nan
        
        try:
            calinski_harabasz = calinski_harabasz_score(X_filtered, labels_filtered)
        except:
            calinski_harabasz = np.nan
        
        try:
            davies_bouldin = davies_bouldin_score(X_filtered, labels_filtered)
        except:
            davies_bouldin = np.nan
        
        metrics = {
            'model_name': model_name,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'n_clusters': n_clusters,
            'n_samples': len(labels_filtered)
        }
        
        # Agregar métricas específicas del modelo
        if model_name == 'K-Means':
            metrics['inertia'] = self.predictions[model_name]['inertia']
        elif model_name == 'DBSCAN':
            metrics['n_noise'] = self.predictions[model_name]['n_noise']
            metrics['noise_percentage'] = (metrics['n_noise'] / len(labels)) * 100
        
        self.metrics[model_name] = metrics
        
        return metrics
    
    def find_optimal_k(self, X, k_range=range(2, 11), method='silhouette'):
        """
        Encuentra el número óptimo de clústeres para K-Means
        
        Args:
            X (pd.DataFrame o np.array): Features normalizadas
            k_range (range): Rango de valores de k a probar
            method (str): Método para determinar k óptimo ('silhouette', 'elbow')
            
        Returns:
            dict: Diccionario con resultados del análisis
        """
        print(f"Buscando número óptimo de clústeres (método: {method})...")
        
        results = {
            'k_values': [],
            'silhouette_scores': [],
            'inertias': [],
            'calinski_harabasz_scores': [],
            'davies_bouldin_scores': []
        }
        
        for k in k_range:
            print(f"  Probando k={k}...", end=' ')
            
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                random_state=self.random_state
            )
            labels = kmeans.fit_predict(X)
            
            # Calcular métricas
            try:
                silhouette = silhouette_score(X, labels)
            except:
                silhouette = np.nan
            
            try:
                calinski_harabasz = calinski_harabasz_score(X, labels)
            except:
                calinski_harabasz = np.nan
            
            try:
                davies_bouldin = davies_bouldin_score(X, labels)
            except:
                davies_bouldin = np.nan
            
            results['k_values'].append(k)
            results['silhouette_scores'].append(silhouette)
            results['inertias'].append(kmeans.inertia_)
            results['calinski_harabasz_scores'].append(calinski_harabasz)
            results['davies_bouldin_scores'].append(davies_bouldin)
            
            print(f"Silhouette: {silhouette:.4f}")
        
        # Determinar k óptimo
        if method == 'silhouette':
            valid_scores = [(i, s) for i, s in enumerate(results['silhouette_scores']) if not np.isnan(s)]
            if valid_scores:
                optimal_idx = max(valid_scores, key=lambda x: x[1])[0]
                optimal_k = results['k_values'][optimal_idx]
            else:
                optimal_k = results['k_values'][0]
        else:  # elbow method
            # Método del codo simplificado (primer punto donde la reducción de inertia es menor)
            optimal_k = results['k_values'][0]
        
        results['optimal_k'] = optimal_k
        results['optimal_silhouette'] = results['silhouette_scores'][optimal_idx] if 'optimal_idx' in locals() else np.nan
        
        print(f"\n  K óptimo encontrado: {optimal_k} (Silhouette: {results['optimal_silhouette']:.4f})")
        
        return results
    
    def get_cluster_characteristics(self, model_name, X, df_original):
        """
        Obtiene las características de cada clúster
        
        Args:
            model_name (str): Nombre del modelo
            X (pd.DataFrame): Features normalizadas
            df_original (pd.DataFrame): Dataset original con variables originales
            
        Returns:
            pd.DataFrame: Características promedio por clúster
        """
        if model_name not in self.predictions:
            raise ValueError(f"Modelo {model_name} no ha sido entrenado")
        
        labels = self.predictions[model_name]['labels']
        
        # Agregar labels al dataset original
        df_with_clusters = df_original.copy()
        df_with_clusters['cluster'] = labels
        
        # Filtrar outliers si existen
        df_with_clusters = df_with_clusters[df_with_clusters['cluster'] != -1]
        
        # Seleccionar columnas numéricas relevantes para caracterización
        numeric_cols = ['driver_age', 'number_of_vehicles', 'number_of_fatalities']
        numeric_cols = [col for col in numeric_cols if col in df_with_clusters.columns]
        
        # Agrupar por clúster y calcular estadísticas
        cluster_stats = df_with_clusters.groupby('cluster')[numeric_cols].agg(['mean', 'std', 'count'])
        
        # Agregar información categórica
        categorical_cols = ['road_conditions', 'weather_conditions', 'accident_severity']
        categorical_cols = [col for col in categorical_cols if col in df_with_clusters.columns]
        
        cluster_categorical = {}
        for col in categorical_cols:
            mode_by_cluster = df_with_clusters.groupby('cluster')[col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A')
            cluster_categorical[col] = mode_by_cluster
        
        # Combinar resultados
        characteristics = cluster_stats.copy()
        for col, values in cluster_categorical.items():
            characteristics[col] = values
        
        return characteristics


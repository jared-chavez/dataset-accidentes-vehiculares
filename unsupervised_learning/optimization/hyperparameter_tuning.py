#!/usr/bin/env python3
"""
Optimización de Hiperparámetros para Análisis No Supervisado
============================================================

Este módulo implementa la optimización de hiperparámetros para modelos de clustering.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuning:
    """
    Clase para optimizar hiperparámetros de modelos de clustering
    """
    
    def __init__(self, random_state=42):
        """
        Inicializa la clase de optimización
        
        Args:
            random_state (int): Semilla aleatoria para reproducibilidad
        """
        self.random_state = random_state
        self.optimization_results = {}
    
    def optimize_kmeans(self, X, k_range=range(2, 11), init_methods=['k-means++'], 
                       n_init_values=[10], max_iter_values=[300], metric='silhouette'):
        """
        Optimiza hiperparámetros de K-Means
        
        Args:
            X (pd.DataFrame o np.array): Features normalizadas
            k_range (range): Rango de valores de k a probar
            init_methods (list): Métodos de inicialización a probar
            n_init_values (list): Valores de n_init a probar
            max_iter_values (list): Valores de max_iter a probar
            metric (str): Métrica a optimizar ('silhouette', 'calinski_harabasz', 'davies_bouldin')
            
        Returns:
            dict: Mejores parámetros y resultados
        """
        print("Optimizando K-Means...")
        print(f"  Métrica objetivo: {metric}")
        
        best_score = -np.inf if metric != 'davies_bouldin' else np.inf
        best_params = None
        results = []
        
        for k in k_range:
            for init in init_methods:
                for n_init in n_init_values:
                    for max_iter in max_iter_values:
                        try:
                            kmeans = KMeans(
                                n_clusters=k,
                                init=init,
                                n_init=n_init,
                                max_iter=max_iter,
                                random_state=self.random_state
                            )
                            labels = kmeans.fit_predict(X)
                            
                            # Calcular métricas
                            if len(set(labels)) < 2:
                                continue
                            
                            silhouette = silhouette_score(X, labels)
                            calinski = calinski_harabasz_score(X, labels)
                            davies = davies_bouldin_score(X, labels)
                            
                            # Seleccionar métrica objetivo
                            if metric == 'silhouette':
                                score = silhouette
                            elif metric == 'calinski_harabasz':
                                score = calinski
                            else:  # davies_bouldin
                                score = -davies  # Negativo porque menor es mejor
                            
                            result = {
                                'k': k,
                                'init': init,
                                'n_init': n_init,
                                'max_iter': max_iter,
                                'silhouette_score': silhouette,
                                'calinski_harabasz_score': calinski,
                                'davies_bouldin_score': davies,
                                'inertia': kmeans.inertia_,
                                'score': score
                            }
                            results.append(result)
                            
                            # Actualizar mejor si es necesario
                            if metric == 'davies_bouldin':
                                if score > best_score:
                                    best_score = score
                                    best_params = result
                            else:
                                if score > best_score:
                                    best_score = score
                                    best_params = result
                                    
                        except Exception as e:
                            print(f"  Error con k={k}, init={init}: {e}")
                            continue
        
        self.optimization_results['K-Means'] = {
            'best_params': best_params,
            'all_results': pd.DataFrame(results)
        }
        
        print(f"  Mejores parámetros encontrados:")
        print(f"    k: {best_params['k']}")
        print(f"    init: {best_params['init']}")
        print(f"    Silhouette: {best_params['silhouette_score']:.4f}")
        
        return best_params
    
    def optimize_dbscan(self, X, eps_range=None, min_samples_range=[3, 5, 10, 15], 
                       metric='silhouette'):
        """
        Optimiza hiperparámetros de DBSCAN
        
        Args:
            X (pd.DataFrame o np.array): Features normalizadas
            eps_range (list): Rango de valores de eps a probar (None para auto-estimar)
            min_samples_range (list): Valores de min_samples a probar
            metric (str): Métrica a optimizar
            
        Returns:
            dict: Mejores parámetros y resultados
        """
        print("Optimizando DBSCAN...")
        print(f"  Métrica objetivo: {metric}")
        
        # Si no se especifica eps_range, estimar usando k-distance graph
        if eps_range is None:
            print("  Estimando rango de eps usando k-distance graph...")
            from sklearn.neighbors import NearestNeighbors
            
            # Calcular distancias a k-ésimo vecino más cercano
            k = max(min_samples_range)
            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors_fit = neighbors.fit(X)
            distances, indices = neighbors_fit.kneighbors(X)
            distances = np.sort(distances, axis=0)
            distances = distances[:, k-1]
            
            # Usar percentiles para determinar rango
            eps_min = np.percentile(distances, 10)
            eps_max = np.percentile(distances, 90)
            eps_range = np.linspace(eps_min, eps_max, 10).tolist()
            print(f"  Rango de eps estimado: {eps_min:.3f} - {eps_max:.3f}")
        
        best_score = -np.inf if metric != 'davies_bouldin' else np.inf
        best_params = None
        results = []
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    # Filtrar outliers para métricas
                    mask = labels != -1
                    if mask.sum() < 2 or n_clusters < 2:
                        continue
                    
                    X_filtered = X[mask] if hasattr(X, 'iloc') else X[mask]
                    labels_filtered = labels[mask]
                    
                    silhouette = silhouette_score(X_filtered, labels_filtered)
                    calinski = calinski_harabasz_score(X_filtered, labels_filtered)
                    davies = davies_bouldin_score(X_filtered, labels_filtered)
                    
                    # Seleccionar métrica objetivo
                    if metric == 'silhouette':
                        score = silhouette
                    elif metric == 'calinski_harabasz':
                        score = calinski
                    else:  # davies_bouldin
                        score = -davies
                    
                    result = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_percentage': (n_noise / len(labels)) * 100,
                        'silhouette_score': silhouette,
                        'calinski_harabasz_score': calinski,
                        'davies_bouldin_score': davies,
                        'score': score
                    }
                    results.append(result)
                    
                    # Actualizar mejor si es necesario
                    if metric == 'davies_bouldin':
                        if score > best_score:
                            best_score = score
                            best_params = result
                    else:
                        if score > best_score:
                            best_score = score
                            best_params = result
                            
                except Exception as e:
                    print(f"  Error con eps={eps:.3f}, min_samples={min_samples}: {e}")
                    continue
        
        self.optimization_results['DBSCAN'] = {
            'best_params': best_params,
            'all_results': pd.DataFrame(results)
        }
        
        if best_params:
            print(f"  Mejores parámetros encontrados:")
            print(f"    eps: {best_params['eps']:.3f}")
            print(f"    min_samples: {best_params['min_samples']}")
            print(f"    Clústeres: {best_params['n_clusters']}")
            print(f"    Outliers: {best_params['n_noise']} ({best_params['noise_percentage']:.1f}%)")
            print(f"    Silhouette: {best_params['silhouette_score']:.4f}")
        
        return best_params
    
    def optimize_hierarchical(self, X, n_clusters_range=range(2, 11), 
                            linkage_methods=['ward', 'complete', 'average'],
                            affinity_methods=['euclidean'], metric='silhouette'):
        """
        Optimiza hiperparámetros de Clustering Jerárquico
        
        Args:
            X (pd.DataFrame o np.array): Features normalizadas
            n_clusters_range (range): Rango de valores de n_clusters
            linkage_methods (list): Métodos de linkage a probar
            affinity_methods (list): Métodos de affinity a probar
            metric (str): Métrica a optimizar
            
        Returns:
            dict: Mejores parámetros y resultados
        """
        print("Optimizando Clustering Jerárquico...")
        print(f"  Métrica objetivo: {metric}")
        
        best_score = -np.inf if metric != 'davies_bouldin' else np.inf
        best_params = None
        results = []
        
        for n_clusters in n_clusters_range:
            for linkage in linkage_methods:
                for affinity in affinity_methods:
                    # Ward solo funciona con euclidean
                    if linkage == 'ward' and affinity != 'euclidean':
                        continue
                    
                    try:
                        hierarchical = AgglomerativeClustering(
                            n_clusters=n_clusters,
                            linkage=linkage,
                            affinity=affinity
                        )
                        labels = hierarchical.fit_predict(X)
                        
                        if len(set(labels)) < 2:
                            continue
                        
                        silhouette = silhouette_score(X, labels)
                        calinski = calinski_harabasz_score(X, labels)
                        davies = davies_bouldin_score(X, labels)
                        
                        # Seleccionar métrica objetivo
                        if metric == 'silhouette':
                            score = silhouette
                        elif metric == 'calinski_harabasz':
                            score = calinski
                        else:  # davies_bouldin
                            score = -davies
                        
                        result = {
                            'n_clusters': n_clusters,
                            'linkage': linkage,
                            'affinity': affinity,
                            'silhouette_score': silhouette,
                            'calinski_harabasz_score': calinski,
                            'davies_bouldin_score': davies,
                            'score': score
                        }
                        results.append(result)
                        
                        # Actualizar mejor si es necesario
                        if metric == 'davies_bouldin':
                            if score > best_score:
                                best_score = score
                                best_params = result
                        else:
                            if score > best_score:
                                best_score = score
                                best_params = result
                                
                    except Exception as e:
                        print(f"  Error con n_clusters={n_clusters}, linkage={linkage}: {e}")
                        continue
        
        self.optimization_results['Hierarchical'] = {
            'best_params': best_params,
            'all_results': pd.DataFrame(results)
        }
        
        if best_params:
            print(f"  Mejores parámetros encontrados:")
            print(f"    n_clusters: {best_params['n_clusters']}")
            print(f"    linkage: {best_params['linkage']}")
            print(f"    affinity: {best_params['affinity']}")
            print(f"    Silhouette: {best_params['silhouette_score']:.4f}")
        
        return best_params


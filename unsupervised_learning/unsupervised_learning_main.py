#!/usr/bin/env python3
"""
Script Principal - Análisis No Supervisado de Accidentes Vehiculares
======================================================================

Este script ejecuta todo el proceso de análisis no supervisado:
1. Preparación de datos
2. Reducción de dimensionalidad (PCA, t-SNE)
3. Clustering (K-Means, DBSCAN, Jerárquico)
4. Optimización de hiperparámetros
5. Evaluación de modelos
6. Generación de resultados y visualizaciones

Ejecutar: python unsupervised_learning_main.py
"""

import sys
import os
from pathlib import Path

# Detectar si se ejecuta desde la raíz o desde unsupervised_learning
script_path = Path(__file__).resolve()
if script_path.parent.name == 'unsupervised_learning':
    # Se ejecuta desde dentro de unsupervised_learning
    BASE_DIR = script_path.parent.parent  # Ir a la raíz del proyecto
    unsupervised_learning_dir = script_path.parent
else:
    # Se ejecuta desde la raíz del proyecto
    BASE_DIR = Path.cwd()
    unsupervised_learning_dir = BASE_DIR / 'unsupervised_learning'

unsupervised_learning_dir_str = str(unsupervised_learning_dir)
if unsupervised_learning_dir_str not in sys.path:
    sys.path.insert(0, unsupervised_learning_dir_str)
sys.path[0] = unsupervised_learning_dir_str

import pandas as pd
import numpy as np

if __package__:
    # Se ejecuta como módulo (python -m), usar importaciones absolutas desde el paquete
    from unsupervised_learning.preprocessing.data_preparation import prepare_data
    from unsupervised_learning.models.clustering_models import ClusteringModels
    from unsupervised_learning.models.dimensionality_reduction import DimensionalityReduction
    from unsupervised_learning.evaluation.metrics_calculation import (
        calculate_clustering_metrics,
        create_metrics_comparison_table,
        get_best_model
    )
    from unsupervised_learning.evaluation.visualizations import (
        plot_clusters_2d,
        plot_clusters_3d,
        plot_dendrogram,
        plot_cluster_heatmap,
        plot_pca_variance_explained,
        plot_pca_components,
        plot_tsne_visualization,
        plot_elbow_method,
        plot_silhouette_scores,
        plot_metrics_comparison,
        plot_cluster_sizes,
        plot_cluster_characteristics_bars
    )
    from unsupervised_learning.optimization.hyperparameter_tuning import HyperparameterTuning
else:
    # Se ejecuta directamente, usar importaciones relativas
    from preprocessing.data_preparation import prepare_data
    from models.clustering_models import ClusteringModels
    from models.dimensionality_reduction import DimensionalityReduction
    from evaluation.metrics_calculation import (
        calculate_clustering_metrics,
        create_metrics_comparison_table,
        get_best_model
    )
    from evaluation.visualizations import (
        plot_clusters_2d,
        plot_clusters_3d,
        plot_dendrogram,
        plot_cluster_heatmap,
        plot_pca_variance_explained,
        plot_pca_components,
        plot_tsne_visualization,
        plot_elbow_method,
        plot_silhouette_scores,
        plot_metrics_comparison,
        plot_cluster_sizes,
        plot_cluster_characteristics_bars
    )
    from optimization.hyperparameter_tuning import HyperparameterTuning

def main():
    """
    Función principal que ejecuta todo el proceso de análisis no supervisado
    """
    print("="*70)
    print("ANÁLISIS NO SUPERVISADO - ACCIDENTES VEHICULARES")
    print("="*70)
    
    # ========================================================================
    # FASE 1: PREPARACIÓN DE DATOS
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 1: PREPARACIÓN DE DATOS")
    print("="*70)
    
    # Preparar datos (sin incluir severidad como feature por defecto)
    X, feature_names, df_original, scaler = prepare_data(include_severity=False)
    print(f"\nDataset preparado: {X.shape[0]} registros, {X.shape[1]} features")
    print(f"Features normalizadas y listas para clustering")
    
    # ========================================================================
    # FASE 2: REDUCCIÓN DE DIMENSIONALIDAD
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 2: REDUCCIÓN DE DIMENSIONALIDAD")
    print("="*70)
    
    # Crear directorio de resultados
    results_path_dim = unsupervised_learning_dir / 'results' / 'dimensionality_reduction_results'
    results_path_dim.mkdir(parents=True, exist_ok=True)
    
    # Aplicar PCA
    print("\n" + "-"*70)
    print("APLICANDO PCA")
    print("-"*70)
    dim_reduction = DimensionalityReduction(random_state=42)
    pca_model = dim_reduction.fit_pca(X, variance_threshold=0.95)
    X_pca = dim_reduction.transformations['PCA']
    
    # Visualizar varianza explicada
    variance_data = dim_reduction.get_variance_explained_plot_data()
    plot_pca_variance_explained(
        variance_data,
        results_path_dim / 'pca_variance_explained.png'
    )
    
    # Visualizar componentes principales (2D)
    X_pca_2d = X_pca[:, :2] if X_pca.shape[1] >= 2 else X_pca
    plot_pca_components(
        X_pca_2d,
        labels=None,
        feature_names=feature_names,
        pca_model=pca_model,
        save_path=results_path_dim / 'pca_components.png',
        n_features_to_show=10
    )
    
    # Guardar información de componentes
    components_info = dim_reduction.get_pca_components_info(feature_names, top_n=10)
    components_df = pd.DataFrame({
        'PC': list(components_info.keys()),
        'Variance_Explained': [info['variance_explained'] for info in components_info.values()],
        'Top_Features': [', '.join([f['feature'] for f in info['top_features'][:5]]) 
                        for info in components_info.values()]
    })
    components_df.to_csv(results_path_dim / 'component_analysis.csv', index=False)
    print(f"\nAnálisis de componentes guardado en: {results_path_dim / 'component_analysis.csv'}")
    
    # Aplicar t-SNE (opcional, puede tardar)
    print("\n" + "-"*70)
    print("APLICANDO t-SNE (puede tardar varios minutos...)")
    print("-"*70)
    try:
        tsne_model = dim_reduction.fit_tsne(X, n_components=2, perplexity=30, max_iter=1000)
        X_tsne = dim_reduction.transformations['t-SNE']
        
        # Visualizar t-SNE
        plot_tsne_visualization(
            X_tsne,
            labels=None,
            save_path=results_path_dim / 'tsne_visualization.png',
            model_name="t-SNE"
        )
        print("t-SNE completado y visualizado")
    except Exception as e:
        print(f"Error aplicando t-SNE: {e}")
        print("Continuando sin t-SNE...")
        X_tsne = None
    
    # ========================================================================
    # FASE 3: DETERMINACIÓN DEL NÚMERO ÓPTIMO DE CLÚSTERES
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 3: DETERMINACIÓN DEL NÚMERO ÓPTIMO DE CLÚSTERES")
    print("="*70)
    
    clustering_models = ClusteringModels(random_state=42)
    
    # Método del codo y Silhouette Score para K-Means
    print("\n" + "-"*70)
    print("ANÁLISIS DE K ÓPTIMO PARA K-MEANS")
    print("-"*70)
    optimal_k_results = clustering_models.find_optimal_k(
        X, 
        k_range=range(2, 11), 
        method='silhouette'
    )
    
    optimal_k = optimal_k_results['optimal_k']
    print(f"\nK óptimo determinado: {optimal_k}")
    
    # Visualizar método del codo
    results_path_clust = unsupervised_learning_dir / 'results' / 'clustering_results'
    results_path_clust.mkdir(parents=True, exist_ok=True)
    
    plot_elbow_method(
        optimal_k_results['k_values'],
        optimal_k_results['inertias'],
        results_path_clust / 'elbow_method.png'
    )
    
    # Visualizar Silhouette Scores
    plot_silhouette_scores(
        optimal_k_results['k_values'],
        optimal_k_results['silhouette_scores'],
        results_path_clust / 'silhouette_scores.png'
    )
    
    # ========================================================================
    # FASE 4: CLUSTERING
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 4: APLICACIÓN DE ALGORITMOS DE CLUSTERING")
    print("="*70)
    
    # K-Means con k óptimo
    print("\n" + "-"*70)
    print("ENTRENANDO K-MEANS")
    print("-"*70)
    kmeans_model = clustering_models.fit_kmeans(X, n_clusters=optimal_k)
    kmeans_labels = clustering_models.predictions['K-Means']['labels']
    kmeans_centroids = clustering_models.predictions['K-Means']['centroids']
    
    # DBSCAN
    print("\n" + "-"*70)
    print("ENTRENANDO DBSCAN")
    print("-"*70)
    # Estimar eps usando k-distance graph
    from sklearn.neighbors import NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 4]
    eps_estimate = np.percentile(distances, 50)  # Mediana
    
    dbscan_model = clustering_models.fit_dbscan(X, eps=eps_estimate, min_samples=5)
    dbscan_labels = clustering_models.predictions['DBSCAN']['labels']
    
    # Clustering Jerárquico
    print("\n" + "-"*70)
    print("ENTRENANDO CLUSTERING JERÁRQUICO")
    print("-"*70)
    hierarchical_model = clustering_models.fit_hierarchical(
        X, 
        n_clusters=optimal_k, 
        linkage='ward'
    )
    hierarchical_labels = clustering_models.predictions['Hierarchical']['labels']
    
    # ========================================================================
    # FASE 5: EVALUACIÓN DE MODELOS
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 5: EVALUACIÓN DE MODELOS")
    print("="*70)
    
    # Evaluar cada modelo
    all_metrics = []
    
    for model_name in ['K-Means', 'DBSCAN', 'Hierarchical']:
        print(f"\nEvaluando {model_name}...")
        metrics = clustering_models.evaluate_model(model_name, X)
        all_metrics.append(metrics)
        
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"  Calinski-Harabasz: {metrics['calinski_harabasz_score']:.4f}")
        print(f"  Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")
        print(f"  Número de clústeres: {metrics['n_clusters']}")
        if 'n_noise' in metrics:
            print(f"  Outliers: {metrics['n_noise']} ({metrics.get('noise_percentage', 0):.1f}%)")
    
    # Crear tabla comparativa
    metrics_df = create_metrics_comparison_table(all_metrics)
    print("\n" + "-"*70)
    print("TABLA COMPARATIVA DE MÉTRICAS")
    print("-"*70)
    print(metrics_df.to_string(index=False))
    
    # Guardar tabla
    metrics_df.to_csv(results_path_clust / 'metrics_comparison.csv', index=False)
    print(f"\nMétricas guardadas en: {results_path_clust / 'metrics_comparison.csv'}")
    
    # Identificar mejor modelo
    best_model = get_best_model(all_metrics, metric='silhouette_score', higher_is_better=True)
    print(f"\nMejor modelo: {best_model['model_name']}")
    print(f"   Silhouette Score: {best_model['silhouette_score']:.4f}")
    
    # Visualizar comparación de métricas
    plot_metrics_comparison(metrics_df, results_path_clust / 'metrics_comparison.png')
    
    # ========================================================================
    # FASE 6: VISUALIZACIONES DE CLUSTERING
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 6: GENERACIÓN DE VISUALIZACIONES")
    print("="*70)
    
    # Usar PCA 2D para visualización
    X_vis = X_pca_2d
    
    # Visualizar K-Means
    print("\nGenerando visualizaciones de K-Means...")
    # Transformar centroides a espacio PCA
    centroids_pca = pca_model.transform(kmeans_centroids)[:, :2]
    plot_clusters_2d(
        X_vis, kmeans_labels, 'K-Means',
        results_path_clust / 'kmeans_clusters.png',
        title_suffix=f"k={optimal_k}",
        centroids=centroids_pca,
        show_centroids=True
    )
    plot_cluster_sizes(kmeans_labels, 'K-Means', 
                      results_path_clust / 'kmeans_cluster_sizes.png')
    
    # Visualizar DBSCAN
    print("Generando visualizaciones de DBSCAN...")
    plot_clusters_2d(
        X_vis, dbscan_labels, 'DBSCAN',
        results_path_clust / 'dbscan_clusters.png',
        title_suffix=f"eps={eps_estimate:.3f}"
    )
    plot_cluster_sizes(dbscan_labels, 'DBSCAN',
                      results_path_clust / 'dbscan_cluster_sizes.png')
    
    # Visualizar Clustering Jerárquico
    print("Generando visualizaciones de Clustering Jerárquico...")
    plot_clusters_2d(
        X_vis, hierarchical_labels, 'Clustering Jerárquico',
        results_path_clust / 'hierarchical_clusters.png',
        title_suffix=f"k={optimal_k}, linkage=ward"
    )
    plot_cluster_sizes(hierarchical_labels, 'Clustering Jerárquico',
                      results_path_clust / 'hierarchical_cluster_sizes.png')
    
    # Dendrograma (usar muestra si hay muchas muestras)
    print("Generando dendrograma...")
    plot_dendrogram(
        X, 'Clustering Jerárquico',
        results_path_clust / 'hierarchical_dendrogram.png',
        method='ward',
        max_display_levels=10
    )
    
    # Si t-SNE está disponible, visualizar con t-SNE
    if X_tsne is not None:
        print("Generando visualizaciones con t-SNE...")
        plot_clusters_2d(
            X_tsne, kmeans_labels, 'K-Means',
            results_path_clust / 'kmeans_clusters_tsne.png',
            title_suffix="t-SNE projection"
        )
        plot_clusters_2d(
            X_tsne, dbscan_labels, 'DBSCAN',
            results_path_clust / 'dbscan_clusters_tsne.png',
            title_suffix="t-SNE projection"
        )
    
    # ========================================================================
    # FASE 7: CARACTERIZACIÓN DE CLÚSTERES
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 7: CARACTERIZACIÓN DE CLÚSTERES")
    print("="*70)
    
    # Caracterizar clústeres del mejor modelo
    best_model_name = best_model['model_name']
    print(f"\nCaracterizando clústeres de {best_model_name}...")
    
    try:
        cluster_chars = clustering_models.get_cluster_characteristics(
            best_model_name, X, df_original
        )
        
        # Guardar características
        cluster_chars.to_csv(results_path_clust / 'cluster_characteristics.csv')
        print(f"Características guardadas en: {results_path_clust / 'cluster_characteristics.csv'}")
        
        # Visualizar heatmap de características
        plot_cluster_heatmap(
            cluster_chars, best_model_name,
            results_path_clust / 'cluster_heatmap.png'
        )
        
        # Visualizar características en barras
        plot_cluster_characteristics_bars(
            cluster_chars, best_model_name,
            results_path_clust / 'cluster_characteristics_bars.png',
            top_n_features=10
        )
        
        print("\nResumen de clústeres:")
        print(cluster_chars.to_string())
        
    except Exception as e:
        print(f"Error caracterizando clústeres: {e}")
    
    # ========================================================================
    # FASE 8: OPTIMIZACIÓN (OPCIONAL)
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 8: OPTIMIZACIÓN DE HIPERPARÁMETROS (OPCIONAL)")
    print("="*70)
    
    try:
        optimize = input("\n¿Desea optimizar hiperparámetros? (s/n, default=n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        optimize = 'n'
        print("\n  Usando valor por defecto: no optimizar")
    
    if optimize == 's':
        print("\nIniciando optimización...")
        tuner = HyperparameterTuning(random_state=42)
        
        # Optimizar K-Means
        print("\nOptimizando K-Means...")
        best_kmeans_params = tuner.optimize_kmeans(
            X, k_range=range(2, 11), metric='silhouette'
        )
        
        # Optimizar DBSCAN
        print("\nOptimizando DBSCAN...")
        best_dbscan_params = tuner.optimize_dbscan(
            X, min_samples_range=[3, 5, 10], metric='silhouette'
        )
        
        # Optimizar Clustering Jerárquico
        print("\nOptimizando Clustering Jerárquico...")
        best_hierarchical_params = tuner.optimize_hierarchical(
            X, n_clusters_range=range(2, 11), metric='silhouette'
        )
        
        # Guardar resultados de optimización
        optimization_results = {
            'K-Means': best_kmeans_params,
            'DBSCAN': best_dbscan_params,
            'Hierarchical': best_hierarchical_params
        }
        
        optimization_df = pd.DataFrame([
            {**params, 'model': model} 
            for model, params in optimization_results.items() 
            if params is not None
        ])
        
        if len(optimization_df) > 0:
            optimization_df.to_csv(results_path_clust / 'optimization_results.csv', index=False)
            print(f"\nResultados de optimización guardados en: {results_path_clust / 'optimization_results.csv'}")
    else:
        print("Optimización omitida")
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n" + "="*70)
    print("ANÁLISIS NO SUPERVISADO COMPLETADO")
    print("="*70)
    
    print("\nRESUMEN DE RESULTADOS:")
    print(f"\nMejor modelo: {best_model['model_name']}")
    print(f"  Silhouette Score: {best_model['silhouette_score']:.4f}")
    print(f"  Calinski-Harabasz: {best_model['calinski_harabasz_score']:.4f}")
    print(f"  Davies-Bouldin: {best_model['davies_bouldin_score']:.4f}")
    print(f"  Número de clústeres: {best_model['n_clusters']}")
    
    print(f"\nResultados guardados en:")
    print(f"  - Clustering: {results_path_clust}")
    print(f"  - Reducción de dimensionalidad: {results_path_dim}")
    
    print("\nPROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)

if __name__ == "__main__":
    main()


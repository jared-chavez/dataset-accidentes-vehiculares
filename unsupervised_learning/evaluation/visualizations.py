#!/usr/bin/env python3
"""
Visualizaciones para Análisis No Supervisado
============================================

Este módulo genera todas las visualizaciones necesarias para evaluar
y visualizar modelos de clustering y reducción de dimensionalidad.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def save_plot(fig, filepath, dpi=300):
    """
    Guarda una figura en archivo
    
    Args:
        fig: Figura de matplotlib
        filepath: Ruta donde guardar
        dpi: Resolución
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico guardado: {filepath}")

def plot_clusters_2d(X_transformed, labels, model_name, save_path, 
                     title_suffix="", centroids=None, show_centroids=True):
    """
    Crea scatter plot 2D de clústeres
    
    Args:
        X_transformed: Datos transformados en 2D (PCA o t-SNE)
        labels: Etiquetas de clústeres
        model_name: Nombre del modelo
        save_path: Ruta donde guardar
        title_suffix: Sufijo para el título
        centroids: Centroides en espacio transformado (opcional)
        show_centroids: Si True, muestra centroides
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Filtrar outliers si existen (labels == -1)
    mask = labels != -1
    if mask.sum() < len(labels):
        X_plot = X_transformed[mask]
        labels_plot = labels[mask]
        n_outliers = (labels == -1).sum()
    else:
        X_plot = X_transformed
        labels_plot = labels
        n_outliers = 0
    
    # Obtener colores únicos para cada clúster
    unique_labels = np.unique(labels_plot)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    # Plotear cada clúster
    for i, label in enumerate(unique_labels):
        mask_cluster = labels_plot == label
        ax.scatter(X_plot[mask_cluster, 0], X_plot[mask_cluster, 1],
                  c=[colors[i]], label=f'Clúster {label}', 
                  alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    # Plotear outliers si existen
    if n_outliers > 0:
        outliers_mask = labels == -1
        ax.scatter(X_transformed[outliers_mask, 0], 
                  X_transformed[outliers_mask, 1],
                  c='black', marker='x', s=100, 
                  label=f'Outliers ({n_outliers})', 
                  alpha=0.8, linewidths=2)
    
    # Plotear centroides si se proporcionan
    if centroids is not None and show_centroids:
        ax.scatter(centroids[:, 0], centroids[:, 1],
                  c='red', marker='*', s=500, 
                  label='Centroides', edgecolors='black', 
                  linewidths=2, zorder=10)
    
    ax.set_xlabel('Primera Dimensión', fontsize=12)
    ax.set_ylabel('Segunda Dimensión', fontsize=12)
    title = f'Clústeres - {model_name}'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_clusters_3d(X_transformed, labels, model_name, save_path, 
                     title_suffix="", centroids=None):
    """
    Crea scatter plot 3D de clústeres
    
    Args:
        X_transformed: Datos transformados en 3D (PCA o t-SNE)
        labels: Etiquetas de clústeres
        model_name: Nombre del modelo
        save_path: Ruta donde guardar
        title_suffix: Sufijo para el título
        centroids: Centroides en espacio transformado (opcional)
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Filtrar outliers
    mask = labels != -1
    if mask.sum() < len(labels):
        X_plot = X_transformed[mask]
        labels_plot = labels[mask]
        n_outliers = (labels == -1).sum()
    else:
        X_plot = X_transformed
        labels_plot = labels
        n_outliers = 0
    
    # Obtener colores únicos
    unique_labels = np.unique(labels_plot)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    # Plotear cada clúster
    for i, label in enumerate(unique_labels):
        mask_cluster = labels_plot == label
        ax.scatter(X_plot[mask_cluster, 0], 
                  X_plot[mask_cluster, 1],
                  X_plot[mask_cluster, 2],
                  c=[colors[i]], label=f'Clúster {label}',
                  alpha=0.6, s=30, edgecolors='black', linewidths=0.3)
    
    # Plotear outliers
    if n_outliers > 0:
        outliers_mask = labels == -1
        ax.scatter(X_transformed[outliers_mask, 0],
                  X_transformed[outliers_mask, 1],
                  X_transformed[outliers_mask, 2],
                  c='black', marker='x', s=100,
                  label=f'Outliers ({n_outliers})',
                  alpha=0.8, linewidths=2)
    
    # Plotear centroides
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                  c='red', marker='*', s=500,
                  label='Centroides', edgecolors='black',
                  linewidths=2)
    
    ax.set_xlabel('Primera Dimensión', fontsize=11)
    ax.set_ylabel('Segunda Dimensión', fontsize=11)
    ax.set_zlabel('Tercera Dimensión', fontsize=11)
    title = f'Clústeres 3D - {model_name}'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_dendrogram(X, model_name, save_path, method='ward', 
                   max_display_levels=10, truncate_mode='level'):
    """
    Crea dendrograma para clustering jerárquico
    
    Args:
        X: Datos normalizados
        model_name: Nombre del modelo
        save_path: Ruta donde guardar
        method: Método de linkage ('ward', 'complete', 'average', 'single')
        max_display_levels: Número máximo de niveles a mostrar
        truncate_mode: Modo de truncamiento ('level', 'lastp', None)
    """
    print(f"Calculando dendrograma (método: {method})...")
    
    # Calcular linkage matrix
    # Si hay muchas muestras, usar una muestra representativa
    if len(X) > 1000:
        print(f"  Dataset grande ({len(X)} muestras), usando muestra de 1000 para dendrograma...")
        from sklearn.utils import resample
        X_sample = resample(X, n_samples=1000, random_state=42)
    else:
        X_sample = X
    
    linkage_matrix = linkage(X_sample, method=method)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    if truncate_mode == 'level':
        dendrogram(
            linkage_matrix,
            truncate_mode='level',
            p=max_display_levels,
            ax=ax,
            leaf_rotation=90,
            leaf_font_size=8
        )
    elif truncate_mode == 'lastp':
        dendrogram(
            linkage_matrix,
            truncate_mode='lastp',
            p=max_display_levels,
            ax=ax,
            leaf_rotation=90,
            leaf_font_size=8
        )
    else:
        dendrogram(
            linkage_matrix,
            ax=ax,
            leaf_rotation=90,
            leaf_font_size=8
        )
    
    ax.set_xlabel('Muestra o (Clúster)', fontsize=12)
    ax.set_ylabel('Distancia', fontsize=12)
    ax.set_title(f'Dendrograma - {model_name} (Linkage: {method})', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_cluster_heatmap(cluster_characteristics, model_name, save_path):
    """
    Crea heatmap de características promedio por clúster
    
    Args:
        cluster_characteristics: DataFrame con características por clúster
        model_name: Nombre del modelo
        save_path: Ruta donde guardar
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Preparar datos para heatmap
    # Si el DataFrame tiene MultiIndex, aplanarlo
    if isinstance(cluster_characteristics.columns, pd.MultiIndex):
        # Seleccionar solo las medias
        heatmap_data = cluster_characteristics.xs('mean', axis=1, level=1)
    else:
        heatmap_data = cluster_characteristics
    
    # Normalizar datos para mejor visualización
    heatmap_data_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-10)
    
    # Crear heatmap
    sns.heatmap(
        heatmap_data_normalized.T,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Valor Normalizado'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_xlabel('Clúster', fontsize=12)
    ax.set_ylabel('Característica', fontsize=12)
    ax.set_title(f'Características Promedio por Clúster - {model_name}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_pca_variance_explained(variance_data, save_path, variance_threshold=0.95):
    """
    Crea gráfico de varianza explicada por componentes principales
    
    Args:
        variance_data: Dict con 'individual' y 'cumulative' variance arrays
        save_path: Ruta donde guardar
        variance_threshold: Umbral de varianza (para línea de referencia)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    individual = variance_data['individual']
    cumulative = variance_data['cumulative']
    n_components = len(individual)
    
    # Gráfico de varianza individual
    axes[0].bar(range(1, min(21, n_components + 1)), 
               individual[:20], 
               color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Componente Principal', fontsize=11)
    axes[0].set_ylabel('Varianza Explicada', fontsize=11)
    axes[0].set_title('Varianza Explicada por Componente', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    if n_components > 20:
        axes[0].text(0.98, 0.98, f'(Mostrando primeros 20 de {n_components})',
                    transform=axes[0].transAxes, ha='right', va='top',
                    fontsize=9, style='italic')
    
    # Gráfico de varianza acumulada
    axes[1].plot(range(1, n_components + 1), cumulative, 
                marker='o', color='steelblue', linewidth=2, markersize=6)
    axes[1].axhline(y=variance_threshold, color='r', linestyle='--', 
                   linewidth=2, label=f'{variance_threshold*100:.0f}% Varianza')
    
    # Encontrar número de componentes para umbral
    n_components_threshold = np.argmax(cumulative >= variance_threshold) + 1
    axes[1].axvline(x=n_components_threshold, color='r', 
                   linestyle='--', linewidth=1, alpha=0.5)
    axes[1].text(n_components_threshold, variance_threshold,
                f'  {n_components_threshold} componentes',
                ha='left', va='bottom', fontsize=10, color='red')
    
    axes[1].set_xlabel('Número de Componentes', fontsize=11)
    axes[1].set_ylabel('Varianza Acumulada', fontsize=11)
    axes[1].set_title('Varianza Acumulada', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    
    plt.suptitle('Análisis de Varianza Explicada - PCA', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_pca_components(X_pca, labels=None, feature_names=None, 
                        pca_model=None, save_path=None, 
                        n_features_to_show=10):
    """
    Crea scatter plot de componentes principales con biplot opcional
    
    Args:
        X_pca: Datos transformados por PCA (2D)
        labels: Etiquetas para colorear (opcional)
        feature_names: Nombres de features originales (para biplot)
        pca_model: Modelo PCA entrenado (para biplot)
        save_path: Ruta donde guardar
        n_features_to_show: Número de features a mostrar en biplot
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Scatter plot
    if labels is not None:
        unique_labels = np.unique(labels[labels != -1])
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[colors[i]], label=f'Clúster {label}',
                      alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
        
        # Outliers
        if -1 in labels:
            outliers_mask = labels == -1
            ax.scatter(X_pca[outliers_mask, 0], X_pca[outliers_mask, 1],
                      c='black', marker='x', s=100,
                      label='Outliers', alpha=0.8, linewidths=2)
        ax.legend()
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50)
    
    # Biplot (vectores de features)
    if pca_model is not None and feature_names is not None:
        components = pca_model.components_[:2]  # Primeros 2 componentes
        variance_explained = pca_model.explained_variance_ratio_[:2]
        
        # Escalar componentes por varianza explicada
        scale = np.sqrt(variance_explained) * 3
        
        # Seleccionar features más importantes
        feature_importance = np.abs(components).sum(axis=0)
        top_indices = np.argsort(feature_importance)[-n_features_to_show:]
        
        for idx in top_indices:
            ax.arrow(0, 0, components[0, idx] * scale[0], 
                   components[1, idx] * scale[1],
                   head_width=0.05, head_length=0.05, 
                   fc='red', ec='red', alpha=0.7, linewidth=1.5)
            ax.text(components[0, idx] * scale[0] * 1.1,
                   components[1, idx] * scale[1] * 1.1,
                   feature_names[idx], fontsize=9, color='red',
                   ha='center', va='center')
    
    ax.set_xlabel(f'Primer Componente Principal (PC1)', fontsize=12)
    ax.set_ylabel(f'Segundo Componente Principal (PC2)', fontsize=12)
    title = 'Componentes Principales'
    if labels is not None:
        title += ' - Clústeres'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    if save_path:
        save_plot(fig, save_path)
    else:
        plt.show()
        plt.close(fig)

def plot_elbow_method(k_values, inertias, save_path, title_suffix=""):
    """
    Crea gráfico del método del codo para K-Means
    
    Args:
        k_values: Lista de valores de k
        inertias: Lista de valores de inertia
        save_path: Ruta donde guardar
        title_suffix: Sufijo para el título
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, inertias, marker='o', linewidth=2, 
           markersize=8, color='steelblue')
    ax.set_xlabel('Número de Clústeres (k)', fontsize=12)
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    title = 'Método del Codo - K-Means'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks(k_values)
    
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_silhouette_scores(k_values, silhouette_scores, save_path, title_suffix=""):
    """
    Crea gráfico de Silhouette Scores por número de clústeres
    
    Args:
        k_values: Lista de valores de k
        silhouette_scores: Lista de Silhouette Scores
        save_path: Ruta donde guardar
        title_suffix: Sufijo para el título
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, silhouette_scores, marker='o', linewidth=2,
           markersize=8, color='steelblue')
    
    # Marcar el mejor k
    best_idx = np.argmax(silhouette_scores)
    best_k = k_values[best_idx]
    best_score = silhouette_scores[best_idx]
    
    ax.scatter([best_k], [best_score], color='red', s=200,
              marker='*', zorder=10, label=f'Mejor k={best_k}')
    ax.axvline(x=best_k, color='red', linestyle='--', 
              linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Número de Clústeres (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    title = 'Silhouette Score por Número de Clústeres'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(k_values)
    
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_metrics_comparison(metrics_df, save_path):
    """
    Crea gráfico comparativo de métricas de clustering
    
    Args:
        metrics_df: DataFrame con métricas por modelo
        save_path: Ruta donde guardar
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics_to_plot = [
        ('Silhouette Score', 'silhouette_score', 'higher'),
        ('Calinski-Harabasz Index', 'calinski_harabasz_score', 'higher'),
        ('Davies-Bouldin Index', 'davies_bouldin_score', 'lower'),
        ('Número de Clústeres', 'n_clusters', None)
    ]
    
    for idx, (title, col, better) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        if col not in metrics_df.columns:
            print(f"Columna {col} no encontrada, saltando...")
            continue
        
        values = pd.to_numeric(metrics_df[col], errors='coerce')
        model_names = metrics_df.get('model_name', metrics_df.index)
        
        bars = ax.bar(range(len(model_names)), values, 
                     color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} por Modelo', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Agregar valores en las barras
        for i, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2., val,
                       f'{val:.3f}',
                       ha='center', va='bottom' if better == 'higher' else 'top',
                       fontsize=9)
        
        # Marcar el mejor si aplica
        if better == 'higher' and not values.isna().all():
            best_idx = values.idxmax()
            bars[best_idx].set_color('green')
        elif better == 'lower' and not values.isna().all():
            best_idx = values.idxmin()
            bars[best_idx].set_color('green')
    
    plt.suptitle('Comparación de Métricas de Clustering', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_cluster_sizes(labels, model_name, save_path):
    """
    Crea gráfico de barras con tamaños de clústeres
    
    Args:
        labels: Etiquetas de clústeres
        model_name: Nombre del modelo
        save_path: Ruta donde guardar
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Contar tamaños de clústeres
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Separar clústeres y outliers
    cluster_mask = unique_labels != -1
    cluster_labels = unique_labels[cluster_mask]
    cluster_counts = counts[cluster_mask]
    
    if -1 in unique_labels:
        outlier_idx = np.where(unique_labels == -1)[0][0]
        outlier_count = counts[outlier_idx]
    else:
        outlier_count = 0
    
    # Gráfico de barras
    bars = ax.bar(range(len(cluster_labels)), cluster_counts,
                 color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(cluster_labels)))
    ax.set_xticklabels([f'Clúster {int(l)}' for l in cluster_labels])
    ax.set_ylabel('Número de Muestras', fontsize=12)
    ax.set_xlabel('Clúster', fontsize=12)
    ax.set_title(f'Tamaño de Clústeres - {model_name}', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Agregar valores en las barras
    for bar, count in zip(bars, cluster_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(count)}',
               ha='center', va='bottom', fontsize=10)
    
    # Agregar información de outliers si existen
    if outlier_count > 0:
        ax.text(0.98, 0.98, f'Outliers: {int(outlier_count)}',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=11, bbox=dict(boxstyle='round', 
                                     facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_cluster_characteristics_bars(cluster_characteristics, model_name, 
                                     save_path, top_n_features=10):
    """
    Crea gráficos de barras con características promedio por clúster
    
    Args:
        cluster_characteristics: DataFrame con características por clúster
        model_name: Nombre del modelo
        save_path: Ruta donde guardar
        top_n_features: Número de features a mostrar
    """
    # Preparar datos
    if isinstance(cluster_characteristics.columns, pd.MultiIndex):
        data = cluster_characteristics.xs('mean', axis=1, level=1)
    else:
        data = cluster_characteristics
    
    # Seleccionar solo columnas numéricas
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data = data[numeric_cols]
    
    # Seleccionar top N features por varianza
    if len(data.columns) > top_n_features:
        variances = data.var(axis=0)
        top_features = variances.nlargest(top_n_features).index
        data = data[top_features]
    
    n_features = len(data.columns)
    n_clusters = len(data)
    
    fig, axes = plt.subplots(1, n_features, figsize=(4*n_features, 6))
    
    if n_features == 1:
        axes = [axes]
    
    for idx, (feature, ax) in enumerate(zip(data.columns, axes)):
        bars = ax.bar(range(n_clusters), data[feature].values,
                     color=plt.cm.Set3(np.linspace(0, 1, n_clusters)),
                     alpha=0.7, edgecolor='black', linewidth=1)
        
        ax.set_xticks(range(n_clusters))
        ax.set_xticklabels([f'C{int(i)}' for i in data.index])
        ax.set_ylabel('Valor Promedio', fontsize=10)
        ax.set_title(feature, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Agregar valores
        for bar, val in zip(bars, data[feature].values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'Características por Clúster - {model_name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_tsne_visualization(X_tsne, labels=None, save_path=None, 
                           model_name="t-SNE", title_suffix=""):
    """
    Crea visualización t-SNE
    
    Args:
        X_tsne: Datos transformados por t-SNE (2D)
        labels: Etiquetas para colorear (opcional)
        save_path: Ruta donde guardar
        model_name: Nombre del modelo
        title_suffix: Sufijo para el título
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if labels is not None:
        unique_labels = np.unique(labels[labels != -1])
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                      c=[colors[i]], label=f'Clúster {label}',
                      alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
        
        if -1 in labels:
            outliers_mask = labels == -1
            ax.scatter(X_tsne[outliers_mask, 0], X_tsne[outliers_mask, 1],
                      c='black', marker='x', s=100,
                      label='Outliers', alpha=0.8, linewidths=2)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=50)
    
    ax.set_xlabel('Primera Dimensión t-SNE', fontsize=12)
    ax.set_ylabel('Segunda Dimensión t-SNE', fontsize=12)
    title = f'Visualización t-SNE - {model_name}'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        save_plot(fig, save_path)
    else:
        plt.show()
        plt.close(fig)


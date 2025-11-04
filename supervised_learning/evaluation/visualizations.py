#!/usr/bin/env python3
"""
Visualizaciones para Evaluación de Modelos
===========================================

Este módulo genera todas las visualizaciones necesarias para evaluar
y comparar modelos de clasificación y regresión.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
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

def plot_confusion_matrix(y_true, y_pred, labels, model_name, save_path):
    """
    Crea y guarda matriz de confusión
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        labels: Etiquetas de clases
        model_name: Nombre del modelo
        save_path: Ruta donde guardar
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalizar matriz para porcentajes
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Crear heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={'label': 'Proporción'}
    )
    
    ax.set_xlabel('Predicción', fontsize=12)
    ax.set_ylabel('Valor Real', fontsize=12)
    ax.set_title(f'Matriz de Confusión - {model_name}', fontsize=14, fontweight='bold')
    
    # Agregar valores absolutos
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j+0.5, i+0.5, f'\n({cm[i, j]})',
                         ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_feature_importance(importance_df, top_n=15, model_name="Model", save_path=None):
    """
    Crea gráfico de importancia de features
    
    Args:
        importance_df: DataFrame con columnas 'feature' e 'importance'
        top_n: Número de features top a mostrar
        model_name: Nombre del modelo
        save_path: Ruta donde guardar (opcional)
    """
    if importance_df is None or len(importance_df) == 0:
        print("No hay datos de importancia disponibles")
        return
    
    # Obtener top N features
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Crear gráfico de barras horizontal
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['importance'].values, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importancia', fontsize=12)
    ax.set_title(f'Top {top_n} Features más Importantes - {model_name}', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    else:
        plt.show()
        plt.close(fig)

def plot_classification_metrics_comparison(metrics_df, save_path):
    """
    Crea gráfico comparativo de métricas de clasificación
    
    Args:
        metrics_df: DataFrame con métricas por modelo
        save_path: Ruta donde guardar
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics_to_plot = [
        ('Accuracy', 'Accuracy'),
        ('Precision (Macro)', 'Precision (Macro)'),
        ('Recall (Macro)', 'Recall (Macro)'),
        ('F1-Score (Macro)', 'F1-Score (Macro)')
    ]
    
    for idx, (title, col) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        # Verificar que la columna existe
        if col not in metrics_df.columns:
            print(f"Columna {col} no encontrada, saltando...")
            continue
        
        # Convertir a numérico si es string
        if metrics_df[col].dtype == 'object':
            values = metrics_df[col].str.replace('%', '').astype(float)
        else:
            values = metrics_df[col]
        
        bars = ax.bar(metrics_df['Modelo'], values, color='steelblue', alpha=0.7)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} por Modelo', fontsize=12, fontweight='bold')
        ax.set_xticklabels(metrics_df['Modelo'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Agregar valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Comparación de Métricas de Clasificación', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_regression_predictions_vs_real(y_true, y_pred, model_name, save_path):
    """
    Crea gráfico de predicciones vs valores reales
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        model_name: Nombre del modelo
        save_path: Ruta donde guardar
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=50)
    
    # Línea perfecta (y=x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 
           'r--', lw=2, label='Predicción Perfecta')
    
    # Calcular R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    ax.set_xlabel('Valores Reales', fontsize=12)
    ax.set_ylabel('Predicciones', fontsize=12)
    ax.set_title(f'Predicciones vs Valores Reales - {model_name}\nR² = {r2:.4f}', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_residuals(y_true, y_pred, model_name, save_path):
    """
    Crea gráfico de residuales
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        model_name: Nombre del modelo
        save_path: Ruta donde guardar
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de residuales vs predicciones
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicciones', fontsize=11)
    axes[0].set_ylabel('Residuales', fontsize=11)
    axes[0].set_title('Residuales vs Predicciones', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Histograma de residuales
    axes[1].hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuales', fontsize=11)
    axes[1].set_ylabel('Frecuencia', fontsize=11)
    axes[1].set_title('Distribución de Residuales', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Análisis de Residuales - {model_name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_regression_metrics_comparison(metrics_df, save_path):
    """
    Crea gráfico comparativo de métricas de regresión
    
    Args:
        metrics_df: DataFrame con métricas por modelo
        save_path: Ruta donde guardar
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics_to_plot = [
        ('MAE', 'MAE', 'lower'),
        ('RMSE', 'RMSE', 'lower'),
        ('R²', 'R²', 'higher'),
        ('MAPE (%)', 'MAPE (%)', 'lower')
    ]
    
    for idx, (title, col, better) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        # Verificar que la columna existe
        if col not in metrics_df.columns:
            print(f"Columna {col} no encontrada, saltando...")
            continue
        
        # Convertir a numérico si es string
        if metrics_df[col].dtype == 'object':
            values = pd.to_numeric(metrics_df[col].str.replace('%', ''), errors='coerce')
        else:
            values = metrics_df[col]
        
        bars = ax.bar(metrics_df['Modelo'], values, color='steelblue', alpha=0.7)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} por Modelo', fontsize=12, fontweight='bold')
        ax.set_xticklabels(metrics_df['Modelo'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Agregar valores en las barras
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Comparación de Métricas de Regresión', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_plot(fig, save_path)

def plot_roc_curves(y_true_dict, y_pred_proba_dict, labels, save_path):
    """
    Crea curvas ROC para clasificación multiclase (One vs Rest)
    
    Args:
        y_true_dict: Diccionario {model_name: y_true}
        y_pred_proba_dict: Diccionario {model_name: y_pred_proba}
        labels: Etiquetas de clases
        save_path: Ruta donde guardar
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    fig, axes = plt.subplots(1, len(labels), figsize=(6*len(labels), 6))
    
    if len(labels) == 1:
        axes = [axes]
    
    # Binarizar labels para One vs Rest
    y_true_bin = label_binarize(list(y_true_dict.values())[0], classes=labels)
    n_classes = len(labels)
    
    for class_idx, (label, ax) in enumerate(zip(labels, axes)):
        for model_name in y_true_dict.keys():
            y_true = y_true_dict[model_name]
            y_pred_proba = y_pred_proba_dict[model_name]
            
            if y_pred_proba is not None:
                y_true_bin = label_binarize(y_true, classes=labels)
                if n_classes == 2:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], 
                                           y_pred_proba[:, class_idx])
                else:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], 
                                           y_pred_proba[:, class_idx])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('Tasa de Falsos Positivos', fontsize=11)
        ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=11)
        ax.set_title(f'ROC Curve - Clase {label}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('Curvas ROC por Clase (One vs Rest)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, save_path)


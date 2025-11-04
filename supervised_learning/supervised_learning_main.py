#!/usr/bin/env python3
"""
Script Principal - Análisis Supervisado de Accidentes Vehiculares
==================================================================

Este script ejecuta todo el proceso de análisis supervisado:
1. Preparación de datos
2. Entrenamiento de modelos de clasificación y regresión
3. Evaluación de modelos
4. Generación de resultados y visualizaciones

Ejecutar: python supervised_learning_main.py
"""

import sys
from pathlib import Path

# Detectar si se ejecuta desde la raíz o desde supervised_learning
script_path = Path(__file__).resolve()
if script_path.parent.name == 'supervised_learning':
    # Se ejecuta desde dentro de supervised_learning
    BASE_DIR = script_path.parent.parent  # Ir a la raíz del proyecto
    supervised_learning_dir = script_path.parent
else:
    # Se ejecuta desde la raíz del proyecto
    BASE_DIR = Path.cwd()
    supervised_learning_dir = BASE_DIR / 'supervised_learning'

# Agregar ruta de módulos al path
sys.path.insert(0, str(supervised_learning_dir))

import pandas as pd
import numpy as np
from preprocessing.data_preparation import prepare_data
from preprocessing.feature_engineering import (
    prepare_features_for_classification,
    prepare_features_for_regression,
    split_data,
    normalize_features
)
from models.classification_models import ClassificationModels
from models.regression_models import RegressionModels
from evaluation.metrics_calculation import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    create_classification_metrics_table,
    create_regression_metrics_table,
    get_best_model_classification,
    get_best_model_regression
)
from evaluation.visualizations import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_classification_metrics_comparison,
    plot_regression_predictions_vs_real,
    plot_residuals,
    plot_regression_metrics_comparison
)

def main():
    """
    Función principal que ejecuta todo el proceso
    """
    print("="*70)
    print("ANALISIS SUPERVISADO - ACCIDENTES VEHICULARES")
    print("="*70)
    
    # ========================================================================
    # FASE 1: PREPARACIÓN DE DATOS
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 1: PREPARACIÓN DE DATOS")
    print("="*70)
    
    df = prepare_data()
    print(f"\nDataset preparado: {df.shape[0]} registros, {df.shape[1]} columnas")
    
    # ========================================================================
    # FASE 2: CLASIFICACIÓN - Predicción de Severidad
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 2: CLASIFICACIÓN - Predicción de Severidad del Accidente")
    print("="*70)
    
    # Preparar features para clasificación
    X_clf, y_clf, feature_names_clf = prepare_features_for_classification(df)
    
    # Dividir datos
    X_train_clf, X_val_clf, X_test_clf, y_train_clf, y_val_clf, y_test_clf = split_data(
        X_clf, y_clf, 
        test_size=0.15, 
        val_size=0.15, 
        random_state=42,
        stratify=y_clf  # Estratificar para mantener distribución de clases
    )
    
    # Normalizar features (algunos modelos lo requieren)
    X_train_clf_scaled, X_val_clf_scaled, X_test_clf_scaled, scaler_clf = normalize_features(
        X_train_clf, X_val_clf, X_test_clf
    )
    
    # Entrenar modelos de clasificación
    clf_models = ClassificationModels()
    # Usar optimize=False para velocidad, cambiar a True para mejores resultados
    clf_models.train_all_models(
        X_train_clf_scaled, y_train_clf,
        X_val_clf_scaled, y_val_clf,
        optimize=False  # Cambiar a True para optimización completa
    )
    
    # Evaluar modelos en test
    print("\n" + "-"*70)
    print("EVALUACIÓN DE MODELOS DE CLASIFICACIÓN EN TEST")
    print("-"*70)
    
    classification_metrics = []
    for model_name, model in clf_models.models.items():
        metrics = clf_models.evaluate_model(
            model, X_test_clf_scaled, y_test_clf, model_name
        )
        classification_metrics.append(metrics)
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
    
    # Crear tabla comparativa
    clf_metrics_df = create_classification_metrics_table(classification_metrics)
    print("\nTabla Comparativa de Clasificación:")
    print(clf_metrics_df.to_string(index=False))
    
    # Guardar tabla
    results_path = supervised_learning_dir / 'results' / 'classification_results'
    results_path.mkdir(parents=True, exist_ok=True)
    clf_metrics_df.to_csv(results_path / 'metrics_table.csv', index=False)
    print(f"\nMétricas guardadas en: {results_path / 'metrics_table.csv'}")
    
    # Identificar mejor modelo
    best_clf = get_best_model_classification(classification_metrics)
    print(f"\nMejor modelo de clasificación: {best_clf['model_name']}")
    print(f"   F1-Score (Macro): {best_clf['f1_macro']:.4f}")
    
    # Generar visualizaciones de clasificación
    print("\nGenerando visualizaciones de clasificación...")
    
    # Matriz de confusión para el mejor modelo
    best_clf_model = clf_models.models[best_clf['model_name']]
    y_pred_best = clf_models.predictions[best_clf['model_name']]['y_pred']
    labels = sorted(y_test_clf.unique())
    
    plot_confusion_matrix(
        y_test_clf, y_pred_best, labels,
        best_clf['model_name'],
        results_path / 'confusion_matrix.png'
    )
    
    # Gráfico comparativo de métricas
    plot_classification_metrics_comparison(
        clf_metrics_df,
        results_path / 'metrics_comparison.png'
    )
    
    # Importancia de features (si está disponible)
    if hasattr(best_clf_model, 'feature_importances_') or hasattr(best_clf_model, 'coef_'):
        importance_df = clf_models.get_feature_importance(
            best_clf_model, feature_names_clf, best_clf['model_name']
        )
        if importance_df is not None:
            plot_feature_importance(
                importance_df, top_n=15,
                model_name=best_clf['model_name'],
                save_path=results_path / 'feature_importance.png'
            )
            # Guardar importancia de features
            importance_df.to_csv(results_path / 'feature_importance.csv', index=False)
    
    # ========================================================================
    # FASE 3: REGRESIÓN - Predicción de Fatalidades
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 3: REGRESIÓN - Predicción de Número de Fatalidades")
    print("="*70)
    
    # Preparar features para regresión
    X_reg, y_reg, feature_names_reg = prepare_features_for_regression(df)
    
    # Dividir datos
    X_train_reg, X_val_reg, X_test_reg, y_train_reg, y_val_reg, y_test_reg = split_data(
        X_reg, y_reg,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    # Normalizar features
    X_train_reg_scaled, X_val_reg_scaled, X_test_reg_scaled, scaler_reg = normalize_features(
        X_train_reg, X_val_reg, X_test_reg
    )
    
    # Entrenar modelos de regresión
    reg_models = RegressionModels()
    # Usar optimize=False para velocidad
    reg_models.train_all_models(
        X_train_reg_scaled, y_train_reg,
        X_val_reg_scaled, y_val_reg,
        optimize=False  # Cambiar a True para optimización completa
    )
    
    # Evaluar modelos en test
    print("\n" + "-"*70)
    print("EVALUACIÓN DE MODELOS DE REGRESIÓN EN TEST")
    print("-"*70)
    
    regression_metrics = []
    for model_name, model in reg_models.models.items():
        metrics = reg_models.evaluate_model(
            model, X_test_reg_scaled, y_test_reg, model_name
        )
        regression_metrics.append(metrics)
        print(f"\n{model_name}:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    # Crear tabla comparativa
    reg_metrics_df = create_regression_metrics_table(regression_metrics)
    print("\nTabla Comparativa de Regresión:")
    print(reg_metrics_df.to_string(index=False))
    
    # Guardar tabla
    results_path_reg = supervised_learning_dir / 'results' / 'regression_results'
    results_path_reg.mkdir(parents=True, exist_ok=True)
    reg_metrics_df.to_csv(results_path_reg / 'metrics_table.csv', index=False)
    print(f"\nMétricas guardadas en: {results_path_reg / 'metrics_table.csv'}")
    
    # Identificar mejor modelo
    best_reg = get_best_model_regression(regression_metrics)
    print(f"\nMejor modelo de regresión: {best_reg['model_name']}")
    print(f"   MAE: {best_reg['mae']:.4f}")
    print(f"   R²: {best_reg['r2']:.4f}")
    
    # Generar visualizaciones de regresión
    print("\nGenerando visualizaciones de regresión...")
    
    # Predicciones vs reales para el mejor modelo
    best_reg_model = reg_models.models[best_reg['model_name']]
    y_pred_best_reg = reg_models.predictions[best_reg['model_name']]['y_pred']
    
    plot_regression_predictions_vs_real(
        y_test_reg, y_pred_best_reg,
        best_reg['model_name'],
        results_path_reg / 'predictions_vs_real.png'
    )
    
    # Análisis de residuales
    plot_residuals(
        y_test_reg, y_pred_best_reg,
        best_reg['model_name'],
        results_path_reg / 'residuals_plot.png'
    )
    
    # Gráfico comparativo de métricas
    plot_regression_metrics_comparison(
        reg_metrics_df,
        results_path_reg / 'metrics_comparison.png'
    )
    
    # Importancia de features
    if hasattr(best_reg_model, 'feature_importances_') or hasattr(best_reg_model, 'coef_'):
        importance_df_reg = reg_models.get_feature_importance(
            best_reg_model, feature_names_reg, best_reg['model_name']
        )
        if importance_df_reg is not None:
            plot_feature_importance(
                importance_df_reg, top_n=15,
                model_name=best_reg['model_name'],
                save_path=results_path_reg / 'feature_importance.png'
            )
            # Guardar importancia de features
            importance_df_reg.to_csv(results_path_reg / 'feature_importance.csv', index=False)
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n" + "="*70)
    print("ANÁLISIS SUPERVISADO COMPLETADO")
    print("="*70)
    
    print("\nRESUMEN DE RESULTADOS:")
    print(f"\nCLASIFICACIÓN (Severidad del Accidente):")
    print(f"  Mejor modelo: {best_clf['model_name']}")
    print(f"  Accuracy: {best_clf['accuracy']:.4f}")
    print(f"  F1-Score (Macro): {best_clf['f1_macro']:.4f}")
    
    print(f"\nREGRESIÓN (Número de Fatalidades):")
    print(f"  Mejor modelo: {best_reg['model_name']}")
    print(f"  MAE: {best_reg['mae']:.4f}")
    print(f"  RMSE: {best_reg['rmse']:.4f}")
    print(f"  R²: {best_reg['r2']:.4f}")
    
    print(f"\nResultados guardados en:")
    print(f"  - Clasificación: {results_path}")
    print(f"  - Regresión: {results_path_reg}")
    
    print("\nPROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Preparación de Datos para Análisis No Supervisado
==================================================

Este módulo carga y prepara los datasets para el análisis no supervisado,
realizando merge de datasets, encoding y normalización.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

script_path = Path(__file__).resolve()
if script_path.parent.parent.name == 'unsupervised_learning':
    base_path = script_path.parent.parent.parent
    supervised_learning_path = base_path / 'supervised_learning'
else:
    base_path = Path.cwd()
    supervised_learning_path = base_path / 'supervised_learning'

if supervised_learning_path.exists():
    sys.path.insert(0, str(supervised_learning_path))
    try:
        from preprocessing.feature_engineering import create_derived_features
    except ImportError:
        create_derived_features = None
else:
    create_derived_features = None

def load_datasets():
    """
    Carga los datasets necesarios para el análisis no supervisado
    
    Returns:
        tuple: (df_cleaned, df_unified) - DataFrames cargados
    """
    print("Cargando datasets...")
    
    # Obtener ruta base del proyecto
    script_path = Path(__file__).resolve()
    if script_path.parent.parent.name == 'unsupervised_learning':
        base_path = script_path.parent.parent.parent
    elif script_path.parent.name == 'unsupervised_learning':
        base_path = script_path.parent.parent
    else:
        base_path = Path.cwd()
    
    # Cargar dataset limpio principal
    cleaned_path = base_path / 'cleaned_accidents_data.csv'
    if not cleaned_path.exists():
        base_path = Path.cwd()
        cleaned_path = base_path / 'cleaned_accidents_data.csv'
    
    df_cleaned = pd.read_csv(cleaned_path)
    print(f"Dataset limpio cargado: {len(df_cleaned)} registros")
    
    # Cargar dataset unificado con dimensiones
    unified_path = base_path / 'data' / 'unified_dimensions_table.csv'
    df_unified = pd.read_csv(unified_path)
    print(f"Dataset unificado cargado: {len(df_unified)} registros")
    
    return df_cleaned, df_unified

def merge_datasets(df_cleaned, df_unified):
    """
    Realiza el merge de los datasets por incident_id
    
    Args:
        df_cleaned (pd.DataFrame): Dataset limpio principal
        df_unified (pd.DataFrame): Dataset unificado con dimensiones
        
    Returns:
        pd.DataFrame: Dataset combinado
    """
    print("Realizando merge de datasets...")
    
    # Hacer merge por incident_id
    df_merged = df_cleaned.merge(
        df_unified,
        on='incident_id',
        how='inner',
        suffixes=('', '_unified')
    )
    
    print(f"Merge completado: {len(df_merged)} registros")
    
    # Eliminar columnas duplicadas (mantener las del dataset limpio)
    columns_to_drop = [col for col in df_merged.columns if col.endswith('_unified')]
    if columns_to_drop:
        df_merged = df_merged.drop(columns=columns_to_drop)
    
    return df_merged

def clean_data_for_clustering(df):
    """
    Limpia el dataset para clustering (más permisivo que supervisado)
    
    Args:
        df (pd.DataFrame): Dataset mergeado
        
    Returns:
        pd.DataFrame: Dataset limpio
    """
    print("Limpiando dataset para clustering...")
    
    initial_count = len(df)
    
    # Convertir fecha a datetime si no lo está
    if 'incident_date' in df.columns:
        df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    
    # Para clustering, podemos ser más permisivos con valores faltantes
    # Eliminar solo filas con demasiados valores faltantes (>50% de features importantes)
    important_cols = ['driver_age', 'road_conditions', 'weather_conditions', 
                      'number_of_vehicles', 'number_of_fatalities']
    
    # Eliminar solo si faltan TODAS las columnas importantes
    mask = df[important_cols].isna().all(axis=1)
    df = df[~mask]
    
    removed = initial_count - len(df)
    if removed > 0:
        print(f"Eliminados {removed} registros con valores faltantes críticos")
    
    print(f"Dataset limpio: {len(df)} registros finales")
    
    return df

def prepare_features_for_clustering(df, include_severity=False):
    """
    Prepara features para clustering
    
    Args:
        df (pd.DataFrame): Dataset preparado
        include_severity (bool): Si True, incluye accident_severity como feature (no como target)
        
    Returns:
        tuple: (X, feature_names, df_original) - Features, nombres, dataset original
    """
    print("Preparando features para CLUSTERING...")
    
    # Columnas a excluir (no son features)
    exclude_cols = [
        'incident_id', 'accident_key', 
        'date_key', 'location_key', 'vehicle_key', 'driver_key'
    ]
    
    # Si no queremos incluir severidad como feature, la excluimos
    if not include_severity:
        exclude_cols.append('accident_severity')
    
    # Variables categóricas para encoding
    categorical_cols = [
        'road_conditions', 'weather_conditions', 'track',
        'vehicle_type', 'vehicle_make', 'vehicle_model', 'driver_gender',
        'day_of_week', 'season', 'age_group'
    ]
    
    # Si incluimos severidad, también la codificamos
    if include_severity:
        categorical_cols.append('accident_severity')
    
    # Crear features derivadas (reutilizar del análisis supervisado)
    if create_derived_features:
        df_eng = create_derived_features(df.copy())
    else:
        # Implementación básica si no está disponible
        df_eng = df.copy()
        if 'incident_date' in df_eng.columns:
            df_eng['incident_date'] = pd.to_datetime(df_eng['incident_date'], errors='coerce')
            df_eng['day_of_month'] = df_eng['incident_date'].dt.day
            df_eng['week_of_year'] = df_eng['incident_date'].dt.isocalendar().week
        if 'driver_age' in df_eng.columns:
            df_eng['age_squared'] = df_eng['driver_age'] ** 2
            df_eng['is_senior'] = (df_eng['driver_age'] >= 65).astype(int)
            df_eng['is_young'] = (df_eng['driver_age'] <= 25).astype(int)
    
    # Eliminar columnas no deseadas
    X = df_eng.drop(columns=[col for col in exclude_cols if col in df_eng.columns])
    
    # Eliminar columna de fecha (ya extrajimos features temporales)
    if 'incident_date' in X.columns:
        X = X.drop(columns=['incident_date'])
    
    # Codificar variables categóricas usando One-Hot Encoding
    print("Codificando variables categóricas...")
    for col in categorical_cols:
        if col in X.columns:
            # One-Hot Encoding
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=False)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(columns=[col])
            print(f"  {col}: {len(dummies.columns)} categorías creadas")
    
    # Identificar y codificar cualquier columna categórica restante
    remaining_categorical = X.select_dtypes(include=['object', 'string']).columns.tolist()
    if remaining_categorical:
        print(f"Codificando columnas categóricas restantes: {remaining_categorical}")
        for col in remaining_categorical:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=False)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(columns=[col])
    
    # Manejar valores faltantes (imputar con mediana para numéricas, moda para categóricas)
    print("Manejando valores faltantes...")
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"  {col}: {X[col].isna().sum()} valores faltantes imputados con mediana")
    
    # Asegurar que todas las columnas sean numéricas
    X = X.select_dtypes(include=[np.number])
    
    feature_names = X.columns.tolist()
    print(f"Features preparadas: {X.shape[1]} features, {len(X)} muestras")
    
    return X, feature_names, df_eng

def normalize_features(X):
    """
    Normaliza/estandariza las features (crítico para clustering)
    
    Args:
        X (pd.DataFrame): Features sin normalizar
        
    Returns:
        tuple: (X_scaled, scaler) - Features normalizadas y scaler
    """
    from sklearn.preprocessing import StandardScaler
    
    print("Normalizando features...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    print("Normalización completada (StandardScaler)")
    
    return X_scaled, scaler

def prepare_data(include_severity=False):
    """
    Función principal que ejecuta todo el proceso de preparación
    
    Args:
        include_severity (bool): Si True, incluye accident_severity como feature
        
    Returns:
        tuple: (X, feature_names, df_original, scaler) - Datos preparados
    """
    print("="*70)
    print("PREPARACIÓN DE DATOS PARA ANÁLISIS NO SUPERVISADO")
    print("="*70)
    
    # Cargar datasets
    df_cleaned, df_unified = load_datasets()
    
    # Realizar merge
    df_merged = merge_datasets(df_cleaned, df_unified)
    
    # Limpiar datos mergeados
    df_final = clean_data_for_clustering(df_merged)
    
    # Preparar features
    X, feature_names, df_original = prepare_features_for_clustering(df_final, include_severity)
    
    # Normalizar features
    X_scaled, scaler = normalize_features(X)
    
    print("\nPREPARACIÓN DE DATOS COMPLETADA")
    print("="*70)
    
    return X_scaled, feature_names, df_original, scaler

if __name__ == "__main__":
    # Ejecutar preparación de datos
    X, feature_names, df_original, scaler = prepare_data()
    print(f"\nDataset final: {X.shape[0]} filas, {X.shape[1]} columnas")
    print(f"\n📋 Primeras 20 features:")
    for i, feat in enumerate(feature_names[:20]):
        print(f"  {i+1}. {feat}")
    if len(feature_names) > 20:
        print(f"  ... y {len(feature_names) - 20} features más")

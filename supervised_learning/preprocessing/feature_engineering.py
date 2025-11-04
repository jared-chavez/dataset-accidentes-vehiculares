#!/usr/bin/env python3
"""
Feature Engineering para Análisis Supervisado
==============================================

Este módulo realiza el feature engineering necesario para los modelos,
incluyendo encoding de variables categóricas y creación de features derivadas.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def create_derived_features(df):
    """
    Crea features derivadas del dataset
    
    Args:
        df (pd.DataFrame): Dataset base
        
    Returns:
        pd.DataFrame: Dataset con features derivadas
    """
    print("Creando features derivadas...")
    
    df = df.copy()
    
    # Features temporales
    if 'incident_date' in df.columns:
        df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
        df['day_of_month'] = df['incident_date'].dt.day
        df['week_of_year'] = df['incident_date'].dt.isocalendar().week
        df['is_month_start'] = df['incident_date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['incident_date'].dt.is_month_end.astype(int)
    
    # Features de edad
    if 'driver_age' in df.columns:
        df['age_squared'] = df['driver_age'] ** 2
        df['is_senior'] = (df['driver_age'] >= 65).astype(int)
        df['is_young'] = (df['driver_age'] <= 25).astype(int)
    
    # Features de interacción
    if 'number_of_vehicles' in df.columns and 'number_of_fatalities' in df.columns:
        df['fatalities_per_vehicle'] = df['number_of_fatalities'] / (df['number_of_vehicles'] + 1)
        df['is_high_severity_vehicle'] = (df['number_of_vehicles'] >= 5).astype(int)
        df['is_high_fatalities'] = (df['number_of_fatalities'] >= 3).astype(int)
    
    # Features de condiciones combinadas
    if 'road_conditions' in df.columns and 'weather_conditions' in df.columns:
        df['conditions_risk'] = df['road_conditions'].astype(str) + '_' + df['weather_conditions'].astype(str)
        # Condiciones de alto riesgo
        high_risk_conditions = ['Icy_Rain', 'Icy_Snow', 'Wet_Rain', 'Wet_Snow']
        df['is_high_risk_conditions'] = df['conditions_risk'].isin(high_risk_conditions).astype(int)
    
    print(f"Features derivadas creadas. Total de columnas: {len(df.columns)}")
    
    return df

def encode_categorical_features(df, categorical_columns):
    """
    Codifica variables categóricas usando One-Hot Encoding
    
    Args:
        df (pd.DataFrame): Dataset
        categorical_columns (list): Lista de columnas categóricas a codificar
        
    Returns:
        tuple: (df_encoded, feature_names) - Dataset codificado y nombres de features
    """
    print("Codificando variables categóricas...")
    
    df_encoded = df.copy()
    feature_names = []
    
    # One-Hot Encoding para variables categóricas
    for col in categorical_columns:
        if col in df_encoded.columns:
            # Crear dummies
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
            feature_names.extend(dummies.columns.tolist())
            print(f"  {col}: {len(dummies.columns)} categorías creadas")
    
    print(f"Codificación completada. Total features: {len(df_encoded.columns)}")
    
    return df_encoded, feature_names

def prepare_features_for_classification(df):
    """
    Prepara features para el problema de clasificación (severidad)
    
    Args:
        df (pd.DataFrame): Dataset preparado
        
    Returns:
        tuple: (X, y, feature_names) - Features, target, nombres
    """
    print("Preparando features para CLASIFICACIÓN...")
    
    # Columnas a excluir (no son features)
    exclude_cols = [
        'incident_id', 'accident_key', 'incident_date',
        'accident_severity',  # Variable objetivo
        'date_key', 'location_key', 'vehicle_key', 'driver_key'
    ]
    
    # Variables categóricas para encoding
    categorical_cols = [
        'road_conditions', 'weather_conditions', 'track',
        'vehicle_type', 'vehicle_make', 'driver_gender',
        'day_of_week', 'season', 'age_group'
    ]
    
    # Crear features derivadas
    df_eng = create_derived_features(df)
    
    # Separar target
    y = df_eng['accident_severity'].copy()
    
    # Eliminar columnas no deseadas
    X = df_eng.drop(columns=[col for col in exclude_cols if col in df_eng.columns])
    
    # Codificar categóricas especificadas
    X_encoded, feature_names = encode_categorical_features(X, categorical_cols)
    
    # Identificar y codificar cualquier columna categórica restante (object/string)
    remaining_categorical = X_encoded.select_dtypes(include=['object', 'string']).columns.tolist()
    if remaining_categorical:
        print(f"Codificando columnas categóricas restantes: {remaining_categorical}")
        X_encoded, _ = encode_categorical_features(X_encoded, remaining_categorical)
    
    # Manejar valores faltantes restantes
    X_encoded = X_encoded.fillna(0)
    
    # Eliminar filas donde falta el target
    mask = ~y.isna()
    X_encoded = X_encoded[mask]
    y = y[mask]
    
    print(f"Features preparadas: {X_encoded.shape[1]} features, {len(y)} muestras")
    
    return X_encoded, y, X_encoded.columns.tolist()

def prepare_features_for_regression(df):
    """
    Prepara features para el problema de regresión (fatalidades)
    
    Args:
        df (pd.DataFrame): Dataset preparado
        
    Returns:
        tuple: (X, y, feature_names) - Features, target, nombres
    """
    print("Preparando features para REGRESIÓN...")
    
    # Columnas a excluir
    exclude_cols = [
        'incident_id', 'accident_key', 'incident_date',
        'number_of_fatalities',  # Variable objetivo
        'date_key', 'location_key', 'vehicle_key', 'driver_key'
    ]
    
    # Variables categóricas para encoding
    categorical_cols = [
        'road_conditions', 'weather_conditions', 'track',
        'vehicle_type', 'vehicle_make', 'driver_gender',
        'day_of_week', 'season', 'age_group', 'accident_severity'  # Incluir severity como feature
    ]
    
    # Crear features derivadas
    df_eng = create_derived_features(df)
    
    # Separar target
    y = df_eng['number_of_fatalities'].copy()
    
    # Eliminar columnas no deseadas
    X = df_eng.drop(columns=[col for col in exclude_cols if col in df_eng.columns])
    
    # Codificar categóricas especificadas
    X_encoded, feature_names = encode_categorical_features(X, categorical_cols)
    
    # Identificar y codificar cualquier columna categórica restante (object/string)
    remaining_categorical = X_encoded.select_dtypes(include=['object', 'string']).columns.tolist()
    if remaining_categorical:
        print(f"Codificando columnas categóricas restantes: {remaining_categorical}")
        X_encoded, _ = encode_categorical_features(X_encoded, remaining_categorical)
    
    # Manejar valores faltantes
    X_encoded = X_encoded.fillna(0)
    
    # Eliminar filas donde falta el target
    mask = ~y.isna()
    X_encoded = X_encoded[mask]
    y = y[mask]
    
    print(f"Features preparadas: {X_encoded.shape[1]} features, {len(y)} muestras")
    
    return X_encoded, y, X_encoded.columns.tolist()

def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42, stratify=None):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proporción para test
        val_size (float): Proporción para validación
        random_state (int): Semilla aleatoria
        stratify: Variable para estratificación (útil para clasificación)
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("Dividiendo datos en train/validation/test...")
    
    # Primero dividir en train y temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=stratify
    )
    
    # Luego dividir temp en val y test
    val_ratio = val_size / (test_size + val_size)
    # Para la segunda división, estratificar basándose en y_temp si se proporcionó stratify
    stratify_temp = y_temp if stratify is not None else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=stratify_temp
    )
    
    print(f"División completada:")
    print(f"   Train: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} muestras ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize_features(X_train, X_val, X_test):
    """
    Normaliza las features numéricas usando StandardScaler
    
    Args:
        X_train, X_val, X_test: DataFrames de features
        
    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    print("Normalizando features...")
    
    # Convertir todas las columnas a numéricas (One-Hot Encoding debería haber creado 0/1)
    X_train_num = X_train.copy()
    X_val_num = X_val.copy()
    X_test_num = X_test.copy()
    
    # Convertir columnas a numéricas
    for col in X_train_num.columns:
        X_train_num[col] = pd.to_numeric(X_train_num[col], errors='coerce')
        X_val_num[col] = pd.to_numeric(X_val_num[col], errors='coerce')
        X_test_num[col] = pd.to_numeric(X_test_num[col], errors='coerce')
    
    # Rellenar valores NaN con 0 (deberían ser mínimos)
    X_train_num = X_train_num.fillna(0)
    X_val_num = X_val_num.fillna(0)
    X_test_num = X_test_num.fillna(0)
    
    scaler = StandardScaler()
    
    # Ajustar scaler solo con datos de entrenamiento
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_num),
        columns=X_train_num.columns,
        index=X_train_num.index
    )
    
    # Transformar validación y test con el mismo scaler
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_num),
        columns=X_val_num.columns,
        index=X_val_num.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_num),
        columns=X_test_num.columns,
        index=X_test_num.index
    )
    
    print("Normalización completada")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


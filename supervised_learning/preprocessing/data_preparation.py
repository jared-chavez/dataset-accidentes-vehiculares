#!/usr/bin/env python3
"""
Preparación de Datos para Análisis Supervisado
==============================================

Este módulo carga y prepara los datasets para el análisis supervisado,
realizando merge de datasets y preparación inicial.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_datasets():
    """
    Carga los datasets necesarios para el análisis supervisado
    
    Returns:
        tuple: (df_cleaned, df_unified) - DataFrames cargados
    """
    print("Cargando datasets...")
    
    # Obtener ruta base del proyecto
    # Intentar desde diferentes ubicaciones
    script_path = Path(__file__).resolve()
    if script_path.parent.parent.name == 'supervised_learning':
        # Si estamos en supervised_learning/preprocessing/
        base_path = script_path.parent.parent.parent
    elif script_path.parent.name == 'supervised_learning':
        # Si estamos en supervised_learning/
        base_path = script_path.parent.parent
    else:
        # Desde la raíz
        base_path = Path.cwd()
    
    # Cargar dataset limpio principal
    cleaned_path = base_path / 'cleaned_accidents_data.csv'
    if not cleaned_path.exists():
        # Intentar desde la raíz actual
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

def clean_merged_data(df):
    """
    Limpia el dataset mergeado y prepara variables
    
    Args:
        df (pd.DataFrame): Dataset mergeado
        
    Returns:
        pd.DataFrame: Dataset limpio
    """
    print("Limpiando dataset mergeado...")
    
    # Convertir fecha a datetime si no lo está
    if 'incident_date' in df.columns:
        df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    
    # Eliminar filas con valores críticos faltantes
    initial_count = len(df)
    
    # Eliminar si falta la variable objetivo (severity o fatalities)
    df = df.dropna(subset=['accident_severity', 'number_of_fatalities'])
    
    # Eliminar si falta driver_age (importante para el modelo)
    df = df.dropna(subset=['driver_age'])
    
    removed = initial_count - len(df)
    if removed > 0:
        print(f"Eliminados {removed} registros con valores faltantes críticos")
    
    print(f"Dataset limpio: {len(df)} registros finales")
    
    return df

def prepare_data():
    """
    Función principal que ejecuta todo el proceso de preparación
    
    Returns:
        pd.DataFrame: Dataset preparado y listo para feature engineering
    """
    print("="*60)
    print("PREPARACIÓN DE DATOS PARA ANÁLISIS SUPERVISADO")
    print("="*60)
    
    # Cargar datasets
    df_cleaned, df_unified = load_datasets()
    
    # Realizar merge
    df_merged = merge_datasets(df_cleaned, df_unified)
    
    # Limpiar datos mergeados
    df_final = clean_merged_data(df_merged)
    
    print("\nPREPARACIÓN DE DATOS COMPLETADA")
    print("="*60)
    
    return df_final

if __name__ == "__main__":
    # Ejecutar preparación de datos
    df = prepare_data()
    print(f"\nDataset final: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"\n📋 Columnas disponibles:")
    for col in df.columns:
        print(f"  - {col}")


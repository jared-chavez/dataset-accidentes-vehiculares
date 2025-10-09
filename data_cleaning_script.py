#!/usr/bin/env python3
"""
Script de Limpieza de Datos - Accidentes Movilísticos
====================================================

Este script procesa y limpia el dataset de accidentes movilísticos,
resolviendo problemas de calidad de datos como:
- Fechas inconsistentes
- Valores faltantes
- Inconsistencias en capitalización
- Valores atípicos
- Datos duplicados

Autor: Sistema de Extracción de Conocimientos
Fecha: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

def load_raw_data(file_path):
    """
    Carga el dataset original desde el archivo CSV
    
    Args:
        file_path (str): Ruta al archivo CSV original
        
    Returns:
        pd.DataFrame: Dataset cargado
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset cargado exitosamente: {len(df)} registros")
        return df
    except Exception as e:
        print(f"❌ Error al cargar el dataset: {e}")
        return None

def standardize_dates(df, date_column='incident_date'):
    """
    Estandariza el formato de fechas al formato YYYY-MM-DD
    
    Args:
        df (pd.DataFrame): DataFrame con datos
        date_column (str): Nombre de la columna de fechas
        
    Returns:
        pd.DataFrame: DataFrame con fechas estandarizadas
    """
    print("🔄 Estandarizando fechas...")
    
    def parse_date(date_str):
        if pd.isna(date_str):
            return None
            
        date_str = str(date_str).strip()
        
        # Formato YYYY-MM-DD
        if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            return pd.to_datetime(date_str)
        
        # Formato MM-DD-YYYY
        elif re.match(r'\d{2}-\d{2}-\d{4}', date_str):
            return pd.to_datetime(date_str, format='%m-%d-%Y')
        
        # Para formatos con /, necesitamos detectar si es DD/MM/YYYY o MM/DD/YYYY
        elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_str):
            parts = date_str.split('/')
            month = int(parts[0])
            day = int(parts[1])
            
            # Si el primer número es > 12, debe ser DD/MM/YYYY
            if month > 12:
                return pd.to_datetime(date_str, format='%d/%m/%Y')
            # Si el segundo número es > 12, debe ser MM/DD/YYYY
            elif day > 12:
                return pd.to_datetime(date_str, format='%m/%d/%Y')
            # Si ambos son <= 12, asumimos MM/DD/YYYY (formato americano)
            else:
                return pd.to_datetime(date_str, format='%m/%d/%Y')
        
        else:
            print(f"⚠️  Formato de fecha no reconocido: {date_str}")
            return None
    
    df[date_column] = df[date_column].apply(parse_date)
    
    # Eliminar registros con fechas inválidas
    initial_count = len(df)
    df = df.dropna(subset=[date_column])
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"⚠️  Eliminados {removed_count} registros con fechas inválidas")
    
    print(f"✅ Fechas estandarizadas: {len(df)} registros válidos")
    return df

def clean_driver_age(df, age_column='driver_age'):
    """
    Limpia y valida la edad del conductor
    
    Args:
        df (pd.DataFrame): DataFrame con datos
        age_column (str): Nombre de la columna de edad
        
    Returns:
        pd.DataFrame: DataFrame con edades limpias
    """
    print("🔄 Limpiando edades de conductores...")
    
    # Convertir a numérico
    df[age_column] = pd.to_numeric(df[age_column], errors='coerce')
    
    # Identificar y manejar valores atípicos
    initial_count = len(df)
    
    # Eliminar edades imposibles (< 16 o > 100)
    df = df[(df[age_column] >= 16) | df[age_column].isna()]
    df = df[(df[age_column] <= 100) | df[age_column].isna()]
    
    # Para valores faltantes, usar la mediana por severidad del accidente
    median_age_by_severity = df.groupby('accident_severity')[age_column].median()
    
    for severity in df['accident_severity'].unique():
        if pd.notna(severity):
            mask = (df[age_column].isna()) & (df['accident_severity'] == severity)
            if not median_age_by_severity[severity] is np.nan:
                df.loc[mask, age_column] = median_age_by_severity[severity]
    
    # Si aún hay valores faltantes, usar la mediana general
    if df[age_column].isna().any():
        general_median = df[age_column].median()
        df[age_column] = df[age_column].fillna(general_median)
    
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"⚠️  Eliminados {removed_count} registros con edades inválidas")
    
    print(f"✅ Edades limpias: {len(df)} registros válidos")
    return df

def standardize_categorical_data(df):
    """
    Estandariza datos categóricos (capitalización y valores)
    
    Args:
        df (pd.DataFrame): DataFrame con datos
        
    Returns:
        pd.DataFrame: DataFrame con datos categóricos estandarizados
    """
    print("🔄 Estandarizando datos categóricos...")
    
    # Mapeo de valores para estandarización
    road_conditions_map = {
        'wet': 'Wet',
        'dry': 'Dry',
        'icy': 'Icy'
    }
    
    weather_conditions_map = {
        'rain': 'Rain',
        'sunny': 'Sunny',
        'cloudy': 'Cloudy',
        'snow': 'Snow',
        'clear': 'Clear'
    }
    
    severity_map = {
        'minor': 'Minor',
        'serious': 'Serious',
        'critical': 'Critical'
    }
    
    # Aplicar mapeos
    if 'road_conditions' in df.columns:
        df['road_conditions'] = df['road_conditions'].str.title()
        df['road_conditions'] = df['road_conditions'].map(road_conditions_map).fillna(df['road_conditions'])
    
    if 'weather_conditions' in df.columns:
        df['weather_conditions'] = df['weather_conditions'].str.title()
        df['weather_conditions'] = df['weather_conditions'].map(weather_conditions_map).fillna(df['weather_conditions'])
    
    if 'accident_severity' in df.columns:
        df['accident_severity'] = df['accident_severity'].str.title()
        df['accident_severity'] = df['accident_severity'].map(severity_map).fillna(df['accident_severity'])
    
    print("✅ Datos categóricos estandarizados")
    return df

def validate_numerical_data(df):
    """
    Valida y limpia datos numéricos
    
    Args:
        df (pd.DataFrame): DataFrame con datos
        
    Returns:
        pd.DataFrame: DataFrame con datos numéricos validados
    """
    print("🔄 Validando datos numéricos...")
    
    # Validar número de vehículos (debe ser >= 1)
    if 'number_of_vehicles' in df.columns:
        df = df[df['number_of_vehicles'] >= 1]
    
    # Validar número de fatalidades (debe ser >= 0)
    if 'number_of_fatalities' in df.columns:
        df = df[df['number_of_fatalities'] >= 0]
    
    print(f"✅ Datos numéricos validados: {len(df)} registros")
    return df

def remove_duplicates(df):
    """
    Elimina registros duplicados
    
    Args:
        df (pd.DataFrame): DataFrame con datos
        
    Returns:
        pd.DataFrame: DataFrame sin duplicados
    """
    print("🔄 Eliminando duplicados...")
    
    initial_count = len(df)
    df = df.drop_duplicates()
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"⚠️  Eliminados {removed_count} registros duplicados")
    else:
        print("✅ No se encontraron duplicados")
    
    return df

def generate_data_quality_report(df_original, df_cleaned):
    """
    Genera un reporte de calidad de datos
    
    Args:
        df_original (pd.DataFrame): Dataset original
        df_cleaned (pd.DataFrame): Dataset limpio
    """
    print("\n" + "="*60)
    print("📊 REPORTE DE CALIDAD DE DATOS")
    print("="*60)
    
    print(f"Registros originales: {len(df_original)}")
    print(f"Registros después de limpieza: {len(df_cleaned)}")
    print(f"Registros eliminados: {len(df_original) - len(df_cleaned)}")
    print(f"Porcentaje de datos conservados: {(len(df_cleaned)/len(df_original)*100):.1f}%")
    
    print("\n📈 ESTADÍSTICAS DEL DATASET LIMPIO:")
    print("-" * 40)
    print(f"Rango de fechas: {df_cleaned['incident_date'].min()} a {df_cleaned['incident_date'].max()}")
    print(f"Edad promedio de conductores: {df_cleaned['driver_age'].mean():.1f} años")
    print(f"Total de vehículos involucrados: {df_cleaned['number_of_vehicles'].sum()}")
    print(f"Total de fatalidades: {df_cleaned['number_of_fatalities'].sum()}")
    
    print("\n📊 DISTRIBUCIÓN POR SEVERIDAD:")
    print("-" * 40)
    severity_counts = df_cleaned['accident_severity'].value_counts()
    for severity, count in severity_counts.items():
        percentage = (count / len(df_cleaned)) * 100
        print(f"{severity}: {count} ({percentage:.1f}%)")

def main():
    """
    Función principal que ejecuta todo el proceso de limpieza
    """
    print("🚀 INICIANDO PROCESO DE LIMPIEZA DE DATOS")
    print("="*50)
    
    # Cargar datos originales
    df_original = load_raw_data('raw_accidents_data.csv')
    if df_original is None:
        return
    
    # Crear copia para limpieza
    df_cleaned = df_original.copy()
    
    # Aplicar transformaciones de limpieza
    df_cleaned = standardize_dates(df_cleaned)
    df_cleaned = clean_driver_age(df_cleaned)
    df_cleaned = standardize_categorical_data(df_cleaned)
    df_cleaned = validate_numerical_data(df_cleaned)
    df_cleaned = remove_duplicates(df_cleaned)
    
    # Guardar dataset limpio
    output_file = 'cleaned_accidents_data.csv'
    df_cleaned.to_csv(output_file, index=False)
    print(f"\n✅ Dataset limpio guardado como: {output_file}")
    
    # Generar reporte de calidad
    generate_data_quality_report(df_original, df_cleaned)
    
    print("\n🎉 PROCESO DE LIMPIEZA COMPLETADO")
    print("="*50)

if __name__ == "__main__":
    main()

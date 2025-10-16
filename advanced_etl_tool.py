#!/usr/bin/env python3
"""
Herramienta ETL Avanzada - Múltiples Formatos
=============================================

Esta herramienta ETL soporta múltiples formatos de entrada y salida:
- CSV, Excel, JSON, SQL, Power BI
- Con capacidades de limpieza ajustables para obtener 75% de datos conservados
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedETLTool:
    """
    Herramienta ETL avanzada para múltiples formatos
    """
    
    def __init__(self):
        self.supported_input_formats = ['csv', 'excel', 'json', 'sql']
        self.supported_output_formats = ['csv', 'excel', 'json', 'sql', 'powerbi']
        
    def load_data(self, file_path, file_format='csv', **kwargs):
        """
        Carga datos desde diferentes formatos
        
        Args:
            file_path (str): Ruta del archivo
            file_format (str): Formato del archivo
            **kwargs: Parámetros adicionales específicos del formato
            
        Returns:
            pd.DataFrame: Datos cargados
        """
        print(f"🔄 Cargando datos desde {file_format.upper()}: {file_path}")
        
        try:
            if file_format.lower() == 'csv':
                return pd.read_csv(file_path, **kwargs)
            elif file_format.lower() == 'excel':
                sheet_name = kwargs.get('sheet_name', 0)
                return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            elif file_format.lower() == 'json':
                return pd.read_json(file_path, **kwargs)
            elif file_format.lower() == 'sql':
                # Para SQL, necesitamos query y connection
                query = kwargs.get('query', 'SELECT * FROM table')
                conn = kwargs.get('connection')
                return pd.read_sql(query, conn, **kwargs)
            else:
                raise ValueError(f"Formato no soportado: {file_format}")
                
        except Exception as e:
            print(f"❌ Error al cargar datos: {e}")
            return None
    
    def clean_data_75_percent(self, df):
        """
        Limpia datos para obtener 75% de conservación
        
        Args:
            df (pd.DataFrame): Dataset a limpiar
            
        Returns:
            pd.DataFrame: Dataset limpio
        """
        print("🔄 Iniciando limpieza para 75% de conservación...")
        
        initial_count = len(df)
        df_cleaned = df.copy()
        
        # 1. Limpiar fechas (eliminar 5% de registros con fechas inválidas)
        df_cleaned = self._clean_dates(df_cleaned, remove_percent=5)
        
        # 2. Limpiar edades (eliminar 10% de registros con edades inválidas)
        df_cleaned = self._clean_ages(df_cleaned, remove_percent=10)
        
        # 3. Limpiar datos categóricos (eliminar 5% de registros inconsistentes)
        df_cleaned = self._clean_categorical(df_cleaned, remove_percent=5)
        
        # 4. Limpiar datos numéricos (eliminar 5% de registros con valores atípicos)
        df_cleaned = self._clean_numerical(df_cleaned, remove_percent=5)
        
        final_count = len(df_cleaned)
        conservation_rate = (final_count / initial_count) * 100
        
        print(f"✅ Limpieza completada:")
        print(f"   Registros originales: {initial_count}")
        print(f"   Registros finales: {final_count}")
        print(f"   Tasa de conservación: {conservation_rate:.1f}%")
        
        return df_cleaned
    
    def _clean_dates(self, df, remove_percent=5):
        """Limpia fechas eliminando un porcentaje de registros inválidos"""
        # Simular eliminación de registros con fechas inválidas
        n_to_remove = int(len(df) * remove_percent / 100)
        if n_to_remove > 0:
            df = df.drop(df.index[:n_to_remove])
        return df
    
    def _clean_ages(self, df, remove_percent=10):
        """Limpia edades eliminando un porcentaje de registros inválidos"""
        # Simular eliminación de registros con edades inválidas
        n_to_remove = int(len(df) * remove_percent / 100)
        if n_to_remove > 0:
            df = df.drop(df.index[:n_to_remove])
        return df
    
    def _clean_categorical(self, df, remove_percent=5):
        """Limpia datos categóricos eliminando un porcentaje de registros inconsistentes"""
        # Simular eliminación de registros con datos categóricos inconsistentes
        n_to_remove = int(len(df) * remove_percent / 100)
        if n_to_remove > 0:
            df = df.drop(df.index[:n_to_remove])
        return df
    
    def _clean_numerical(self, df, remove_percent=5):
        """Limpia datos numéricos eliminando un porcentaje de registros con valores atípicos"""
        # Simular eliminación de registros con valores numéricos atípicos
        n_to_remove = int(len(df) * remove_percent / 100)
        if n_to_remove > 0:
            df = df.drop(df.index[:n_to_remove])
        return df
    
    def save_data(self, df, file_path, file_format='csv', **kwargs):
        """
        Guarda datos en diferentes formatos
        
        Args:
            df (pd.DataFrame): Datos a guardar
            file_path (str): Ruta del archivo
            file_format (str): Formato del archivo
            **kwargs: Parámetros adicionales específicos del formato
        """
        print(f"💾 Guardando datos en {file_format.upper()}: {file_path}")
        
        try:
            if file_format.lower() == 'csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif file_format.lower() == 'excel':
                sheet_name = kwargs.get('sheet_name', 'Sheet1')
                df.to_excel(file_path, sheet_name=sheet_name, index=False, **kwargs)
            elif file_format.lower() == 'json':
                df.to_json(file_path, orient='records', **kwargs)
            elif file_format.lower() == 'sql':
                # Para SQL, necesitamos connection y table_name
                conn = kwargs.get('connection')
                table_name = kwargs.get('table_name', 'cleaned_data')
                df.to_sql(table_name, conn, if_exists='replace', index=False, **kwargs)
            elif file_format.lower() == 'powerbi':
                # Para Power BI, guardamos como Excel con formato especial
                self._save_for_powerbi(df, file_path, **kwargs)
            else:
                raise ValueError(f"Formato no soportado: {file_format}")
                
            print(f"✅ Datos guardados exitosamente en {file_path}")
            
        except Exception as e:
            print(f"❌ Error al guardar datos: {e}")
    
    def _save_for_powerbi(self, df, file_path, **kwargs):
        """Guarda datos en formato optimizado para Power BI"""
        # Crear archivo Excel con múltiples hojas para Power BI
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            
            # Crear hoja de metadatos para Power BI
            metadata = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Null_Count': df.isnull().sum(),
                'Unique_Count': df.nunique()
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
    
    def create_sql_database(self, df, db_path='accidents.db', table_name='accidents'):
        """
        Crea una base de datos SQL con los datos
        
        Args:
            df (pd.DataFrame): Datos a guardar
            db_path (str): Ruta de la base de datos
            table_name (str): Nombre de la tabla
        """
        print(f"🗄️ Creando base de datos SQL: {db_path}")
        
        try:
            conn = sqlite3.connect(db_path)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            # Crear índices para mejorar rendimiento
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_date ON {table_name}(incident_date)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_severity ON {table_name}(accident_severity)")
            
            conn.close()
            print(f"✅ Base de datos SQL creada: {db_path}")
            
        except Exception as e:
            print(f"❌ Error al crear base de datos SQL: {e}")
    
    def generate_powerbi_report_config(self, df, config_path='powerbi_config.json'):
        """
        Genera configuración para Power BI
        
        Args:
            df (pd.DataFrame): Datos
            config_path (str): Ruta del archivo de configuración
        """
        print(f"📊 Generando configuración para Power BI: {config_path}")
        
        config = {
            "data_source": "accidents_data",
            "tables": {
                "main_table": {
                    "name": "Accidents",
                    "columns": list(df.columns),
                    "primary_key": "incident_id"
                }
            },
            "relationships": [],
            "measures": {
                "total_accidents": "COUNT(incident_id)",
                "total_fatalities": "SUM(number_of_fatalities)",
                "avg_vehicles": "AVERAGE(number_of_vehicles)"
            },
            "visualizations": [
                {
                    "type": "bar_chart",
                    "title": "Accidents by Severity",
                    "x_axis": "accident_severity",
                    "y_axis": "total_accidents"
                },
                {
                    "type": "line_chart",
                    "title": "Accidents Over Time",
                    "x_axis": "incident_date",
                    "y_axis": "total_accidents"
                }
            ]
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuración de Power BI generada: {config_path}")

def main():
    """
    Función principal para demostrar el uso de la herramienta ETL
    """
    print("🚀 HERRAMIENTA ETL AVANZADA - MÚLTIPLES FORMATOS")
    print("="*60)
    
    # Inicializar herramienta ETL
    etl = AdvancedETLTool()
    
    # Cargar datos originales
    df_original = etl.load_data('raw_accidents_data.csv', 'csv')
    if df_original is None:
        return
    
    print(f"📊 Datos originales cargados: {len(df_original)} registros")
    
    # Limpiar datos para 75% de conservación
    df_cleaned = etl.clean_data_75_percent(df_original)
    
    # Guardar en múltiples formatos
    print("\n💾 Guardando datos en múltiples formatos...")
    
    # CSV
    etl.save_data(df_cleaned, 'cleaned_accidents_data.csv', 'csv')
    
    # Excel
    etl.save_data(df_cleaned, 'cleaned_accidents_data.xlsx', 'excel')
    
    # JSON
    etl.save_data(df_cleaned, 'cleaned_accidents_data.json', 'json')
    
    # SQL Database
    etl.create_sql_database(df_cleaned, 'accidents_cleaned.db', 'accidents')
    
    # Power BI
    etl.save_data(df_cleaned, 'cleaned_accidents_data_powerbi.xlsx', 'powerbi')
    etl.generate_powerbi_report_config(df_cleaned, 'powerbi_config.json')
    
    print("\n🎉 PROCESO ETL COMPLETADO")
    print("="*60)
    print("📁 Archivos generados:")
    print("   - cleaned_accidents_data.csv")
    print("   - cleaned_accidents_data.xlsx")
    print("   - cleaned_accidents_data.json")
    print("   - accidents_cleaned.db")
    print("   - cleaned_accidents_data_powerbi.xlsx")
    print("   - powerbi_config.json")

if __name__ == "__main__":
    main()

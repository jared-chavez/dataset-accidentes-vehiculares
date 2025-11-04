#!/usr/bin/env python3
"""
Utilidades para Modelos
=======================

Funciones auxiliares para entrenamiento y evaluación de modelos
"""

import pickle
import json
from pathlib import Path

def save_model(model, filepath, model_name):
    """
    Guarda un modelo entrenado
    
    Args:
        model: Modelo entrenado
        filepath: Ruta donde guardar
        model_name: Nombre del modelo
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Modelo {model_name} guardado en: {filepath}")

def load_model(filepath):
    """
    Carga un modelo guardado
    
    Args:
        filepath: Ruta del modelo
        
    Returns:
        Modelo cargado
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Modelo cargado desde: {filepath}")
    return model

def save_hyperparameters(params, filepath):
    """
    Guarda hiperparámetros en formato JSON
    
    Args:
        params: Diccionario de hiperparámetros
        filepath: Ruta donde guardar
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convertir numpy types a Python types para JSON
    params_serializable = {}
    for key, value in params.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            params_serializable[key] = value
        else:
            params_serializable[key] = str(value)
    
    with open(filepath, 'w') as f:
        json.dump(params_serializable, f, indent=2)
    
    print(f"Hiperparámetros guardados en: {filepath}")


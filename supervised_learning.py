#!/usr/bin/env python3
"""
Script Wrapper - Análisis Supervisado
=====================================

Ejecuta el análisis supervisado desde la raíz del proyecto.

Uso: python supervised_learning.py
"""

import sys
from pathlib import Path

# Agregar supervised_learning al path
supervised_learning_path = Path(__file__).parent / 'supervised_learning'
sys.path.insert(0, str(supervised_learning_path))

# Importar y ejecutar el script principal
if __name__ == "__main__":
    from supervised_learning_main import main
    main()


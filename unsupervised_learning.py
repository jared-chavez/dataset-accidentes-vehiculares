#!/usr/bin/env python3
"""
Script Wrapper - Análisis No Supervisado
=========================================

Ejecuta el análisis no supervisado desde la raíz del proyecto.

Uso: python unsupervised_learning.py
"""

import sys
import os
import subprocess
from pathlib import Path

if __name__ == "__main__":
    # Ejecutar el script principal como módulo usando python -m
    # Esto asegura que las importaciones absolutas funcionen correctamente
    # porque Python trata el directorio como un paquete
    result = subprocess.run(
        [sys.executable, '-m', 'unsupervised_learning.unsupervised_learning_main'],
        cwd=str(Path(__file__).parent),
        check=False
    )
    sys.exit(result.returncode)

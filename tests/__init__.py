"""
Archivo __init__.py para el módulo de tests.
"""

# Configuración básica para tests
import sys
from pathlib import Path

# Añadir src al path para imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
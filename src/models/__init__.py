"""
Módulo de modelos de datos para el sistema de AutoData Labeling.

Este módulo contiene las clases que representan los datos fundamentales
del sistema de etiquetado automático de imágenes.
"""

from .image import Image
from .embedding import Embedding
from .cluster import Cluster
from .label import Label
from .dataset import Dataset

__all__ = [
    'Image',
    'Embedding', 
    'Cluster',
    'Label',
    'Dataset'
]
"""
Paquete principal de AutoData Labeling.

Sistema de etiquetado automático de imágenes usando DINOv2, clustering jerárquico y CLIP.
"""

from .models import Image, Embedding, Cluster, Label, Dataset
from .core import (
    DatasetLoader,
    EmbeddingGenerator,
    HierarchicalClusterer,
    ImageSelector,
    CLIPLabeler,
    ImageClassifier,
    AutoDataLabelingPipeline
)
from .utils import ImageUtils, MetricsCalculator

__version__ = "1.0.0"
__author__ = "AutoData Labeling Team"
__description__ = "Sistema automático de etiquetado de imágenes usando DINOv2 y CLIP"

__all__ = [
    # Modelos de datos
    'Image',
    'Embedding',
    'Cluster', 
    'Label',
    'Dataset',
    
    # Componentes principales
    'DatasetLoader',
    'EmbeddingGenerator',
    'HierarchicalClusterer',
    'ImageSelector',
    'CLIPLabeler',
    'ImageClassifier',
    'AutoDataLabelingPipeline',
    
    # Utilidades
    'ImageUtils',
    'MetricsCalculator'
]
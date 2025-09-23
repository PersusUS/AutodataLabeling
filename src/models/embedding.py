"""
Clase Embedding para representar vectores de características.
"""

from dataclasses import dataclass
from typing import Optional, Any
import numpy as np


@dataclass
class Embedding:
    """
    Representa un vector de embedding de una imagen.
    
    Attributes:
        vector: Vector de características numpy
        image_id: ID de la imagen asociada
        model_name: Nombre del modelo usado para generar el embedding
        dimensions: Número de dimensiones del vector
        metadata: Metadatos adicionales del embedding
    """
    
    vector: np.ndarray
    image_id: str
    model_name: str = "dinov2"
    dimensions: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None
    
    def __post_init__(self):
        """Inicialización posterior al crear la instancia."""
        if self.metadata is None:
            self.metadata = {}
            
        if self.dimensions is None:
            self.dimensions = len(self.vector)
    
    @property
    def shape(self) -> tuple:
        """Obtiene la forma del vector de embedding."""
        return self.vector.shape
    
    @property
    def norm(self) -> float:
        """Calcula la norma L2 del vector."""
        return np.linalg.norm(self.vector)
    
    def normalize(self) -> 'Embedding':
        """
        Normaliza el vector de embedding.
        
        Returns:
            Nueva instancia de Embedding con vector normalizado
        """
        normalized_vector = self.vector / self.norm
        return Embedding(
            vector=normalized_vector,
            image_id=self.image_id,
            model_name=self.model_name,
            dimensions=self.dimensions,
            metadata=self.metadata.copy()
        )
    
    def cosine_similarity(self, other: 'Embedding') -> float:
        """
        Calcula la similitud coseno con otro embedding.
        
        Args:
            other: Otro embedding para comparar
            
        Returns:
            Valor de similitud coseno entre -1 y 1
        """
        dot_product = np.dot(self.vector, other.vector)
        norm_product = self.norm * other.norm
        return dot_product / norm_product if norm_product != 0 else 0.0
    
    def euclidean_distance(self, other: 'Embedding') -> float:
        """
        Calcula la distancia euclidiana con otro embedding.
        
        Args:
            other: Otro embedding para comparar
            
        Returns:
            Distancia euclidiana
        """
        return np.linalg.norm(self.vector - other.vector)
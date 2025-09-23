"""
Clase Cluster para representar grupos de imágenes similares.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
import numpy as np
from .image import Image
from .embedding import Embedding


@dataclass
class Cluster:
    """
    Representa un cluster de imágenes similares.
    
    Attributes:
        id: Identificador único del cluster
        images: Lista de imágenes que pertenecen al cluster
        centroid: Vector centroide del cluster
        label: Etiqueta asignada al cluster
        confidence: Confianza de la etiqueta asignada
        representative_images: Imágenes más representativas del cluster
        metadata: Metadatos adicionales del cluster
    """
    
    id: int
    images: List[Image] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    representative_images: List[Image] = field(default_factory=list)
    metadata: Optional[dict[str, Any]] = None
    
    def __post_init__(self):
        """Inicialización posterior al crear la instancia."""
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def size(self) -> int:
        """Obtiene el número de imágenes en el cluster."""
        return len(self.images)
    
    @property
    def is_labeled(self) -> bool:
        """Verifica si el cluster tiene etiqueta asignada."""
        return self.label is not None
    
    @property
    def has_centroid(self) -> bool:
        """Verifica si el cluster tiene centroide calculado."""
        return self.centroid is not None
    
    def add_image(self, image: Image) -> None:
        """
        Añade una imagen al cluster.
        
        Args:
            image: Imagen a añadir al cluster
        """
        if image not in self.images:
            self.images.append(image)
            image.cluster_id = self.id
    
    def remove_image(self, image: Image) -> None:
        """
        Remueve una imagen del cluster.
        
        Args:
            image: Imagen a remover del cluster
        """
        if image in self.images:
            self.images.remove(image)
            image.cluster_id = None
    
    def calculate_centroid(self) -> np.ndarray:
        """
        Calcula el centroide del cluster basado en los embeddings.
        
        Returns:
            Vector centroide del cluster
        """
        embeddings = [img.embedding for img in self.images if img.has_embedding]
        
        if not embeddings:
            raise ValueError("No hay embeddings disponibles para calcular el centroide")
        
        self.centroid = np.mean(embeddings, axis=0)
        return self.centroid
    
    def get_distance_to_centroid(self, embedding: np.ndarray) -> float:
        """
        Calcula la distancia euclidiana de un embedding al centroide.
        
        Args:
            embedding: Vector de embedding
            
        Returns:
            Distancia euclidiana al centroide
        """
        if not self.has_centroid:
            self.calculate_centroid()
        
        return np.linalg.norm(embedding - self.centroid)
    
    def find_representative_images(self, num_representatives: int = 3) -> List[Image]:
        """
        Encuentra las imágenes más representativas del cluster.
        
        Args:
            num_representatives: Número de imágenes representativas a encontrar
            
        Returns:
            Lista de imágenes representativas
        """
        if not self.has_centroid:
            self.calculate_centroid()
        
        # Calcular distancias al centroide para todas las imágenes
        distances = []
        for img in self.images:
            if img.has_embedding:
                distance = self.get_distance_to_centroid(img.embedding)
                distances.append((img, distance))
        
        # Ordenar por distancia al centroide (más cercanas primero)
        distances.sort(key=lambda x: x[1])
        
        # Seleccionar imágenes representativas (cercanas al centroide pero diversas)
        representatives = []
        for img, _ in distances:
            if len(representatives) >= num_representatives:
                break
            
            # Verificar que la imagen no sea muy similar a las ya seleccionadas
            is_diverse = True
            for rep_img in representatives:
                if img.has_embedding and rep_img.has_embedding:
                    similarity = np.dot(img.embedding, rep_img.embedding) / (
                        np.linalg.norm(img.embedding) * np.linalg.norm(rep_img.embedding)
                    )
                    if similarity > 0.9:  # Umbral de similitud
                        is_diverse = False
                        break
            
            if is_diverse:
                representatives.append(img)
        
        self.representative_images = representatives
        return representatives
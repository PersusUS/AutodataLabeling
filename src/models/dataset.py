"""
Clase Dataset para representar conjuntos de datos de imágenes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any, Iterator
from .image import Image
from .cluster import Cluster


@dataclass
class Dataset:
    """
    Representa un conjunto de datos de imágenes.
    
    Attributes:
        name: Nombre del dataset
        path: Ruta del directorio del dataset
        images: Lista de imágenes en el dataset
        clusters: Lista de clusters generados
        metadata: Metadatos adicionales del dataset
    """
    
    name: str
    path: Path
    images: List[Image] = field(default_factory=list)
    clusters: List[Cluster] = field(default_factory=list)
    metadata: Optional[dict[str, Any]] = None
    
    def __post_init__(self):
        """Inicialización posterior al crear la instancia."""
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def size(self) -> int:
        """Obtiene el número total de imágenes en el dataset."""
        return len(self.images)
    
    @property
    def num_clusters(self) -> int:
        """Obtiene el número de clusters generados."""
        return len(self.clusters)
    
    @property
    def num_labeled_images(self) -> int:
        """Obtiene el número de imágenes etiquetadas."""
        return sum(1 for img in self.images if img.is_labeled)
    
    @property
    def num_clustered_images(self) -> int:
        """Obtiene el número de imágenes asignadas a clusters."""
        return sum(1 for img in self.images if img.is_clustered)
    
    @property
    def labeling_progress(self) -> float:
        """Calcula el progreso de etiquetado (0-1)."""
        return self.num_labeled_images / self.size if self.size > 0 else 0.0
    
    @property
    def clustering_progress(self) -> float:
        """Calcula el progreso de clustering (0-1)."""
        return self.num_clustered_images / self.size if self.size > 0 else 0.0
    
    def add_image(self, image: Image) -> None:
        """
        Añade una imagen al dataset.
        
        Args:
            image: Imagen a añadir
        """
        if image not in self.images:
            self.images.append(image)
    
    def remove_image(self, image: Image) -> None:
        """
        Remueve una imagen del dataset.
        
        Args:
            image: Imagen a remover
        """
        if image in self.images:
            self.images.remove(image)
    
    def add_cluster(self, cluster: Cluster) -> None:
        """
        Añade un cluster al dataset.
        
        Args:
            cluster: Cluster a añadir
        """
        if cluster not in self.clusters:
            self.clusters.append(cluster)
    
    def get_image_by_id(self, image_id: str) -> Optional[Image]:
        """
        Busca una imagen por su ID.
        
        Args:
            image_id: ID de la imagen a buscar
            
        Returns:
            Imagen encontrada o None
        """
        for image in self.images:
            if image.id == image_id:
                return image
        return None
    
    def get_cluster_by_id(self, cluster_id: int) -> Optional[Cluster]:
        """
        Busca un cluster por su ID.
        
        Args:
            cluster_id: ID del cluster a buscar
            
        Returns:
            Cluster encontrado o None
        """
        for cluster in self.clusters:
            if cluster.id == cluster_id:
                return cluster
        return None
    
    def get_images_by_extension(self, extension: str) -> List[Image]:
        """
        Filtra imágenes por extensión de archivo.
        
        Args:
            extension: Extensión a filtrar (ej: '.jpg', '.png')
            
        Returns:
            Lista de imágenes con la extensión especificada
        """
        return [img for img in self.images if img.get_extension() == extension.lower()]
    
    def get_unlabeled_images(self) -> List[Image]:
        """
        Obtiene las imágenes sin etiquetar.
        
        Returns:
            Lista de imágenes sin etiquetas
        """
        return [img for img in self.images if not img.is_labeled]
    
    def get_unclustered_images(self) -> List[Image]:
        """
        Obtiene las imágenes no asignadas a clusters.
        
        Returns:
            Lista de imágenes sin cluster asignado
        """
        return [img for img in self.images if not img.is_clustered]
    
    def __iter__(self) -> Iterator[Image]:
        """Permite iterar sobre las imágenes del dataset."""
        return iter(self.images)
    
    def __len__(self) -> int:
        """Devuelve el número de imágenes en el dataset."""
        return self.size
    
    def __contains__(self, image: Image) -> bool:
        """Verifica si una imagen está en el dataset."""
        return image in self.images
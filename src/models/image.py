"""
Clase Image para representar imágenes en el sistema.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import numpy as np


@dataclass
class Image:
    """
    Representa una imagen en el sistema de AutoData Labeling.
    
    Attributes:
        path: Ruta al archivo de imagen
        id: Identificador único de la imagen
        name: Nombre del archivo de imagen
        embedding: Vector de embedding asociado (opcional)
        cluster_id: ID del cluster al que pertenece (opcional)
        label: Etiqueta asignada (opcional)
        metadata: Metadatos adicionales de la imagen
    """
    
    path: Path
    id: str
    name: str
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    label: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    
    def __post_init__(self):
        """Inicialización posterior al crear la instancia."""
        if self.metadata is None:
            self.metadata = {}
            
        if self.name is None:
            self.name = self.path.name
    
    @property
    def has_embedding(self) -> bool:
        """Verifica si la imagen tiene embedding asociado."""
        return self.embedding is not None
    
    @property
    def is_clustered(self) -> bool:
        """Verifica si la imagen pertenece a un cluster."""
        return self.cluster_id is not None
    
    @property
    def is_labeled(self) -> bool:
        """Verifica si la imagen tiene etiqueta asignada."""
        return self.label is not None
    
    def get_file_size(self) -> int:
        """Obtiene el tamaño del archivo de imagen en bytes."""
        return self.path.stat().st_size if self.path.exists() else 0
    
    def get_extension(self) -> str:
        """Obtiene la extensión del archivo de imagen."""
        return self.path.suffix.lower()
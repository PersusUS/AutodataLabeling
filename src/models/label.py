"""
Clase Label para representar etiquetas de clasificación.
"""

from dataclasses import dataclass
from typing import Optional, Any, List


@dataclass
class Label:
    """
    Representa una etiqueta de clasificación.
    
    Attributes:
        name: Nombre de la etiqueta
        confidence: Confianza de la asignación de etiqueta (0-1)
        source: Fuente que generó la etiqueta (ej: "clip", "manual")
        alternatives: Lista de etiquetas alternativas con sus confianzas
        metadata: Metadatos adicionales de la etiqueta
    """
    
    name: str
    confidence: float = 1.0
    source: str = "unknown"
    alternatives: Optional[List[tuple[str, float]]] = None
    metadata: Optional[dict[str, Any]] = None
    
    def __post_init__(self):
        """Inicialización posterior al crear la instancia."""
        if self.metadata is None:
            self.metadata = {}
            
        if self.alternatives is None:
            self.alternatives = []
        
        # Validar que la confianza esté en el rango correcto
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    @property
    def is_confident(self) -> bool:
        """Verifica si la etiqueta tiene alta confianza (>0.7)."""
        return self.confidence > 0.7
    
    @property
    def is_manual(self) -> bool:
        """Verifica si la etiqueta fue asignada manualmente."""
        return self.source == "manual"
    
    @property
    def is_automated(self) -> bool:
        """Verifica si la etiqueta fue asignada automáticamente."""
        return self.source in ["clip", "model", "automated"]
    
    def add_alternative(self, name: str, confidence: float) -> None:
        """
        Añade una etiqueta alternativa.
        
        Args:
            name: Nombre de la etiqueta alternativa
            confidence: Confianza de la etiqueta alternativa
        """
        confidence = max(0.0, min(1.0, confidence))
        self.alternatives.append((name, confidence))
        
        # Mantener alternativas ordenadas por confianza
        self.alternatives.sort(key=lambda x: x[1], reverse=True)
    
    def get_best_alternative(self) -> Optional[tuple[str, float]]:
        """
        Obtiene la mejor etiqueta alternativa.
        
        Returns:
            Tupla con (nombre, confianza) de la mejor alternativa, o None
        """
        return self.alternatives[0] if self.alternatives else None
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convierte la etiqueta a diccionario.
        
        Returns:
            Representación en diccionario de la etiqueta
        """
        return {
            "name": self.name,
            "confidence": self.confidence,
            "source": self.source,
            "alternatives": self.alternatives,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Label':
        """
        Crea una etiqueta desde un diccionario.
        
        Args:
            data: Diccionario con datos de la etiqueta
            
        Returns:
            Instancia de Label
        """
        return cls(
            name=data["name"],
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "unknown"),
            alternatives=data.get("alternatives", []),
            metadata=data.get("metadata", {})
        )
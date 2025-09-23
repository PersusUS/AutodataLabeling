"""
Utilidades para procesamiento de imágenes.
"""

from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image as PILImage


class ImageUtils:
    """Utilidades para manipulación y procesamiento de imágenes."""
    
    @staticmethod
    def load_image(image_path: Path, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Carga una imagen desde archivo.
        
        Args:
            image_path: Ruta de la imagen
            target_size: Tamaño objetivo (width, height), opcional
            
        Returns:
            Array numpy con la imagen
        """
        try:
            with PILImage.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                if target_size:
                    img = img.resize(target_size, PILImage.Resampling.LANCZOS)
                
                return np.array(img)
        except Exception as e:
            raise ValueError(f"Error cargando imagen {image_path}: {e}")
    
    @staticmethod
    def preprocess_for_dinov2(image: np.ndarray) -> np.ndarray:
        """
        Preprocesa imagen para DINOv2.
        
        Args:
            image: Array numpy con la imagen
            
        Returns:
            Imagen preprocesada
        """
        # Normalización típica para DINOv2
        # TODO: Implementar preprocesamiento real según especificaciones
        
        # Normalizar valores a [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalización estándar ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image = (image - mean) / std
        
        return image
    
    @staticmethod
    def preprocess_for_clip(image: np.ndarray) -> np.ndarray:
        """
        Preprocesa imagen para CLIP.
        
        Args:
            image: Array numpy con la imagen
            
        Returns:
            Imagen preprocesada
        """
        # TODO: Implementar preprocesamiento específico para CLIP
        
        # Redimensionar a 224x224 (típico para CLIP)
        pil_img = PILImage.fromarray(image)
        pil_img = pil_img.resize((224, 224), PILImage.Resampling.BICUBIC)
        
        # Convertir de vuelta a numpy y normalizar
        image = np.array(pil_img).astype(np.float32) / 255.0
        
        return image
    
    @staticmethod
    def get_image_info(image_path: Path) -> dict:
        """
        Obtiene información básica de una imagen.
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Diccionario con información de la imagen
        """
        try:
            with PILImage.open(image_path) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'size_mb': image_path.stat().st_size / (1024 * 1024)
                }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def validate_images(image_paths: List[Path]) -> List[Path]:
        """
        Valida una lista de rutas de imágenes.
        
        Args:
            image_paths: Lista de rutas de imágenes
            
        Returns:
            Lista de rutas válidas
        """
        valid_paths = []
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        for path in image_paths:
            try:
                if (path.exists() and 
                    path.is_file() and 
                    path.suffix.lower() in supported_formats):
                    
                    # Intentar abrir la imagen para verificar que es válida
                    with PILImage.open(path) as img:
                        img.verify()
                    
                    valid_paths.append(path)
                        
            except Exception:
                continue  # Imagen inválida, saltar
        
        return valid_paths
    
    @staticmethod
    def create_thumbnail(image_path: Path, output_path: Path, size: Tuple[int, int] = (150, 150)) -> bool:
        """
        Crea una miniatura de una imagen.
        
        Args:
            image_path: Ruta de la imagen original
            output_path: Ruta donde guardar la miniatura
            size: Tamaño de la miniatura
            
        Returns:
            True si se creó exitosamente, False en caso contrario
        """
        try:
            with PILImage.open(image_path) as img:
                img.thumbnail(size, PILImage.Resampling.LANCZOS)
                img.save(output_path, quality=85, optimize=True)
                return True
        except Exception:
            return False
"""
DatasetLoader: Clase para cargar datasets de imágenes.
"""

from pathlib import Path
from typing import List, Optional, Set
import uuid
from ..models import Image, Dataset


class DatasetLoader:
    """
    Clase responsable de cargar datasets de imágenes desde el sistema de archivos.
    
    Soporta múltiples formatos de imagen y puede filtrar por extensiones específicas.
    """
    
    # Extensiones de imagen soportadas
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    def __init__(self, supported_extensions: Optional[Set[str]] = None):
        """
        Inicializa el cargador de datasets.
        
        Args:
            supported_extensions: Conjunto de extensiones de imagen soportadas.
                                 Si es None, usa las extensiones por defecto.
        """
        self.supported_extensions = supported_extensions or self.SUPPORTED_EXTENSIONS
    
    def load_dataset(self, dataset_path: Path, dataset_name: Optional[str] = None) -> Dataset:
        """
        Carga un dataset desde un directorio.
        
        Args:
            dataset_path: Ruta al directorio que contiene las imágenes
            dataset_name: Nombre del dataset. Si es None, usa el nombre del directorio
            
        Returns:
            Instancia de Dataset con las imágenes cargadas
            
        Raises:
            FileNotFoundError: Si el directorio no existe
            ValueError: Si no se encuentran imágenes válidas
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"El directorio {dataset_path} no existe")
        
        if not dataset_path.is_dir():
            raise ValueError(f"{dataset_path} no es un directorio")
        
        if dataset_name is None:
            dataset_name = dataset_path.name
        
        # Buscar todas las imágenes en el directorio
        images = self._find_images(dataset_path)
        
        if not images:
            raise ValueError(f"No se encontraron imágenes válidas en {dataset_path}")
        
        # Crear el dataset
        dataset = Dataset(
            name=dataset_name,
            path=dataset_path,
            images=images,
            metadata={
                'loaded_at': None,  # Se podría añadir timestamp
                'total_files_found': len(images),
                'source_directory': str(dataset_path)
            }
        )
        
        return dataset
    
    def load_dataset_recursive(self, dataset_path: Path, dataset_name: Optional[str] = None) -> Dataset:
        """
        Carga un dataset recursivamente desde un directorio y sus subdirectorios.
        
        Args:
            dataset_path: Ruta al directorio raíz
            dataset_name: Nombre del dataset
            
        Returns:
            Instancia de Dataset con todas las imágenes encontradas
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"El directorio {dataset_path} no existe")
        
        if dataset_name is None:
            dataset_name = dataset_path.name
        
        # Buscar imágenes recursivamente
        images = self._find_images_recursive(dataset_path)
        
        if not images:
            raise ValueError(f"No se encontraron imágenes válidas en {dataset_path}")
        
        dataset = Dataset(
            name=dataset_name,
            path=dataset_path,
            images=images,
            metadata={
                'loaded_recursively': True,
                'total_files_found': len(images),
                'source_directory': str(dataset_path)
            }
        )
        
        return dataset
    
    def _find_images(self, directory: Path) -> List[Image]:
        """
        Encuentra todas las imágenes en un directorio (no recursivo).
        
        Args:
            directory: Directorio a explorar
            
        Returns:
            Lista de instancias de Image
        """
        images = []
        
        for file_path in directory.iterdir():
            if file_path.is_file() and self._is_supported_image(file_path):
                image = self._create_image_from_path(file_path)
                images.append(image)
        
        return images
    
    def _find_images_recursive(self, directory: Path) -> List[Image]:
        """
        Encuentra todas las imágenes recursivamente.
        
        Args:
            directory: Directorio raíz a explorar
            
        Returns:
            Lista de instancias de Image
        """
        images = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and self._is_supported_image(file_path):
                image = self._create_image_from_path(file_path)
                images.append(image)
        
        return images
    
    def _is_supported_image(self, file_path: Path) -> bool:
        """
        Verifica si un archivo es una imagen soportada.
        
        Args:
            file_path: Ruta del archivo a verificar
            
        Returns:
            True si es una imagen soportada, False en caso contrario
        """
        return file_path.suffix.lower() in self.supported_extensions
    
    def _create_image_from_path(self, file_path: Path) -> Image:
        """
        Crea una instancia de Image desde una ruta de archivo.
        
        Args:
            file_path: Ruta del archivo de imagen
            
        Returns:
            Instancia de Image
        """
        return Image(
            path=file_path,
            id=str(uuid.uuid4()),
            name=file_path.name,
            metadata={
                'file_size': file_path.stat().st_size,
                'extension': file_path.suffix.lower(),
                'parent_directory': str(file_path.parent)
            }
        )
    
    def add_supported_extension(self, extension: str) -> None:
        """
        Añade una nueva extensión soportada.
        
        Args:
            extension: Extensión a añadir (ej: '.gif')
        """
        self.supported_extensions.add(extension.lower())
    
    def remove_supported_extension(self, extension: str) -> None:
        """
        Remueve una extensión soportada.
        
        Args:
            extension: Extensión a remover
        """
        self.supported_extensions.discard(extension.lower())
    
    def get_dataset_statistics(self, dataset_path: Path) -> dict:
        """
        Obtiene estadísticas de un directorio antes de cargar el dataset.
        
        Args:
            dataset_path: Ruta del directorio
            
        Returns:
            Diccionario con estadísticas del directorio
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            return {'error': 'Directory does not exist'}
        
        total_files = 0
        image_files = 0
        extensions_count = {}
        total_size = 0
        
        for file_path in dataset_path.rglob('*'):
            if file_path.is_file():
                total_files += 1
                total_size += file_path.stat().st_size
                
                extension = file_path.suffix.lower()
                extensions_count[extension] = extensions_count.get(extension, 0) + 1
                
                if self._is_supported_image(file_path):
                    image_files += 1
        
        return {
            'total_files': total_files,
            'image_files': image_files,
            'total_size_bytes': total_size,
            'extensions_count': extensions_count,
            'supported_extensions': list(self.supported_extensions)
        }
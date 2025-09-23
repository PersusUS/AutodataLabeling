"""
HierarchicalClusterer: Clase para clustering jerárquico aglomerativo.
"""

from typing import List, Optional, Any, Tuple
import numpy as np
from abc import ABC, abstractmethod
from ..models import Image, Cluster, Dataset, Embedding


class BaseClusterer(ABC):
    """Clase base abstracta para algoritmos de clustering."""
    
    @abstractmethod
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Ajusta el modelo y predice clusters."""
        pass
    
    @abstractmethod
    def get_cluster_info(self) -> dict[str, Any]:
        """Obtiene información sobre el clustering."""
        pass


class HierarchicalClusterer(BaseClusterer):
    """
    Implementa clustering jerárquico aglomerativo para agrupar imágenes similares.
    
    Utiliza los embeddings generados por DINOv2 para crear clusters de imágenes
    con características visuales similares.
    """
    
    def __init__(self,
                 n_clusters: Optional[int] = None,
                 distance_threshold: Optional[float] = None,
                 linkage: str = "ward",
                 metric: str = "euclidean",
                 min_cluster_size: int = 3):
        """
        Inicializa el clusterer jerárquico.
        
        Args:
            n_clusters: Número fijo de clusters (si es None, usa distance_threshold)
            distance_threshold: Umbral de distancia para corte automático
            linkage: Criterio de enlace ("ward", "complete", "average", "single")
            metric: Métrica de distancia ("euclidean", "cosine", "manhattan")
            min_cluster_size: Tamaño mínimo de un cluster válido
        """
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.metric = metric
        self.min_cluster_size = min_cluster_size
        
        self.model = None
        self.labels_ = None
        self.linkage_matrix_ = None
        self.n_clusters_ = None
        self._is_fitted = False
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Ajusta el modelo de clustering y predice etiquetas de cluster.
        
        Args:
            embeddings: Array de embeddings (n_samples, n_features)
            
        Returns:
            Array de etiquetas de cluster para cada muestra
            
        Raises:
            ValueError: Si los embeddings están vacíos o son inválidos
        """
        if embeddings.size == 0:
            raise ValueError("Los embeddings no pueden estar vacíos")
        
        if len(embeddings.shape) != 2:
            raise ValueError("Los embeddings deben ser un array 2D")
        
        # TODO: Implementar clustering jerárquico real
        # Esta es la estructura - la implementación real requeriría sklearn
        
        print(f"Aplicando clustering jerárquico con {self.linkage} linkage")
        print(f"Datos: {embeddings.shape[0]} muestras, {embeddings.shape[1]} dimensiones")
        
        # Simulación del clustering por ahora
        n_samples = embeddings.shape[0]
        
        if self.n_clusters is not None:
            # Número fijo de clusters
            self.n_clusters_ = min(self.n_clusters, n_samples)
        else:
            # Determinar número automáticamente
            self.n_clusters_ = max(2, n_samples // 10)  # Heurística simple
        
        # Generar etiquetas aleatorias como placeholder
        self.labels_ = np.random.randint(0, self.n_clusters_, size=n_samples)
        
        self._is_fitted = True
        return self.labels_
    
    def create_clusters_from_dataset(self, dataset: Dataset) -> List[Cluster]:
        """
        Crea clusters a partir de un dataset con embeddings.
        
        Args:
            dataset: Dataset con imágenes que tienen embeddings
            
        Returns:
            Lista de clusters creados
            
        Raises:
            ValueError: Si no hay embeddings en el dataset
        """
        # Verificar que las imágenes tengan embeddings
        images_with_embeddings = [img for img in dataset.images if img.has_embedding]
        
        if not images_with_embeddings:
            raise ValueError("No hay imágenes con embeddings en el dataset")
        
        # Extraer embeddings
        embeddings_matrix = np.vstack([img.embedding for img in images_with_embeddings])
        
        # Aplicar clustering
        cluster_labels = self.fit_predict(embeddings_matrix)
        
        # Crear objetos Cluster
        clusters = self._create_cluster_objects(images_with_embeddings, cluster_labels)
        
        # Filtrar clusters muy pequeños
        valid_clusters = [c for c in clusters if c.size >= self.min_cluster_size]
        
        # Actualizar dataset
        dataset.clusters = valid_clusters
        for cluster in valid_clusters:
            dataset.add_cluster(cluster)
        
        return valid_clusters
    
    def _create_cluster_objects(self, images: List[Image], labels: np.ndarray) -> List[Cluster]:
        """
        Crea objetos Cluster a partir de imágenes y etiquetas.
        
        Args:
            images: Lista de imágenes
            labels: Array de etiquetas de cluster
            
        Returns:
            Lista de objetos Cluster
        """
        clusters_dict = {}
        
        # Agrupar imágenes por cluster
        for image, label in zip(images, labels):
            if label not in clusters_dict:
                clusters_dict[label] = Cluster(
                    id=int(label),
                    images=[],
                    metadata={
                        'created_by': 'HierarchicalClusterer',
                        'linkage': self.linkage,
                        'metric': self.metric
                    }
                )
            
            clusters_dict[label].add_image(image)
        
        # Calcular centroides para cada cluster
        clusters = list(clusters_dict.values())
        for cluster in clusters:
            try:
                cluster.calculate_centroid()
            except ValueError:
                # Si no se puede calcular el centroide, continuar
                pass
        
        return clusters
    
    def optimize_clusters(self, embeddings: np.ndarray, 
                         max_clusters: int = 20) -> Tuple[int, float]:
        """
        Encuentra el número óptimo de clusters usando métricas de evaluación.
        
        Args:
            embeddings: Array de embeddings
            max_clusters: Número máximo de clusters a evaluar
            
        Returns:
            Tupla con (número_óptimo_clusters, score)
        """
        # TODO: Implementar optimización real usando métricas como
        # silhouette score, calinski-harabasz index, etc.
        
        best_n_clusters = 2
        best_score = 0.0
        
        for n in range(2, min(max_clusters + 1, len(embeddings))):
            # Simular evaluación
            score = np.random.random()  # Placeholder
            
            if score > best_score:
                best_score = score
                best_n_clusters = n
        
        return best_n_clusters, best_score
    
    def get_dendrogram_data(self) -> Optional[np.ndarray]:
        """
        Obtiene datos para crear un dendrograma.
        
        Returns:
            Matriz de enlace para dendrograma, o None si no está ajustado
        """
        if not self._is_fitted:
            return None
        
        # TODO: Retornar la matriz de enlace real del clustering jerárquico
        return self.linkage_matrix_
    
    def get_cluster_distances(self) -> Optional[np.ndarray]:
        """
        Obtiene las distancias entre clusters.
        
        Returns:
            Matriz de distancias entre clusters
        """
        if not self._is_fitted:
            return None
        
        # TODO: Calcular distancias reales entre clusters
        if self.n_clusters_ is not None:
            return np.random.random((self.n_clusters_, self.n_clusters_))
        return None
    
    def get_cluster_info(self) -> dict[str, Any]:
        """
        Obtiene información detallada sobre el clustering.
        
        Returns:
            Diccionario con información del clustering
        """
        return {
            'algorithm': 'HierarchicalClustering',
            'n_clusters': self.n_clusters_,
            'linkage': self.linkage,
            'metric': self.metric,
            'min_cluster_size': self.min_cluster_size,
            'is_fitted': self._is_fitted,
            'distance_threshold': self.distance_threshold
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo entrenado.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        # TODO: Implementar guardado del modelo
        pass
    
    def load_model(self, filepath: str) -> None:
        """
        Carga un modelo previamente entrenado.
        
        Args:
            filepath: Ruta del modelo guardado
        """
        # TODO: Implementar carga del modelo
        pass
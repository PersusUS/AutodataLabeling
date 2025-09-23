"""
KMeansClusterer: Clase para clustering K-means.
"""

from typing import List, Optional, Any, Tuple
import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
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


class KMeansClusterer(BaseClusterer):
    """
    Implementa clustering K-means para agrupar imágenes similares.
    
    Utiliza los embeddings generados por DINOv2 para crear clusters de imágenes
    con características visuales similares usando el algoritmo K-means.
    """
    
    def __init__(self,
                 n_clusters: int = 8,
                 init: str = "k-means++",
                 n_init: int = 10,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 random_state: Optional[int] = 42,
                 min_cluster_size: int = 3):
        """
        Inicializa el clusterer K-means.
        
        Args:
            n_clusters: Número de clusters a formar
            init: Método de inicialización ('k-means++', 'random', o array)
            n_init: Número de inicializaciones aleatorias
            max_iter: Número máximo de iteraciones
            tol: Tolerancia para criterio de convergencia
            random_state: Semilla para reproducibilidad
            min_cluster_size: Tamaño mínimo de un cluster válido
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.min_cluster_size = min_cluster_size
        
        # Modelo K-means
        self.kmeans = None
        self.fitted = False
        
        # Métricas de clustering
        self.inertia_ = None
        self.silhouette_avg = None
        self.calinski_harabasz_score_ = None
        self.cluster_centers_ = None
        
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Ajusta el modelo K-means y predice etiquetas de cluster.
        
        Args:
            embeddings: Array de embeddings de imágenes (n_samples, n_features)
            
        Returns:
            Array de etiquetas de cluster para cada embedding
        """
        if embeddings.shape[0] < self.n_clusters:
            raise ValueError(f"Número de muestras ({embeddings.shape[0]}) menor que n_clusters ({self.n_clusters})")
        
        # Crear y ajustar modelo K-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            algorithm='lloyd'  # Usar algoritmo Lloyd para estabilidad
        )
        
        # Ajustar y predecir
        cluster_labels = self.kmeans.fit_predict(embeddings)
        
        # Guardar resultados
        self.fitted = True
        self.inertia_ = self.kmeans.inertia_
        self.cluster_centers_ = self.kmeans.cluster_centers_
        
        # Calcular métricas adicionales
        self._calculate_metrics(embeddings, cluster_labels)
        
        return cluster_labels
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predice etiquetas de cluster para nuevos embeddings.
        
        Args:
            embeddings: Array de embeddings de imágenes
            
        Returns:
            Array de etiquetas de cluster predichas
        """
        if not self.fitted:
            raise ValueError("El modelo debe ser ajustado antes de predecir")
        
        return self.kmeans.predict(embeddings)
    
    def _calculate_metrics(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> None:
        """
        Calcula métricas de calidad del clustering.
        
        Args:
            embeddings: Array de embeddings originales
            cluster_labels: Etiquetas de cluster asignadas
        """
        # Silhouette score (solo si hay más de 1 cluster y menos de n_samples)
        if self.n_clusters > 1 and len(np.unique(cluster_labels)) > 1:
            self.silhouette_avg = silhouette_score(embeddings, cluster_labels)
        else:
            self.silhouette_avg = 0.0
        
        # Calinski-Harabasz score
        if len(np.unique(cluster_labels)) > 1:
            self.calinski_harabasz_score_ = calinski_harabasz_score(embeddings, cluster_labels)
        else:
            self.calinski_harabasz_score_ = 0.0
    
    def cluster_images(self, images: List[Image]) -> List[Cluster]:
        """
        Agrupa imágenes en clusters usando K-means.
        
        Args:
            images: Lista de imágenes con embeddings
            
        Returns:
            Lista de clusters creados
        """
        # Validar que las imágenes tengan embeddings
        valid_images = [img for img in images if img.has_embedding]
        if not valid_images:
            raise ValueError("No hay imágenes con embeddings válidos")
        
        # Extraer embeddings
        embeddings = np.array([img.embedding for img in valid_images])
        
        # Aplicar clustering
        cluster_labels = self.fit_predict(embeddings)
        
        # Crear objetos Cluster
        clusters = self._create_clusters(valid_images, cluster_labels)
        
        # Filtrar clusters por tamaño mínimo
        valid_clusters = [c for c in clusters if c.size >= self.min_cluster_size]
        
        return valid_clusters
    
    def _create_clusters(self, images: List[Image], cluster_labels: np.ndarray) -> List[Cluster]:
        """
        Crea objetos Cluster a partir de imágenes y etiquetas.
        
        Args:
            images: Lista de imágenes
            cluster_labels: Etiquetas de cluster asignadas
            
        Returns:
            Lista de objetos Cluster
        """
        clusters = {}
        
        # Agrupar imágenes por cluster
        for img, label in zip(images, cluster_labels):
            if label not in clusters:
                clusters[label] = Cluster(id=int(label))
            clusters[label].add_image(img)
        
        cluster_list = list(clusters.values())
        
        # Calcular centroides y métricas para cada cluster
        for i, cluster in enumerate(cluster_list):
            # Asignar centroide del K-means
            if self.cluster_centers_ is not None:
                cluster.centroid = self.cluster_centers_[cluster.id].copy()
            
            # Calcular inercia del cluster
            cluster.calculate_inertia()
            
            # Calcular silhouette score individual (aproximado)
            if len(cluster_list) > 1:
                cluster_embeddings = np.array([img.embedding for img in cluster.images])
                if len(cluster_embeddings) > 0:
                    # Silhouette score simplificado para el cluster
                    cluster.silhouette_score = self._calculate_cluster_silhouette(
                        cluster_embeddings, cluster.centroid, cluster_list, cluster.id
                    )
        
        return cluster_list
    
    def _calculate_cluster_silhouette(self, 
                                    cluster_embeddings: np.ndarray,
                                    cluster_centroid: np.ndarray,
                                    all_clusters: List[Cluster],
                                    cluster_id: int) -> float:
        """
        Calcula un silhouette score aproximado para un cluster individual.
        
        Args:
            cluster_embeddings: Embeddings del cluster
            cluster_centroid: Centroide del cluster
            all_clusters: Lista de todos los clusters
            cluster_id: ID del cluster actual
            
        Returns:
            Silhouette score aproximado del cluster
        """
        if len(cluster_embeddings) <= 1:
            return 0.0
        
        # Distancia intra-cluster promedio
        intra_distances = []
        for embedding in cluster_embeddings:
            distances_to_others = [
                np.linalg.norm(embedding - other) 
                for other in cluster_embeddings 
                if not np.array_equal(embedding, other)
            ]
            if distances_to_others:
                intra_distances.append(np.mean(distances_to_others))
        
        avg_intra_distance = np.mean(intra_distances) if intra_distances else 0.0
        
        # Distancia inter-cluster mínima promedio
        min_inter_distances = []
        for embedding in cluster_embeddings:
            cluster_distances = []
            for other_cluster in all_clusters:
                if other_cluster.id != cluster_id and other_cluster.has_centroid:
                    distance = np.linalg.norm(embedding - other_cluster.centroid)
                    cluster_distances.append(distance)
            
            if cluster_distances:
                min_inter_distances.append(min(cluster_distances))
        
        avg_min_inter_distance = np.mean(min_inter_distances) if min_inter_distances else 0.0
        
        # Silhouette score
        if max(avg_intra_distance, avg_min_inter_distance) == 0:
            return 0.0
        
        return (avg_min_inter_distance - avg_intra_distance) / max(avg_intra_distance, avg_min_inter_distance)
    
    def get_cluster_info(self) -> dict[str, Any]:
        """
        Obtiene información detallada sobre el clustering K-means.
        
        Returns:
            Diccionario con información del clustering
        """
        if not self.fitted:
            return {"status": "not_fitted"}
        
        return {
            "algorithm": "K-means",
            "n_clusters": self.n_clusters,
            "inertia": self.inertia_,
            "silhouette_avg": self.silhouette_avg,
            "calinski_harabasz_score": self.calinski_harabasz_score_,
            "n_iterations": self.kmeans.n_iter_ if self.kmeans else None,
            "parameters": {
                "init": self.init,
                "n_init": self.n_init,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "random_state": self.random_state,
                "min_cluster_size": self.min_cluster_size
            },
            "cluster_centers_shape": self.cluster_centers_.shape if self.cluster_centers_ is not None else None
        }
    
    def find_optimal_k(self, 
                      embeddings: np.ndarray,
                      k_range: range = range(2, 15),
                      method: str = "elbow") -> int:
        """
        Encuentra el número óptimo de clusters usando el método del codo o silhouette.
        
        Args:
            embeddings: Array de embeddings para analizar
            k_range: Rango de valores k a probar
            method: Método a usar ("elbow" para inercia, "silhouette" para silhouette score)
            
        Returns:
            Número óptimo de clusters
        """
        if method == "elbow":
            return self._find_optimal_k_elbow(embeddings, k_range)
        elif method == "silhouette":
            return self._find_optimal_k_silhouette(embeddings, k_range)
        else:
            raise ValueError("Método debe ser 'elbow' o 'silhouette'")
    
    def _find_optimal_k_elbow(self, embeddings: np.ndarray, k_range: range) -> int:
        """Encuentra k óptimo usando el método del codo."""
        inertias = []
        
        for k in k_range:
            if k > embeddings.shape[0]:
                break
            
            kmeans = KMeans(
                n_clusters=k,
                init=self.init,
                n_init=self.n_init,
                random_state=self.random_state
            )
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
        
        # Encontrar el "codo" usando diferencias de segundo orden
        if len(inertias) < 3:
            return k_range.start
        
        # Calcular diferencias de segundo orden
        second_diffs = []
        for i in range(1, len(inertias) - 1):
            second_diff = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_diffs.append(second_diff)
        
        # El codo es donde la segunda diferencia es máxima
        optimal_idx = np.argmax(second_diffs) + 1  # +1 porque empezamos desde índice 1
        optimal_k = list(k_range)[optimal_idx]
        
        return optimal_k
    
    def _find_optimal_k_silhouette(self, embeddings: np.ndarray, k_range: range) -> int:
        """Encuentra k óptimo usando silhouette score."""
        silhouette_scores = []
        
        for k in k_range:
            if k > embeddings.shape[0] or k < 2:
                continue
            
            kmeans = KMeans(
                n_clusters=k,
                init=self.init,
                n_init=self.n_init,
                random_state=self.random_state
            )
            labels = kmeans.fit_predict(embeddings)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(embeddings, labels)
                silhouette_scores.append((k, score))
        
        if not silhouette_scores:
            return k_range.start
        
        # Retornar k con mayor silhouette score
        optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
        return optimal_k
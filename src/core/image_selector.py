"""
ImageSelector: Clase para seleccionar imágenes representativas de clusters.
"""

from typing import List, Optional, Tuple
import numpy as np
from ..models import Image, Cluster


class ImageSelector:
    """
    Selecciona imágenes representativas de cada cluster.
    
    Encuentra las 3 imágenes más cercanas al centroide pero alejadas entre sí
    para maximizar la diversidad dentro de cada cluster.
    """
    
    def __init__(self, 
                 num_representatives: int = 3,
                 diversity_threshold: float = 0.1,
                 max_iterations: int = 100):
        """
        Inicializa el selector de imágenes.
        
        Args:
            num_representatives: Número de imágenes representativas por cluster
            diversity_threshold: Umbral mínimo de diversidad entre imágenes
            max_iterations: Máximo número de iteraciones para la selección
        """
        self.num_representatives = num_representatives
        self.diversity_threshold = diversity_threshold
        self.max_iterations = max_iterations
    
    def select_representative_images(self, cluster: Cluster) -> List[Image]:
        """
        Selecciona imágenes representativas de un cluster.
        
        Args:
            cluster: Cluster del cual seleccionar imágenes
            
        Returns:
            Lista de imágenes representativas
            
        Raises:
            ValueError: Si el cluster no tiene centroide o imágenes suficientes
        """
        if not cluster.has_centroid:
            cluster.calculate_centroid()
        
        if cluster.size < self.num_representatives:
            # Si hay menos imágenes que representativas solicitadas, devolver todas
            return cluster.images.copy()
        
        # Obtener imágenes con embeddings
        images_with_embeddings = [img for img in cluster.images if img.has_embedding]
        
        if len(images_with_embeddings) < self.num_representatives:
            return images_with_embeddings
        
        # Seleccionar usando algoritmo de diversidad
        representatives = self._select_diverse_representatives(
            images_with_embeddings, 
            cluster.centroid
        )
        
        # Actualizar cluster con representativas
        cluster.representative_images = representatives
        
        return representatives
    
    def select_representatives_for_all_clusters(self, clusters: List[Cluster]) -> dict[int, List[Image]]:
        """
        Selecciona representativas para todos los clusters.
        
        Args:
            clusters: Lista de clusters
            
        Returns:
            Diccionario {cluster_id: [images_representativas]}
        """
        representatives_dict = {}
        
        for cluster in clusters:
            try:
                representatives = self.select_representative_images(cluster)
                representatives_dict[cluster.id] = representatives
            except ValueError as e:
                print(f"Error procesando cluster {cluster.id}: {e}")
                representatives_dict[cluster.id] = []
        
        return representatives_dict
    
    def _select_diverse_representatives(self, 
                                     images: List[Image], 
                                     centroid: np.ndarray) -> List[Image]:
        """
        Selecciona imágenes diversas cercanas al centroide.
        
        Algoritmo:
        1. Calcula distancias al centroide
        2. Ordena por distancia (más cercanas primero)
        3. Selecciona iterativamente maximizando diversidad
        
        Args:
            images: Lista de imágenes candidatas
            centroid: Centroide del cluster
            
        Returns:
            Lista de imágenes representativas
        """
        # Calcular distancias al centroide
        distances_to_centroid = []
        for img in images:
            distance = np.linalg.norm(img.embedding - centroid)
            distances_to_centroid.append((img, distance))
        
        # Ordenar por distancia al centroide (más cercanas primero)
        distances_to_centroid.sort(key=lambda x: x[1])
        
        # Selección diversa
        representatives = []
        candidates = [img for img, _ in distances_to_centroid]
        
        # Seleccionar la primera (más cercana al centroide)
        if candidates:
            representatives.append(candidates[0])
            candidates.remove(candidates[0])
        
        # Seleccionar las restantes maximizando diversidad
        while len(representatives) < self.num_representatives and candidates:
            best_candidate = self._find_most_diverse_candidate(
                candidates, 
                representatives
            )
            
            if best_candidate:
                representatives.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                # Si no se encuentra candidato diverso, tomar el siguiente más cercano
                representatives.append(candidates[0])
                candidates.remove(candidates[0])
        
        return representatives
    
    def _find_most_diverse_candidate(self, 
                                   candidates: List[Image], 
                                   selected: List[Image]) -> Optional[Image]:
        """
        Encuentra el candidato más diverso respecto a los ya seleccionados.
        
        Args:
            candidates: Lista de candidatos
            selected: Lista de imágenes ya seleccionadas
            
        Returns:
            Imagen candidata más diversa, o None si ninguna es suficientemente diversa
        """
        best_candidate = None
        max_min_distance = 0.0
        
        for candidate in candidates:
            # Calcular distancia mínima a las ya seleccionadas
            min_distance = float('inf')
            
            for selected_img in selected:
                distance = self._calculate_embedding_distance(
                    candidate.embedding, 
                    selected_img.embedding
                )
                min_distance = min(min_distance, distance)
            
            # Seleccionar candidato con mayor distancia mínima (más diverso)
            if min_distance > max_min_distance and min_distance > self.diversity_threshold:
                max_min_distance = min_distance
                best_candidate = candidate
        
        return best_candidate
    
    def _calculate_embedding_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcula distancia entre dos embeddings.
        
        Args:
            embedding1: Primer embedding
            embedding2: Segundo embedding
            
        Returns:
            Distancia entre embeddings
        """
        # Usar distancia euclidiana normalizada
        return np.linalg.norm(embedding1 - embedding2)
    
    def calculate_diversity_score(self, images: List[Image]) -> float:
        """
        Calcula un score de diversidad para un conjunto de imágenes.
        
        Args:
            images: Lista de imágenes
            
        Returns:
            Score de diversidad (mayor = más diverso)
        """
        if len(images) < 2:
            return 0.0
        
        total_distance = 0.0
        num_pairs = 0
        
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                if images[i].has_embedding and images[j].has_embedding:
                    distance = self._calculate_embedding_distance(
                        images[i].embedding, 
                        images[j].embedding
                    )
                    total_distance += distance
                    num_pairs += 1
        
        return total_distance / num_pairs if num_pairs > 0 else 0.0
    
    def get_selection_statistics(self, cluster: Cluster) -> dict:
        """
        Obtiene estadísticas sobre la selección de representativas.
        
        Args:
            cluster: Cluster analizado
            
        Returns:
            Diccionario con estadísticas
        """
        if not cluster.representative_images:
            return {'error': 'No representative images found'}
        
        # Calcular estadísticas
        diversity_score = self.calculate_diversity_score(cluster.representative_images)
        
        # Distancias al centroide de las representativas
        centroid_distances = []
        if cluster.has_centroid:
            for img in cluster.representative_images:
                if img.has_embedding:
                    distance = np.linalg.norm(img.embedding - cluster.centroid)
                    centroid_distances.append(distance)
        
        return {
            'num_representatives': len(cluster.representative_images),
            'cluster_size': cluster.size,
            'diversity_score': diversity_score,
            'avg_distance_to_centroid': np.mean(centroid_distances) if centroid_distances else 0,
            'max_distance_to_centroid': np.max(centroid_distances) if centroid_distances else 0,
            'min_distance_to_centroid': np.min(centroid_distances) if centroid_distances else 0
        }
    
    def visualize_selection(self, cluster: Cluster) -> dict:
        """
        Genera datos para visualizar la selección de representativas.
        
        Args:
            cluster: Cluster a visualizar
            
        Returns:
            Diccionario con datos para visualización
        """
        # TODO: Implementar generación de datos para visualización
        # Podría incluir coordenadas 2D via PCA/t-SNE, etc.
        
        return {
            'cluster_id': cluster.id,
            'total_images': cluster.size,
            'representative_images': [img.id for img in cluster.representative_images],
            'centroid_position': cluster.centroid.tolist() if cluster.has_centroid else None
        }
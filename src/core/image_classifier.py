"""
ImageClassifier: Clase para clasificar nuevas imágenes usando clusters entrenados.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from ..models import Image, Cluster, Dataset, Label


class ImageClassifier:
    """
    Clasificador de imágenes basado en clusters previamente entrenados.
    
    Utiliza los clusters etiquetados para clasificar nuevas imágenes
    asignándolas al cluster más similar basándose en embeddings.
    """
    
    def __init__(self, 
                 distance_metric: str = "euclidean",
                 confidence_threshold: float = 0.5,
                 use_centroid_distance: bool = True):
        """
        Inicializa el clasificador de imágenes.
        
        Args:
            distance_metric: Métrica de distancia ("euclidean", "cosine", "manhattan")
            confidence_threshold: Umbral mínimo de confianza para clasificación
            use_centroid_distance: Si usar distancia al centroide o promedio de distancias
        """
        self.distance_metric = distance_metric
        self.confidence_threshold = confidence_threshold
        self.use_centroid_distance = use_centroid_distance
        
        self.trained_clusters: List[Cluster] = []
        self.is_trained = False
    
    def train(self, clusters: List[Cluster]) -> None:
        """
        Entrena el clasificador con clusters etiquetados.
        
        Args:
            clusters: Lista de clusters con etiquetas para entrenamiento
            
        Raises:
            ValueError: Si no hay clusters válidos para entrenar
        """
        # Filtrar clusters válidos (con etiquetas y centroide)
        valid_clusters = []
        for cluster in clusters:
            if cluster.is_labeled and cluster.has_centroid:
                valid_clusters.append(cluster)
            elif cluster.is_labeled and not cluster.has_centroid:
                # Calcular centroide si no existe
                try:
                    cluster.calculate_centroid()
                    valid_clusters.append(cluster)
                except ValueError:
                    print(f"No se pudo calcular centroide para cluster {cluster.id}")
        
        if not valid_clusters:
            raise ValueError("No hay clusters válidos para entrenar el clasificador")
        
        self.trained_clusters = valid_clusters
        self.is_trained = True
        
        print(f"Clasificador entrenado con {len(valid_clusters)} clusters")
    
    def predict(self, image: Image) -> Label:
        """
        Predice la etiqueta de una nueva imagen.
        
        Args:
            image: Imagen a clasificar
            
        Returns:
            Label con la predicción y confianza
            
        Raises:
            RuntimeError: Si el clasificador no ha sido entrenado
            ValueError: Si la imagen no tiene embedding
        """
        if not self.is_trained:
            raise RuntimeError("El clasificador debe ser entrenado antes de hacer predicciones")
        
        if not image.has_embedding:
            raise ValueError("La imagen debe tener un embedding para ser clasificada")
        
        # Encontrar el cluster más similar
        best_cluster, confidence = self._find_best_cluster(image.embedding)
        
        if confidence < self.confidence_threshold:
            return Label(
                name="uncertain",
                confidence=confidence,
                source="classifier",
                metadata={
                    'reason': 'low_confidence',
                    'best_cluster_id': best_cluster.id if best_cluster else None,
                    'distance_metric': self.distance_metric
                }
            )
        
        # Crear etiqueta de predicción
        label = Label(
            name=best_cluster.label,
            confidence=confidence,
            source="classifier",
            metadata={
                'assigned_cluster_id': best_cluster.id,
                'distance_metric': self.distance_metric,
                'cluster_size': best_cluster.size
            }
        )
        
        return label
    
    def predict_batch(self, images: List[Image]) -> List[Label]:
        """
        Predice etiquetas para múltiples imágenes.
        
        Args:
            images: Lista de imágenes a clasificar
            
        Returns:
            Lista de etiquetas predichas
        """
        labels = []
        for image in images:
            try:
                label = self.predict(image)
                labels.append(label)
            except ValueError as e:
                # Si una imagen no se puede clasificar, crear etiqueta de error
                error_label = Label(
                    name="error",
                    confidence=0.0,
                    source="classifier",
                    metadata={'error': str(e)}
                )
                labels.append(error_label)
        
        return labels
    
    def classify_dataset(self, dataset: Dataset) -> Dataset:
        """
        Clasifica todas las imágenes de un dataset.
        
        Args:
            dataset: Dataset con imágenes a clasificar
            
        Returns:
            Dataset actualizado con clasificaciones
        """
        for image in dataset.images:
            if image.has_embedding:
                try:
                    label = self.predict(image)
                    image.label = label.name
                except Exception as e:
                    print(f"Error clasificando imagen {image.id}: {e}")
                    image.label = "error"
        
        # Actualizar metadatos del dataset
        dataset.metadata.update({
            'classified': True,
            'classifier_model': 'cluster_based',
            'num_clusters_used': len(self.trained_clusters)
        })
        
        return dataset
    
    def _find_best_cluster(self, embedding: np.ndarray) -> Tuple[Optional[Cluster], float]:
        """
        Encuentra el cluster más similar a un embedding.
        
        Args:
            embedding: Vector de embedding de la imagen
            
        Returns:
            Tupla con (mejor_cluster, confianza)
        """
        if not self.trained_clusters:
            return None, 0.0
        
        best_cluster = None
        best_similarity = -float('inf')  # Para similitud (mayor es mejor)
        
        for cluster in self.trained_clusters:
            if self.use_centroid_distance:
                similarity = self._calculate_similarity(embedding, cluster.centroid)
            else:
                # Usar promedio de distancias a todas las imágenes del cluster
                similarities = []
                for img in cluster.images[:10]:  # Usar máximo 10 imágenes por eficiencia
                    if img.has_embedding:
                        sim = self._calculate_similarity(embedding, img.embedding)
                        similarities.append(sim)
                
                similarity = np.mean(similarities) if similarities else -float('inf')
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster
        
        # Convertir similitud a confianza (0-1)
        confidence = self._similarity_to_confidence(best_similarity)
        
        return best_cluster, confidence
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcula similitud entre dos embeddings.
        
        Args:
            embedding1: Primer embedding
            embedding2: Segundo embedding
            
        Returns:
            Valor de similitud (mayor = más similar)
        """
        if self.distance_metric == "cosine":
            # Similitud coseno
            dot_product = np.dot(embedding1, embedding2)
            norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            return dot_product / norm_product if norm_product != 0 else 0.0
        
        elif self.distance_metric == "euclidean":
            # Convertir distancia euclidiana a similitud (inverso)
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1.0 / (1.0 + distance)  # Similitud inversa
        
        elif self.distance_metric == "manhattan":
            # Convertir distancia Manhattan a similitud
            distance = np.sum(np.abs(embedding1 - embedding2))
            return 1.0 / (1.0 + distance)
        
        else:
            raise ValueError(f"Métrica de distancia no soportada: {self.distance_metric}")
    
    def _similarity_to_confidence(self, similarity: float) -> float:
        """
        Convierte similitud a confianza normalizada.
        
        Args:
            similarity: Valor de similitud
            
        Returns:
            Confianza entre 0 y 1
        """
        if self.distance_metric == "cosine":
            # Similitud coseno ya está entre -1 y 1, normalizar a 0-1
            return (similarity + 1.0) / 2.0
        else:
            # Para métricas basadas en distancia, la similitud ya está normalizada
            return max(0.0, min(1.0, similarity))
    
    def get_cluster_probabilities(self, image: Image) -> Dict[int, float]:
        """
        Obtiene probabilidades de pertenencia a cada cluster.
        
        Args:
            image: Imagen a analizar
            
        Returns:
            Diccionario {cluster_id: probabilidad}
        """
        if not self.is_trained or not image.has_embedding:
            return {}
        
        probabilities = {}
        similarities = []
        
        # Calcular similitudes a todos los clusters
        for cluster in self.trained_clusters:
            if self.use_centroid_distance:
                similarity = self._calculate_similarity(image.embedding, cluster.centroid)
            else:
                # Promedio de similitudes
                cluster_similarities = []
                for img in cluster.images[:5]:  # Usar máximo 5 por eficiencia
                    if img.has_embedding:
                        sim = self._calculate_similarity(image.embedding, img.embedding)
                        cluster_similarities.append(sim)
                similarity = np.mean(cluster_similarities) if cluster_similarities else 0.0
            
            similarities.append(similarity)
            probabilities[cluster.id] = similarity
        
        # Normalizar a probabilidades (softmax simple)
        if similarities:
            max_sim = max(similarities)
            exp_sims = [np.exp(sim - max_sim) for sim in similarities]
            total = sum(exp_sims)
            
            if total > 0:
                for i, cluster in enumerate(self.trained_clusters):
                    probabilities[cluster.id] = exp_sims[i] / total
        
        return probabilities
    
    def evaluate_performance(self, test_images: List[Image], true_labels: List[str]) -> Dict[str, float]:
        """
        Evalúa el rendimiento del clasificador.
        
        Args:
            test_images: Imágenes de prueba
            true_labels: Etiquetas verdaderas correspondientes
            
        Returns:
            Diccionario con métricas de rendimiento
        """
        if len(test_images) != len(true_labels):
            raise ValueError("El número de imágenes y etiquetas debe ser igual")
        
        predictions = self.predict_batch(test_images)
        predicted_labels = [pred.name for pred in predictions]
        
        # Calcular métricas básicas
        correct = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)
        total = len(true_labels)
        accuracy = correct / total if total > 0 else 0.0
        
        # Confianza promedio
        avg_confidence = np.mean([pred.confidence for pred in predictions])
        
        # Distribución de etiquetas predichas
        unique_labels = set(predicted_labels)
        label_distribution = {label: predicted_labels.count(label) for label in unique_labels}
        
        return {
            'accuracy': accuracy,
            'total_samples': total,
            'correct_predictions': correct,
            'average_confidence': avg_confidence,
            'label_distribution': label_distribution,
            'num_clusters_used': len(self.trained_clusters)
        }
    
    def get_classifier_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el clasificador.
        
        Returns:
            Diccionario con información del estado del clasificador
        """
        return {
            'is_trained': self.is_trained,
            'num_clusters': len(self.trained_clusters),
            'distance_metric': self.distance_metric,
            'confidence_threshold': self.confidence_threshold,
            'use_centroid_distance': self.use_centroid_distance,
            'cluster_labels': [cluster.label for cluster in self.trained_clusters]
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo entrenado.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        # TODO: Implementar guardado del modelo clasificador
        pass
    
    def load_model(self, filepath: str) -> None:
        """
        Carga un modelo previamente entrenado.
        
        Args:
            filepath: Ruta del modelo guardado
        """
        # TODO: Implementar carga del modelo clasificador
        pass
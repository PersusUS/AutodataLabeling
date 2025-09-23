"""
Calculador de métricas para evaluación del sistema.
"""

from typing import List, Dict, Any
import numpy as np
from collections import Counter
from ..models import Cluster, Dataset, Label


class MetricsCalculator:
    """Calcula métricas de evaluación para el sistema de etiquetado automático."""
    
    @staticmethod
    def calculate_clustering_metrics(clusters: List[Cluster]) -> Dict[str, float]:
        """
        Calcula métricas de calidad del clustering K-means.
        
        Args:
            clusters: Lista de clusters
            
        Returns:
            Diccionario con métricas de clustering
        """
        if not clusters:
            return {'error': 'No clusters provided'}
        
        # Métricas básicas
        total_images = sum(c.size for c in clusters)
        num_clusters = len(clusters)
        cluster_sizes = [c.size for c in clusters]
        
        # Estadísticas de tamaños
        avg_cluster_size = np.mean(cluster_sizes)
        std_cluster_size = np.std(cluster_sizes)
        min_cluster_size = min(cluster_sizes)
        max_cluster_size = max(cluster_sizes)
        
        # Coeficiente de variación (medida de balance)
        cv_cluster_size = std_cluster_size / avg_cluster_size if avg_cluster_size > 0 else 0
        
        # Métricas específicas de K-means
        total_inertia = sum(c.inertia for c in clusters if c.inertia is not None)
        avg_inertia = total_inertia / num_clusters if num_clusters > 0 else 0
        
        # Silhouette scores individuales
        cluster_silhouette_scores = [c.silhouette_score for c in clusters if c.silhouette_score is not None]
        avg_silhouette_score = np.mean(cluster_silhouette_scores) if cluster_silhouette_scores else 0
        
        # Métricas de cohesión (basadas en centroides)
        intra_cluster_distances = []
        for cluster in clusters:
            if cluster.has_centroid and cluster.size > 1:
                distances = []
                for img in cluster.images:
                    if img.has_embedding:
                        distance = np.linalg.norm(img.embedding - cluster.centroid)
                        distances.append(distance)
                if distances:
                    intra_cluster_distances.extend(distances)
        
        avg_intra_cluster_distance = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
        
        # WCSS (Within-Cluster Sum of Squares) - otra forma de expresar inercia
        total_wcss = total_inertia
        
        return {
            'num_clusters': num_clusters,
            'total_images': total_images,
            'avg_cluster_size': avg_cluster_size,
            'std_cluster_size': std_cluster_size,
            'min_cluster_size': min_cluster_size,
            'max_cluster_size': max_cluster_size,
            'cluster_size_cv': cv_cluster_size,
            'avg_intra_cluster_distance': avg_intra_cluster_distance,
            'cluster_balance_score': 1.0 / (1.0 + cv_cluster_size),  # Menor CV = mejor balance
            # Métricas específicas de K-means
            'total_inertia': total_inertia,
            'avg_inertia_per_cluster': avg_inertia,
            'avg_silhouette_score': avg_silhouette_score,
            'total_wcss': total_wcss,
            'inertia_per_image': total_inertia / total_images if total_images > 0 else 0
        }
    
    @staticmethod
    def calculate_labeling_metrics(clusters: List[Cluster]) -> Dict[str, Any]:
        """
        Calcula métricas de calidad del etiquetado.
        
        Args:
            clusters: Lista de clusters etiquetados
            
        Returns:
            Diccionario con métricas de etiquetado
        """
        if not clusters:
            return {'error': 'No clusters provided'}
        
        # Estadísticas básicas
        labeled_clusters = [c for c in clusters if c.is_labeled]
        unlabeled_clusters = [c for c in clusters if not c.is_labeled]
        
        labeling_coverage = len(labeled_clusters) / len(clusters) if clusters else 0
        
        # Distribución de confianzas
        confidences = [c.confidence for c in labeled_clusters if c.confidence is not None]
        avg_confidence = np.mean(confidences) if confidences else 0
        std_confidence = np.std(confidences) if confidences else 0
        
        # Distribución de etiquetas
        labels = [c.label for c in labeled_clusters if c.label]
        label_counts = Counter(labels)
        unique_labels = len(label_counts)
        
        # Entropía de etiquetas (medida de diversidad)
        if labels:
            total_labels = len(labels)
            label_probs = [count / total_labels for count in label_counts.values()]
            label_entropy = -sum(p * np.log2(p) for p in label_probs if p > 0)
        else:
            label_entropy = 0
        
        return {
            'total_clusters': len(clusters),
            'labeled_clusters': len(labeled_clusters),
            'unlabeled_clusters': len(unlabeled_clusters),
            'labeling_coverage': labeling_coverage,
            'unique_labels': unique_labels,
            'avg_confidence': avg_confidence,
            'std_confidence': std_confidence,
            'label_entropy': label_entropy,
            'label_distribution': dict(label_counts),
            'high_confidence_clusters': sum(1 for c in confidences if c > 0.7),
            'low_confidence_clusters': sum(1 for c in confidences if c < 0.3)
        }
    
    @staticmethod
    def calculate_dataset_statistics(dataset: Dataset) -> Dict[str, Any]:
        """
        Calcula estadísticas generales del dataset.
        
        Args:
            dataset: Dataset a analizar
            
        Returns:
            Diccionario con estadísticas del dataset
        """
        stats = {
            'name': dataset.name,
            'total_images': dataset.size,
            'num_clusters': dataset.num_clusters,
            'labeled_images': dataset.num_labeled_images,
            'clustered_images': dataset.num_clustered_images,
            'labeling_progress': dataset.labeling_progress,
            'clustering_progress': dataset.clustering_progress
        }
        
        # Estadísticas de archivos
        if dataset.images:
            extensions = [img.get_extension() for img in dataset.images]
            extension_counts = Counter(extensions)
            
            # Tamaños de archivos
            file_sizes = [img.get_file_size() for img in dataset.images if img.path.exists()]
            if file_sizes:
                stats.update({
                    'avg_file_size_mb': np.mean(file_sizes) / (1024 * 1024),
                    'total_size_mb': sum(file_sizes) / (1024 * 1024),
                    'extension_distribution': dict(extension_counts)
                })
        
        return stats
    
    @staticmethod
    def calculate_classification_metrics(predictions: List[str], 
                                       true_labels: List[str]) -> Dict[str, float]:
        """
        Calcula métricas de clasificación.
        
        Args:
            predictions: Etiquetas predichas
            true_labels: Etiquetas verdaderas
            
        Returns:
            Diccionario con métricas de clasificación
        """
        if len(predictions) != len(true_labels):
            return {'error': 'Mismatched lengths'}
        
        # Accuracy
        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        accuracy = correct / len(predictions) if predictions else 0
        
        # Precision, Recall, F1 por clase
        unique_labels = set(true_labels + predictions)
        class_metrics = {}
        
        for label in unique_labels:
            tp = sum(1 for pred, true in zip(predictions, true_labels) 
                    if pred == label and true == label)
            fp = sum(1 for pred, true in zip(predictions, true_labels) 
                    if pred == label and true != label)
            fn = sum(1 for pred, true in zip(predictions, true_labels) 
                    if pred != label and true == label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': true_labels.count(label)
            }
        
        # Promedios macro
        precisions = [metrics['precision'] for metrics in class_metrics.values()]
        recalls = [metrics['recall'] for metrics in class_metrics.values()]
        f1_scores = [metrics['f1_score'] for metrics in class_metrics.values()]
        
        return {
            'accuracy': accuracy,
            'macro_precision': np.mean(precisions),
            'macro_recall': np.mean(recalls),
            'macro_f1': np.mean(f1_scores),
            'num_classes': len(unique_labels),
            'class_metrics': class_metrics
        }
    
    @staticmethod
    def calculate_kmeans_specific_metrics(embeddings: np.ndarray, 
                                        cluster_labels: np.ndarray,
                                        cluster_centers: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas específicas para K-means.
        
        Args:
            embeddings: Embeddings de todas las imágenes
            cluster_labels: Etiquetas de cluster asignadas
            cluster_centers: Centros de los clusters
            
        Returns:
            Diccionario con métricas específicas de K-means
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        metrics = {}
        
        if len(np.unique(cluster_labels)) > 1:
            # Silhouette Score
            try:
                metrics['silhouette_score'] = silhouette_score(embeddings, cluster_labels)
            except:
                metrics['silhouette_score'] = 0.0
            
            # Calinski-Harabasz Index (Variance Ratio)
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, cluster_labels)
            except:
                metrics['calinski_harabasz_score'] = 0.0
            
            # Davies-Bouldin Index (menor es mejor)
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, cluster_labels)
            except:
                metrics['davies_bouldin_score'] = float('inf')
        else:
            metrics['silhouette_score'] = 0.0
            metrics['calinski_harabasz_score'] = 0.0
            metrics['davies_bouldin_score'] = float('inf')
        
        # Inercia total (suma de distancias cuadráticas a centroides)
        total_inertia = 0.0
        for i, embedding in enumerate(embeddings):
            cluster_id = cluster_labels[i]
            distance_squared = np.sum((embedding - cluster_centers[cluster_id]) ** 2)
            total_inertia += distance_squared
        
        metrics['total_inertia'] = total_inertia
        metrics['avg_inertia_per_sample'] = total_inertia / len(embeddings) if len(embeddings) > 0 else 0
        
        # Between-cluster sum of squares (BCSS)
        overall_centroid = np.mean(embeddings, axis=0)
        bcss = 0.0
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            cluster_center = cluster_centers[cluster_id]
            bcss += cluster_size * np.sum((cluster_center - overall_centroid) ** 2)
        
        metrics['between_cluster_ss'] = bcss
        
        # Total sum of squares (TSS)
        tss = np.sum((embeddings - overall_centroid) ** 2)
        metrics['total_ss'] = tss
        
        # Ratio BCSS/TSS (mayor es mejor - indica mejor separación)
        metrics['bcss_tss_ratio'] = bcss / tss if tss > 0 else 0
        
        # Ratio WCSS/TSS (menor es mejor - indica menor varianza intra-cluster)
        metrics['wcss_tss_ratio'] = total_inertia / tss if tss > 0 else 0
        
        return metrics
    
    @staticmethod
    def calculate_elbow_method_curve(embeddings: np.ndarray, 
                                   k_range: range = range(1, 11)) -> Dict[int, float]:
        """
        Calcula la curva del método del codo para determinar k óptimo.
        
        Args:
            embeddings: Embeddings para clustering
            k_range: Rango de valores k a probar
            
        Returns:
            Diccionario con k -> inercia
        """
        from sklearn.cluster import KMeans
        
        inertias = {}
        for k in k_range:
            if k > len(embeddings):
                break
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            inertias[k] = kmeans.inertia_
        
        return inertias
    
    @staticmethod
    def calculate_silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Calcula el score de silueta para evaluar clustering.
        
        Args:
            embeddings: Embeddings de las imágenes
            labels: Etiquetas de cluster
            
        Returns:
            Score de silueta (-1 a 1, mayor es mejor)
        """
        if len(np.unique(labels)) < 2:
            return 0.0
        
        # Implementación simplificada del score de silueta
        n_samples = len(embeddings)
        silhouette_scores = []
        
        for i in range(n_samples):
            # Distancia promedio intra-cluster
            same_cluster_mask = labels == labels[i]
            same_cluster_embeddings = embeddings[same_cluster_mask]
            
            if len(same_cluster_embeddings) > 1:
                a = np.mean([np.linalg.norm(embeddings[i] - other) 
                           for other in same_cluster_embeddings if not np.array_equal(embeddings[i], other)])
            else:
                a = 0
            
            # Distancia promedio al cluster más cercano
            different_cluster_distances = []
            for other_label in np.unique(labels):
                if other_label != labels[i]:
                    other_cluster_mask = labels == other_label
                    other_cluster_embeddings = embeddings[other_cluster_mask]
                    avg_distance = np.mean([np.linalg.norm(embeddings[i] - other) 
                                          for other in other_cluster_embeddings])
                    different_cluster_distances.append(avg_distance)
            
            b = min(different_cluster_distances) if different_cluster_distances else 0
            
            # Score de silueta para esta muestra
            s = (b - a) / max(a, b) if max(a, b) > 0 else 0
            silhouette_scores.append(s)
        
        return np.mean(silhouette_scores)
    
    @staticmethod
    def generate_report(dataset: Dataset, clusters: List[Cluster]) -> Dict[str, Any]:
        """
        Genera un reporte completo de métricas.
        
        Args:
            dataset: Dataset analizado
            clusters: Clusters generados
            
        Returns:
            Reporte completo con todas las métricas
        """
        return {
            'dataset_stats': MetricsCalculator.calculate_dataset_statistics(dataset),
            'clustering_metrics': MetricsCalculator.calculate_clustering_metrics(clusters),
            'labeling_metrics': MetricsCalculator.calculate_labeling_metrics(clusters),
            'timestamp': None,  # Se podría añadir timestamp
            'summary': {
                'total_clusters': len(clusters),
                'total_images': dataset.size,
                'labeling_coverage': len([c for c in clusters if c.is_labeled]) / len(clusters) if clusters else 0,
                'avg_cluster_size': sum(c.size for c in clusters) / len(clusters) if clusters else 0
            }
        }
"""
Tests específicos para K-means clustering.
"""

import pytest
import numpy as np
from pathlib import Path
from src.models import Image, Cluster
from src.core import KMeansClusterer
from src.utils import MetricsCalculator


class TestKMeansClustering:
    """Tests para clustering K-means."""
    
    def setup_method(self):
        """Configuración para cada test."""
        # Crear imágenes de prueba con embeddings sintéticos
        self.images = []
        self.embeddings = []
        
        # Cluster 1: Embeddings cercanos a [1, 1, 1, ...]
        for i in range(10):
            embedding = np.random.normal(1.0, 0.1, 768)
            image = Image(
                path=Path(f"test_{i}.jpg"),
                id=f"test_{i}",
                name=f"test_{i}.jpg"
            )
            image.embedding = embedding
            self.images.append(image)
            self.embeddings.append(embedding)
        
        # Cluster 2: Embeddings cercanos a [-1, -1, -1, ...]
        for i in range(10, 20):
            embedding = np.random.normal(-1.0, 0.1, 768)
            image = Image(
                path=Path(f"test_{i}.jpg"),
                id=f"test_{i}",
                name=f"test_{i}.jpg"
            )
            image.embedding = embedding
            self.images.append(image)
            self.embeddings.append(embedding)
        
        # Cluster 3: Embeddings cercanos a [0, 2, 0, ...]
        for i in range(20, 30):
            embedding = np.random.normal(0.0, 0.1, 768)
            embedding[1] = np.random.normal(2.0, 0.1)  # Segunda dimensión diferente
            image = Image(
                path=Path(f"test_{i}.jpg"),
                id=f"test_{i}",
                name=f"test_{i}.jpg"
            )
            image.embedding = embedding
            self.images.append(image)
            self.embeddings.append(embedding)
        
        self.embeddings = np.array(self.embeddings)
    
    def test_kmeans_clusterer_creation(self):
        """Test creación de KMeansClusterer."""
        clusterer = KMeansClusterer(
            n_clusters=3,
            init="k-means++",
            random_state=42
        )
        
        assert clusterer.n_clusters == 3
        assert clusterer.init == "k-means++"
        assert clusterer.random_state == 42
        assert not clusterer.fitted
    
    def test_fit_predict(self):
        """Test ajuste y predicción del modelo."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        
        labels = clusterer.fit_predict(self.embeddings)
        
        assert clusterer.fitted
        assert len(labels) == len(self.embeddings)
        assert len(np.unique(labels)) <= 3
        assert clusterer.inertia_ is not None
        assert clusterer.cluster_centers_ is not None
    
    def test_cluster_images(self):
        """Test clustering de imágenes."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        
        clusters = clusterer.cluster_images(self.images)
        
        assert len(clusters) <= 3  # Puede ser menor si algunos clusters son muy pequeños
        
        total_images_in_clusters = sum(c.size for c in clusters)
        assert total_images_in_clusters <= len(self.images)
        
        # Verificar que cada cluster tiene centroide
        for cluster in clusters:
            assert cluster.has_centroid
            assert cluster.centroid is not None
            assert cluster.inertia is not None
    
    def test_cluster_properties(self):
        """Test propiedades específicas de clusters K-means."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        clusters = clusterer.cluster_images(self.images)
        
        for cluster in clusters:
            # Test cálculo de inercia
            inertia = cluster.calculate_inertia()
            assert inertia >= 0
            assert cluster.inertia == inertia
            
            # Test WCSS
            wcss = cluster.calculate_within_cluster_sum_of_squares()
            assert wcss == inertia
    
    def test_get_cluster_info(self):
        """Test información del clustering."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        
        # Antes del ajuste
        info = clusterer.get_cluster_info()
        assert info["status"] == "not_fitted"
        
        # Después del ajuste
        clusterer.fit_predict(self.embeddings)
        info = clusterer.get_cluster_info()
        
        assert info["algorithm"] == "K-means"
        assert info["n_clusters"] == 3
        assert "inertia" in info
        assert "silhouette_avg" in info
        assert "calinski_harabasz_score" in info
    
    def test_find_optimal_k_elbow(self):
        """Test búsqueda de k óptimo usando método del codo."""
        clusterer = KMeansClusterer(random_state=42)
        
        optimal_k = clusterer.find_optimal_k(
            self.embeddings, 
            k_range=range(2, 6),
            method="elbow"
        )
        
        assert 2 <= optimal_k <= 5
    
    def test_find_optimal_k_silhouette(self):
        """Test búsqueda de k óptimo usando silhouette score."""
        clusterer = KMeansClusterer(random_state=42)
        
        optimal_k = clusterer.find_optimal_k(
            self.embeddings,
            k_range=range(2, 6), 
            method="silhouette"
        )
        
        assert 2 <= optimal_k <= 5
    
    def test_predict_new_embeddings(self):
        """Test predicción en nuevos embeddings."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        
        # Ajustar con embeddings originales
        clusterer.fit_predict(self.embeddings)
        
        # Crear nuevos embeddings similares
        new_embeddings = np.array([
            np.random.normal(1.0, 0.1, 768),  # Debería ir al cluster 1
            np.random.normal(-1.0, 0.1, 768)  # Debería ir al cluster 2
        ])
        
        predictions = clusterer.predict(new_embeddings)
        
        assert len(predictions) == 2
        assert all(0 <= pred < 3 for pred in predictions)


class TestKMeansMetrics:
    """Tests para métricas específicas de K-means."""
    
    def setup_method(self):
        """Configuración para tests de métricas."""
        # Crear datos sintéticos con clusters bien separados
        np.random.seed(42)
        
        # Cluster 1
        cluster1 = np.random.normal([2, 2], 0.5, (20, 2))
        labels1 = np.zeros(20)
        
        # Cluster 2  
        cluster2 = np.random.normal([-2, -2], 0.5, (20, 2))
        labels2 = np.ones(20)
        
        # Cluster 3
        cluster3 = np.random.normal([2, -2], 0.5, (20, 2))
        labels3 = np.ones(20) * 2
        
        self.embeddings = np.vstack([cluster1, cluster2, cluster3])
        self.labels = np.hstack([labels1, labels2, labels3])
        
        # Centroides de los clusters
        self.cluster_centers = np.array([
            [2, 2],
            [-2, -2], 
            [2, -2]
        ])
    
    def test_kmeans_specific_metrics(self):
        """Test métricas específicas de K-means."""
        metrics = MetricsCalculator.calculate_kmeans_specific_metrics(
            self.embeddings, 
            self.labels.astype(int),
            self.cluster_centers
        )
        
        # Verificar que todas las métricas están presentes
        expected_metrics = [
            'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
            'total_inertia', 'avg_inertia_per_sample', 'between_cluster_ss',
            'total_ss', 'bcss_tss_ratio', 'wcss_tss_ratio'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # Verificar rangos esperados
        assert -1 <= metrics['silhouette_score'] <= 1
        assert metrics['calinski_harabasz_score'] >= 0
        assert metrics['davies_bouldin_score'] >= 0
        assert metrics['total_inertia'] >= 0
        assert 0 <= metrics['bcss_tss_ratio'] <= 1
        assert 0 <= metrics['wcss_tss_ratio'] <= 1
    
    def test_elbow_method_curve(self):
        """Test curva del método del codo."""
        curve = MetricsCalculator.calculate_elbow_method_curve(
            self.embeddings,
            k_range=range(1, 6)
        )
        
        assert len(curve) == 5
        
        # La inercia debe decrecer con k
        inertias = list(curve.values())
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i + 1]
    
    def test_clustering_metrics_with_kmeans(self):
        """Test métricas de clustering con clusters K-means."""
        # Crear clusters con propiedades K-means
        clusters = []
        for i in range(3):
            cluster = Cluster(id=i)
            cluster.centroid = self.cluster_centers[i]
            cluster.inertia = 10.0 + i * 5  # Inercias simuladas
            cluster.silhouette_score = 0.7 - i * 0.1  # Silhouette scores simulados
            
            # Añadir algunas imágenes mock
            for j in range(20):
                cluster.images.append(f"mock_image_{i}_{j}")
            
            clusters.append(cluster)
        
        metrics = MetricsCalculator.calculate_clustering_metrics(clusters)
        
        # Verificar métricas específicas de K-means
        assert 'total_inertia' in metrics
        assert 'avg_inertia_per_cluster' in metrics
        assert 'avg_silhouette_score' in metrics
        
        assert metrics['total_inertia'] == 35.0  # 10 + 15 + 20
        assert metrics['avg_inertia_per_cluster'] == 35.0 / 3
        assert abs(metrics['avg_silhouette_score'] - 0.6) < 0.01  # (0.7 + 0.6 + 0.5) / 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
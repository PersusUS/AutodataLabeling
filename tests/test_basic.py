"""
Tests básicos para el sistema AutoData Labeling.
"""

import pytest
from pathlib import Path
import numpy as np
from src.models import Image, Embedding, Cluster, Label, Dataset
from src.core import DatasetLoader, EmbeddingGenerator, HierarchicalClusterer


class TestModels:
    """Tests para los modelos de datos."""
    
    def test_image_creation(self):
        """Test creación de imagen."""
        image_path = Path("test_image.jpg")
        image = Image(
            path=image_path,
            id="test_id",
            name="test_image.jpg"
        )
        
        assert image.path == image_path
        assert image.id == "test_id"
        assert image.name == "test_image.jpg"
        assert not image.has_embedding
        assert not image.is_clustered
        assert not image.is_labeled
    
    def test_embedding_creation(self):
        """Test creación de embedding."""
        vector = np.random.random(768)
        embedding = Embedding(
            vector=vector,
            image_id="test_id",
            model_name="dinov2"
        )
        
        assert embedding.image_id == "test_id"
        assert embedding.model_name == "dinov2"
        assert embedding.dimensions == 768
        assert np.array_equal(embedding.vector, vector)
    
    def test_cluster_creation(self):
        """Test creación de cluster."""
        cluster = Cluster(id=1)
        
        assert cluster.id == 1
        assert cluster.size == 0
        assert not cluster.is_labeled
        assert not cluster.has_centroid
    
    def test_label_creation(self):
        """Test creación de etiqueta."""
        label = Label(
            name="test_label",
            confidence=0.8,
            source="clip"
        )
        
        assert label.name == "test_label"
        assert label.confidence == 0.8
        assert label.source == "clip"
        assert label.is_confident
        assert label.is_automated
    
    def test_dataset_creation(self):
        """Test creación de dataset."""
        dataset_path = Path("test_dataset")
        dataset = Dataset(
            name="test_dataset",
            path=dataset_path
        )
        
        assert dataset.name == "test_dataset"
        assert dataset.path == dataset_path
        assert dataset.size == 0
        assert dataset.num_clusters == 0


class TestCore:
    """Tests para componentes principales."""
    
    def test_dataset_loader_init(self):
        """Test inicialización del DatasetLoader."""
        loader = DatasetLoader()
        
        assert loader.supported_extensions
        assert '.jpg' in loader.supported_extensions
        assert '.png' in loader.supported_extensions
    
    def test_embedding_generator_init(self):
        """Test inicialización del EmbeddingGenerator."""
        generator = EmbeddingGenerator(
            model_size="small",
            device="cpu"
        )
        
        assert generator.model_size == "small"
        assert generator.device == "cpu"
        assert not generator._is_initialized
    
    def test_hierarchical_clusterer_init(self):
        """Test inicialización del HierarchicalClusterer."""
        clusterer = HierarchicalClusterer(
            linkage="ward",
            min_cluster_size=5
        )
        
        assert clusterer.linkage == "ward"
        assert clusterer.min_cluster_size == 5
        assert not clusterer._is_fitted


class TestUtils:
    """Tests para utilidades."""
    
    def test_metrics_calculation(self):
        """Test cálculo de métricas básicas."""
        from src.utils import MetricsCalculator
        
        # Crear clusters de prueba
        clusters = [
            Cluster(id=1, metadata={'size': 10}),
            Cluster(id=2, metadata={'size': 15}),
            Cluster(id=3, metadata={'size': 8})
        ]
        
        # Simular tamaños
        for i, cluster in enumerate(clusters):
            cluster.images = [None] * [10, 15, 8][i]  # Mock images
        
        metrics = MetricsCalculator.calculate_clustering_metrics(clusters)
        
        assert 'num_clusters' in metrics
        assert 'total_images' in metrics
        assert metrics['num_clusters'] == 3
        assert metrics['total_images'] == 33


if __name__ == "__main__":
    # Ejecutar tests básicos
    pytest.main([__file__, "-v"])
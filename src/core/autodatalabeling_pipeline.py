"""
AutoDataLabelingPipeline: Clase principal que orquesta todo el proceso de etiquetado automático.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import time
from ..models import Dataset, Image, Cluster
from .dataset_loader import DatasetLoader
from .embedding_generator import EmbeddingGenerator
from .hierarchical_clusterer import HierarchicalClusterer
from .image_selector import ImageSelector
from .clip_labeler import CLIPLabeler
from .image_classifier import ImageClassifier


class AutoDataLabelingPipeline:
    """
    Pipeline principal para el etiquetado automático de imágenes.
    
    Integra todos los componentes del sistema: carga de datos, generación de embeddings,
    clustering, selección de representativas, etiquetado con CLIP y clasificación.
    """
    
    def __init__(self,
                 embedding_model_size: str = "base",
                 num_representatives: int = 3,
                 clip_model: str = "ViT-B/32",
                 clustering_linkage: str = "ward",
                 device: str = "auto"):
        """
        Inicializa el pipeline de etiquetado automático.
        
        Args:
            embedding_model_size: Tamaño del modelo DINOv2 ("small", "base", "large", "giant")
            num_representatives: Número de imágenes representativas por cluster
            clip_model: Modelo CLIP a usar para etiquetado
            clustering_linkage: Método de enlace para clustering jerárquico
            device: Dispositivo a usar ("cpu", "cuda", "auto")
        """
        # Inicializar componentes
        self.dataset_loader = DatasetLoader()
        self.embedding_generator = EmbeddingGenerator(
            model_size=embedding_model_size,
            device=device
        )
        self.clusterer = HierarchicalClusterer(
            linkage=clustering_linkage,
            min_cluster_size=3
        )
        self.image_selector = ImageSelector(
            num_representatives=num_representatives
        )
        self.clip_labeler = CLIPLabeler(
            model_name=clip_model,
            device=device
        )
        self.classifier = ImageClassifier()
        
        # Estado del pipeline
        self.current_dataset: Optional[Dataset] = None
        self.trained_clusters: List[Cluster] = []
        self.is_trained = False
        
        # Estadísticas de ejecución
        self.execution_stats = {
            'total_time': 0.0,
            'embedding_time': 0.0,
            'clustering_time': 0.0,
            'labeling_time': 0.0,
            'images_processed': 0,
            'clusters_created': 0
        }
    
    def train_pipeline(self, dataset_path: Path, dataset_name: Optional[str] = None) -> Dataset:
        """
        Entrena el pipeline completo con un dataset de imágenes.
        
        Args:
            dataset_path: Ruta al directorio con imágenes de entrenamiento
            dataset_name: Nombre del dataset (opcional)
            
        Returns:
            Dataset procesado con clusters etiquetados
            
        Raises:
            FileNotFoundError: Si el directorio no existe
            ValueError: Si no hay imágenes válidas
        """
        start_time = time.time()
        print(f"Iniciando entrenamiento del pipeline con dataset: {dataset_path}")
        
        try:
            # Paso 1: Cargar dataset
            print("📁 Cargando dataset...")
            self.current_dataset = self.dataset_loader.load_dataset_recursive(
                dataset_path, dataset_name
            )
            print(f"✅ Dataset cargado: {len(self.current_dataset.images)} imágenes")
            
            # Paso 2: Generar embeddings
            print("🧠 Generando embeddings con DINOv2...")
            embedding_start = time.time()
            self.current_dataset = self.embedding_generator.generate_dataset_embeddings(
                self.current_dataset
            )
            self.execution_stats['embedding_time'] = time.time() - embedding_start
            print(f"✅ Embeddings generados para {len(self.current_dataset.images)} imágenes")
            
            # Paso 3: Clustering jerárquico
            print("🎯 Aplicando clustering jerárquico...")
            clustering_start = time.time()
            self.trained_clusters = self.clusterer.create_clusters_from_dataset(
                self.current_dataset
            )
            self.execution_stats['clustering_time'] = time.time() - clustering_start
            print(f"✅ Creados {len(self.trained_clusters)} clusters")
            
            # Paso 4: Seleccionar imágenes representativas
            print("🎭 Seleccionando imágenes representativas...")
            for cluster in self.trained_clusters:
                self.image_selector.select_representative_images(cluster)
            print(f"✅ Representativas seleccionadas para {len(self.trained_clusters)} clusters")
            
            # Paso 5: Etiquetar clusters con CLIP
            print("🏷️ Etiquetando clusters con CLIP...")
            labeling_start = time.time()
            cluster_labels = self.clip_labeler.label_clusters(self.trained_clusters)
            self.execution_stats['labeling_time'] = time.time() - labeling_start
            print(f"✅ Etiquetados {len(cluster_labels)} clusters")
            
            # Paso 6: Entrenar clasificador
            print("🤖 Entrenando clasificador...")
            self.classifier.train(self.trained_clusters)
            self.is_trained = True
            print("✅ Clasificador entrenado")
            
            # Actualizar estadísticas
            total_time = time.time() - start_time
            self.execution_stats.update({
                'total_time': total_time,
                'images_processed': len(self.current_dataset.images),
                'clusters_created': len(self.trained_clusters)
            })
            
            print(f"🎉 Pipeline entrenado exitosamente en {total_time:.2f} segundos")
            return self.current_dataset
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
            raise
    
    def classify_new_images(self, images_path: Path) -> List[Dict[str, Any]]:
        """
        Clasifica nuevas imágenes usando el pipeline entrenado.
        
        Args:
            images_path: Ruta con nuevas imágenes a clasificar
            
        Returns:
            Lista de diccionarios con resultados de clasificación
            
        Raises:
            RuntimeError: Si el pipeline no ha sido entrenado
        """
        if not self.is_trained:
            raise RuntimeError("El pipeline debe ser entrenado antes de clasificar imágenes")
        
        print(f"🔍 Clasificando nuevas imágenes desde: {images_path}")
        
        # Cargar nuevas imágenes
        new_dataset = self.dataset_loader.load_dataset_recursive(images_path, "new_images")
        print(f"📁 Cargadas {len(new_dataset.images)} nuevas imágenes")
        
        # Generar embeddings para las nuevas imágenes
        print("🧠 Generando embeddings para nuevas imágenes...")
        new_dataset = self.embedding_generator.generate_dataset_embeddings(new_dataset)
        
        # Clasificar cada imagen
        print("🎯 Clasificando imágenes...")
        results = []
        for image in new_dataset.images:
            try:
                label = self.classifier.predict(image)
                probabilities = self.classifier.get_cluster_probabilities(image)
                
                result = {
                    'image_path': str(image.path),
                    'image_name': image.name,
                    'predicted_label': label.name,
                    'confidence': label.confidence,
                    'source': label.source,
                    'cluster_probabilities': probabilities,
                    'metadata': label.metadata
                }
                results.append(result)
                
            except Exception as e:
                print(f"⚠️ Error clasificando {image.name}: {e}")
                results.append({
                    'image_path': str(image.path),
                    'image_name': image.name,
                    'predicted_label': 'error',
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        print(f"✅ Clasificadas {len(results)} imágenes")
        return results
    
    def classify_single_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Clasifica una sola imagen.
        
        Args:
            image_path: Ruta de la imagen a clasificar
            
        Returns:
            Diccionario con resultado de clasificación
        """
        if not self.is_trained:
            raise RuntimeError("El pipeline debe ser entrenado antes de clasificar")
        
        # Crear imagen temporal
        image = Image(
            path=image_path,
            id="temp_image",
            name=image_path.name
        )
        
        # Generar embedding
        embedding_obj = self.embedding_generator.generate_embedding(image)
        image.embedding = embedding_obj.vector
        
        # Clasificar
        label = self.classifier.predict(image)
        probabilities = self.classifier.get_cluster_probabilities(image)
        
        return {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'predicted_label': label.name,
            'confidence': label.confidence,
            'cluster_probabilities': probabilities,
            'metadata': label.metadata
        }
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen del estado del pipeline.
        
        Returns:
            Diccionario con información del pipeline
        """
        summary = {
            'is_trained': self.is_trained,
            'execution_stats': self.execution_stats.copy(),
            'components': {
                'embedding_generator': self.embedding_generator.get_model_info(),
                'clusterer': self.clusterer.get_cluster_info(),
                'clip_labeler': self.clip_labeler.get_model_info(),
                'classifier': self.classifier.get_classifier_info()
            }
        }
        
        if self.current_dataset:
            summary['dataset_info'] = {
                'name': self.current_dataset.name,
                'total_images': self.current_dataset.size,
                'labeled_images': self.current_dataset.num_labeled_images,
                'num_clusters': self.current_dataset.num_clusters
            }
        
        if self.trained_clusters:
            summary['cluster_info'] = {
                'total_clusters': len(self.trained_clusters),
                'cluster_labels': [(c.id, c.label, c.size) for c in self.trained_clusters],
                'avg_cluster_size': sum(c.size for c in self.trained_clusters) / len(self.trained_clusters)
            }
        
        return summary
    
    def save_trained_model(self, model_path: Path) -> None:
        """
        Guarda el modelo entrenado.
        
        Args:
            model_path: Ruta donde guardar el modelo
        """
        if not self.is_trained:
            raise RuntimeError("No hay modelo entrenado para guardar")
        
        # TODO: Implementar guardado del modelo completo
        # Incluiría clusters, embeddings, configuraciones, etc.
        print(f"💾 Guardando modelo en: {model_path}")
    
    def load_trained_model(self, model_path: Path) -> None:
        """
        Carga un modelo previamente entrenado.
        
        Args:
            model_path: Ruta del modelo guardado
        """
        # TODO: Implementar carga del modelo completo
        print(f"📂 Cargando modelo desde: {model_path}")
    
    def visualize_clusters(self) -> Dict[str, Any]:
        """
        Genera datos para visualizar los clusters.
        
        Returns:
            Diccionario con datos de visualización
        """
        if not self.trained_clusters:
            return {'error': 'No hay clusters entrenados'}
        
        # TODO: Implementar generación de datos para visualización
        # Podría incluir reducción dimensional (PCA, t-SNE) para graficar
        
        visualization_data = {
            'clusters': [],
            'total_clusters': len(self.trained_clusters),
            'total_images': sum(c.size for c in self.trained_clusters)
        }
        
        for cluster in self.trained_clusters:
            cluster_data = {
                'id': cluster.id,
                'label': cluster.label,
                'size': cluster.size,
                'confidence': cluster.confidence,
                'representative_images': [str(img.path) for img in cluster.representative_images]
            }
            visualization_data['clusters'].append(cluster_data)
        
        return visualization_data
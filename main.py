"""
Ejemplo de uso básico del sistema AutoData Labeling.
"""

from pathlib import Path
from src.core import AutoDataLabelingPipeline
from src.utils import MetricsCalculator

def main():
    """Ejemplo principal de uso del sistema."""
    
    # Configurar rutas de datos
    training_data_path = Path("data/training_images")
    new_images_path = Path("data/new_images")
    
    print("🚀 Iniciando AutoData Labeling Pipeline")
    
    # 1. Crear y configurar pipeline
    pipeline = AutoDataLabelingPipeline(
        embedding_model_size="base",
        num_representatives=3,
        clip_model="ViT-B/32",
        clustering_linkage="ward",
        device="auto"
    )
    
    try:
        # 2. Entrenar el sistema
        print("\n📚 Fase de Entrenamiento")
        trained_dataset = pipeline.train_pipeline(
            dataset_path=training_data_path,
            dataset_name="training_set"
        )
        
        # 3. Mostrar resumen del entrenamiento
        summary = pipeline.get_pipeline_summary()
        print(f"\n✅ Entrenamiento completado:")
        print(f"   - Imágenes procesadas: {summary['dataset_info']['total_images']}")
        print(f"   - Clusters generados: {summary['dataset_info']['num_clusters']}")
        print(f"   - Tiempo total: {summary['execution_stats']['total_time']:.2f}s")
        
        # 4. Generar reporte de métricas
        print("\n📊 Generando reporte de métricas...")
        report = MetricsCalculator.generate_report(
            trained_dataset, 
            pipeline.trained_clusters
        )
        
        print(f"   - Cobertura de etiquetado: {report['labeling_metrics']['labeling_coverage']:.2%}")
        print(f"   - Confianza promedio: {report['labeling_metrics']['avg_confidence']:.3f}")
        print(f"   - Balance de clusters: {report['clustering_metrics']['cluster_balance_score']:.3f}")
        
        # 5. Mostrar etiquetas encontradas
        print(f"\n🏷️ Etiquetas generadas:")
        for cluster in pipeline.trained_clusters[:5]:  # Mostrar primeras 5
            print(f"   - Cluster {cluster.id}: '{cluster.label}' ({cluster.size} imágenes, confianza: {cluster.confidence:.3f})")
        
        # 6. Clasificar nuevas imágenes (si existe el directorio)
        if new_images_path.exists():
            print(f"\n🔍 Clasificando nuevas imágenes desde: {new_images_path}")
            
            results = pipeline.classify_new_images(new_images_path)
            
            print(f"   Resultados de clasificación:")
            for result in results[:10]:  # Mostrar primeros 10
                print(f"   - {result['image_name']}: {result['predicted_label']} "
                      f"(confianza: {result['confidence']:.3f})")
        else:
            print(f"\n⚠️ Directorio de nuevas imágenes no encontrado: {new_images_path}")
            print("   Creando clasificación de ejemplo...")
            
            # Ejemplo con imagen individual (si existe)
            example_images = list(training_data_path.rglob("*.jpg"))[:3] if training_data_path.exists() else []
            
            for img_path in example_images:
                result = pipeline.classify_single_image(img_path)
                print(f"   - {result['image_name']}: {result['predicted_label']} "
                      f"(confianza: {result['confidence']:.3f})")
        
        # 7. Generar datos de visualización
        print(f"\n🎨 Generando datos de visualización...")
        viz_data = pipeline.visualize_clusters()
        print(f"   - Datos preparados para {viz_data['total_clusters']} clusters")
        print(f"   - Total de imágenes: {viz_data['total_images']}")
        
        print(f"\n🎉 Proceso completado exitosamente!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Directorio no encontrado - {e}")
        print("💡 Asegúrate de tener un directorio 'data/training_images' con imágenes")
        
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {e}")
        print("💡 Verifica la configuración y que las dependencias estén instaladas")


def create_example_structure():
    """Crea estructura de directorios de ejemplo."""
    
    directories = [
        "data/training_images",
        "data/new_images", 
        "results",
        "cache",
        "models"
    ]
    
    print("📁 Creando estructura de directorios de ejemplo...")
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")
    
    # Crear archivo README en data
    data_readme = Path("data/README.md")
    if not data_readme.exists():
        data_readme.write_text("""# Datos para AutoData Labeling

## Estructura

- `training_images/`: Coloca aquí las imágenes para entrenar el sistema
- `new_images/`: Coloca aquí las imágenes que quieres clasificar

## Formatos soportados

- JPG/JPEG
- PNG  
- BMP
- TIFF/TIF
- WebP

## Recomendaciones

- Usa al menos 100-500 imágenes para entrenamiento
- Las imágenes deben ser representativas de las categorías que quieres detectar
- Evita imágenes muy pequeñas (menos de 224x224 píxeles)
""")
    
    print("📝 Estructura creada. Coloca tus imágenes en 'data/training_images/'")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        create_example_structure()
    else:
        main()
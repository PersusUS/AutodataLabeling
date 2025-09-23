# AutoData Labeling - Sistema de Etiquetado Automático de Imágenes

## Descripción

AutoData Labeling es un sistema inteligente para el etiquetado automático de imágenes que utiliza técnicas avanzadas de aprendizaje automático. El sistema procesa datasets de imágenes sin etiquetar y genera automáticamente etiquetas descriptivas usando la siguiente pipeline:

1. **Generación de Embeddings**: Convierte imágenes en vectores de características usando DINOv2
2. **Clustering Jerárquico**: Agrupa imágenes similares usando clustering aglomerativo  
3. **Selección de Representativas**: Identifica las imágenes más representativas de cada cluster
4. **Etiquetado con CLIP**: Genera etiquetas descriptivas usando el modelo CLIP
5. **Clasificación**: Clasifica nuevas imágenes usando los clusters entrenados

## Arquitectura del Sistema

### Modelos de Datos (`src/models/`)
- **Image**: Representa imágenes individuales con metadatos y embeddings
- **Embedding**: Vectores de características generados por DINOv2
- **Cluster**: Grupos de imágenes similares con centroides y etiquetas
- **Label**: Etiquetas con confianza y alternativas
- **Dataset**: Colecciones de imágenes con estadísticas

### Componentes Principales (`src/core/`)
- **DatasetLoader**: Carga y valida datasets de imágenes
- **EmbeddingGenerator**: Genera embeddings usando DINOv2
- **HierarchicalClusterer**: Implementa clustering jerárquico aglomerativo
- **ImageSelector**: Selecciona imágenes representativas de clusters
- **CLIPLabeler**: Genera etiquetas usando CLIP
- **ImageClassifier**: Clasifica nuevas imágenes
- **AutoDataLabelingPipeline**: Orquesta todo el proceso

### Utilidades (`src/utils/`)
- **ImageUtils**: Procesamiento y validación de imágenes
- **MetricsCalculator**: Cálculo de métricas de evaluación
- **Visualizer**: Herramientas de visualización (por implementar)
- **FileUtils**: Utilidades de archivos (por implementar)

## Flujo de Trabajo

### 1. Entrenamiento del Sistema
```python
from src.core import AutoDataLabelingPipeline
from pathlib import Path

# Crear pipeline
pipeline = AutoDataLabelingPipeline(
    embedding_model_size="base",
    num_representatives=3,
    clip_model="ViT-B/32"
)

# Entrenar con dataset
dataset_path = Path("path/to/training/images")
trained_dataset = pipeline.train_pipeline(dataset_path)
```

### 2. Clasificación de Nuevas Imágenes
```python
# Clasificar nuevas imágenes
new_images_path = Path("path/to/new/images")
results = pipeline.classify_new_images(new_images_path)

# Clasificar imagen individual
single_result = pipeline.classify_single_image(Path("path/to/image.jpg"))
```

## Requisitos del Sistema

### Dependencias Principales
- Python 3.8+
- PyTorch (para DINOv2)
- Transformers (para CLIP)
- scikit-learn (para clustering)
- Pillow (procesamiento de imágenes)
- NumPy
- Pandas (análisis de datos)

### Hardware Recomendado
- GPU con CUDA (recomendado para DINOv2 y CLIP)
- 16GB+ RAM para datasets grandes
- Espacio suficiente para almacenar embeddings

## Instalación

```bash
# Clonar repositorio
git clone <repository-url>
cd AutodataLabeling

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\\Scripts\\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Configuración

El sistema se puede configurar mediante:

1. **Archivo de configuración**: `config/config.yaml`  
2. **Variables de entorno**
3. **Parámetros de inicialización**

## Estructura de Directorios

```
AutodataLabeling/
├── src/
│   ├── __init__.py
│   ├── models/          # Modelos de datos
│   ├── core/           # Lógica principal
│   └── utils/          # Utilidades
├── config/             # Configuraciones
├── data/              # Datos de ejemplo
├── tests/             # Tests unitarios
├── logs/              # Logs del sistema
├── requirements.txt   # Dependencias
└── README.md         # Este archivo
```

## Ejemplos de Uso

### Ejemplo Básico
```python
from src import AutoDataLabelingPipeline
from pathlib import Path

# Configurar pipeline
pipeline = AutoDataLabelingPipeline()

# Entrenar
pipeline.train_pipeline(Path("training_data"))

# Clasificar
results = pipeline.classify_new_images(Path("new_images"))
for result in results:
    print(f"{result['image_name']}: {result['predicted_label']} ({result['confidence']:.2f})")
```

### Análisis de Métricas
```python
from src.utils import MetricsCalculator

# Generar reporte completo
report = MetricsCalculator.generate_report(dataset, clusters)
print(f"Precisión promedio: {report['clustering_metrics']['cluster_balance_score']:.3f}")
```

## Extensibilidad

El sistema está diseñado para ser extensible:

1. **Nuevos modelos de embedding**: Implementar `BaseEmbeddingGenerator`
2. **Algoritmos de clustering**: Implementar `BaseClusterer`  
3. **Métodos de etiquetado**: Implementar `BaseLabelGenerator`
4. **Métricas personalizadas**: Extender `MetricsCalculator`

## Limitaciones Actuales

- Los modelos DINOv2 y CLIP requieren implementación real
- No incluye interfaz gráfica de usuario
- Visualizaciones por implementar
- No incluye persistencia de modelos entrenados

## Contribución

Para contribuir al proyecto:

1. Fork el repositorio
2. Crear branch para nueva funcionalidad
3. Implementar cambios con tests
4. Enviar pull request

## Licencia

[Especificar licencia]

## Contacto

[Información de contacto del equipo]

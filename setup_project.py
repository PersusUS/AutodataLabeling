"""
Script de configuración inicial para AutoData Labeling.
"""

import os
import sys
from pathlib import Path


def create_directory_structure():
    """Crea la estructura de directorios necesaria."""
    
    directories = [
        "data/training_images",
        "data/new_images",
        "data/examples",
        "results/reports",
        "results/visualizations", 
        "cache/embeddings",
        "cache/models",
        "models/trained",
        "logs"
    ]
    
    print("📁 Creando estructura de directorios...")
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
        
        # Crear archivo .gitkeep para directorios vacíos
        gitkeep = dir_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()


def create_env_file():
    """Crea archivo de variables de entorno de ejemplo."""
    
    env_content = """# Variables de entorno para AutoData Labeling

# Configuración de dispositivos
DEVICE=auto
CUDA_VISIBLE_DEVICES=0

# Configuración de modelos
DINOV2_MODEL_SIZE=base
CLIP_MODEL_NAME=ViT-B/32

# Configuración de datos
DATA_DIR=./data
CACHE_DIR=./cache
RESULTS_DIR=./results

# Configuración de logging
LOG_LEVEL=INFO
LOG_FILE=./logs/autodatalabeling.log

# Configuración de performance
NUM_WORKERS=4
BATCH_SIZE=32
USE_MIXED_PRECISION=false

# Configuración opcional de Hugging Face
# HF_HOME=./cache/huggingface
# TRANSFORMERS_CACHE=./cache/transformers
"""
    
    env_file = Path(".env.example")
    env_file.write_text(env_content)
    print("📝 Archivo .env.example creado")


def create_gitignore():
    """Crea archivo .gitignore."""
    
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/

# AutoData Labeling specific
cache/
models/trained/
results/
logs/
data/training_images/*
data/new_images/*
!data/training_images/.gitkeep
!data/new_images/.gitkeep

# Large files
*.pkl
*.h5
*.hdf5
*.pt
*.pth
*.onnx

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    gitignore_file = Path(".gitignore")
    gitignore_file.write_text(gitignore_content)
    print("📝 Archivo .gitignore creado")


def create_setup_py():
    """Crea archivo setup.py para instalación."""
    
    setup_content = '''"""
Setup script para AutoData Labeling.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Leer README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Leer requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split("\\n")
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="autodatalabeling",
    version="1.0.0",
    author="AutoData Labeling Team",
    author_email="team@autodatalabeling.com",
    description="Sistema automático de etiquetado de imágenes usando DINOv2 y CLIP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/AutodataLabeling",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "autodatalabeling=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
)
'''
    
    setup_file = Path("setup.py")
    setup_file.write_text(setup_content)
    print("📝 Archivo setup.py creado")


def create_makefile():
    """Crea Makefile para comandos comunes."""
    
    makefile_content = """# Makefile para AutoData Labeling

.PHONY: help install install-dev test lint format clean setup run

help:
	@echo "Comandos disponibles:"
	@echo "  install     - Instalar dependencias"
	@echo "  install-dev - Instalar dependencias de desarrollo"
	@echo "  test        - Ejecutar tests"
	@echo "  lint        - Verificar código con flake8"
	@echo "  format      - Formatear código con black"
	@echo "  clean       - Limpiar archivos temporales"
	@echo "  setup       - Configuración inicial completa"
	@echo "  run         - Ejecutar ejemplo principal"

install:
	pip install -r requirements.txt

install-dev: install
	pip install pytest pytest-cov black flake8 mypy

test:
	python -m pytest tests/ -v --cov=src

lint:
	flake8 src/ tests/ --max-line-length=100

format:
	black src/ tests/ --line-length=100

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

setup:
	python setup_project.py
	@echo "✅ Configuración inicial completada"
	@echo "💡 Siguiente paso: coloca imágenes en data/training_images/"

run:
	python main.py

run-setup:
	python main.py --setup
"""
    
    makefile = Path("Makefile")
    makefile.write_text(makefile_content)
    print("📝 Makefile creado")


def print_next_steps():
    """Imprime los siguientes pasos."""
    
    print("\n" + "="*60)
    print("🎉 CONFIGURACIÓN INICIAL COMPLETADA")
    print("="*60)
    
    print("\n📋 SIGUIENTES PASOS:")
    print("\n1. 📦 Instalar dependencias:")
    print("   pip install -r requirements.txt")
    
    print("\n2. 📁 Añadir imágenes de entrenamiento:")
    print("   - Coloca al menos 100-500 imágenes en: data/training_images/")
    print("   - Formatos soportados: JPG, PNG, BMP, TIFF, WebP")
    
    print("\n3. 🚀 Ejecutar el sistema:")
    print("   python main.py")
    
    print("\n4. 📊 Revisar resultados:")
    print("   - Logs: logs/")
    print("   - Resultados: results/")
    print("   - Cache: cache/")
    
    print("\n💡 COMANDOS ÚTILES:")
    print("   python main.py --setup     # Crear estructura de ejemplo")
    print("   python -m pytest tests/    # Ejecutar tests")
    print("   make help                  # Ver todos los comandos")
    
    print("\n📚 DOCUMENTACIÓN:")
    print("   - README.md: Documentación principal")
    print("   - config/config.yaml: Configuración del sistema")
    print("   - .env.example: Variables de entorno")
    
    print("\n" + "="*60)


def main():
    """Función principal de configuración."""
    
    print("🔧 Configurando AutoData Labeling Project...")
    print()
    
    try:
        create_directory_structure()
        create_env_file()
        create_gitignore()
        create_setup_py()
        create_makefile()
        
        print_next_steps()
        
    except Exception as e:
        print(f"❌ Error durante la configuración: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
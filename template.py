from pathlib import Path
import logging


# Logging Configuration

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s:",
)


# Project Configuration

project_name = "signLanguage"


# Project File Structure

list_of_files = [
    "data/.gitkeep",

    f"{project_name}/__init__.py",

    # Components
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_pusher.py",

    # Configuration
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/configuration/s3_operations.py",

    # Constants
    f"{project_name}/constant/__init__.py",
    f"{project_name}/constant/training_pipeline/__init__.py",
    f"{project_name}/constant/application.py",

    # Entities
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/artifacts_entity.py",
    f"{project_name}/entity/config_entity.py",

    # Exception
    f"{project_name}/exception/__init__.py",

    # Logger
    f"{project_name}/logger/__init__.py",

    # Pipeline
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",

    # Utilities
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",

    # Application
    "template/index.html",
    "app.py",

    # Configuration files
    ".dockerignore",
    "Dockerfile",
    "requirements.txt",
]


# Create Project Structure

for filepath in list_of_files:

    filepath = Path(filepath)

    # Create parent directories
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Create file only if it does not exist or is empty
    if not filepath.exists() or filepath.stat().st_size == 0:

        filepath.touch()

        logging.info(
            f"Creating file: {filepath}"
        )

    else:

        logging.info(
            f"{filepath} already exists"
        )
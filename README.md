# CopyrightSearch

CopyrightSearch is a fine-tuned small language model designed to identify copyright statements, authors, and contributions in source C and header files.

## Project Overview
sft_project/
│── data/
│   │── raw/                # Datos sin procesar (descargados de GitHub o local)
│   │── processed/          # Datos limpios y preparados
│   │── hf_dataset/         # Datos listos para subir a HF
│   ├── extract_data.py     # Extrae datos de GitHub API o un directorio local
│   ├── prepare_data.py     # Prepara los datos (limpieza, preprocesamiento)
│   ├── upload_to_hf.py     # Sube los datos procesados a HF en formato dataset
│
│── training/
│   │── config.yml          # Configuraciones para entrenamiento (YAML, JSON)
│   │── logs/               # Logs de entrenamiento
│   ├── train_sft.py        # Script de entrenamiento SFT
│   ├── model_setup.py      # Carga y configura el modelo (incluye pad token)
│   ├── training_args.py    # Define los argumentos de entrenamiento
│
│── inference/
│   ├── infer.py            # Carga el modelo entrenado y realiza inferencias
│
│── scripts/
│   ├── install_requirements.sh  # Instala dependencias necesarias
│   ├── run_all.sh               # Ejecuta todo el pipeline secuencialmente
│
│── notebooks/              # Notebooks opcionales para exploración y debugging
│   ├── data_analysis.ipynb
│   ├── training_monitoring.ipynb
│
│── README.md               # Documentación del proyecto
│── requirements.txt        # Dependencias del proyecto
│── .gitignore              # Archivos a ignorar en Git
│── .env                    # Archivo de entorno para credenciales


## Dataset Overview
- **Total source files (.c / .h):** 250.505
- **Training Process:**
  1. **First phase:** Training with 48.201 samples
  2. **Second phase:** Expanding to 150.000 samples
  3. **Final phase:** Training with the full dataset (250.505 samples)

## Retraining Strategy
- The model will be iteratively retrained with increasing dataset sizes to improve accuracy and generalization.

## Future Improvements
- Optimize model performance and inference speed.
- Enhance detection accuracy with additional fine-tuning steps.
- Explore potential integrations with external AI frameworks.

### License
This project follows an open-source approach. Ensure compliance with relevant licenses when using the dataset and trained models.

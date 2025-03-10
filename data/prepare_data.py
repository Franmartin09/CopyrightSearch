#PREAPRA LOS DATOS Y LOS SUBE A HF
import json
import os
from huggingface_hub import HfApi, HfFolder
from dotenv import load_dotenv
load_dotenv()

# Configuración
HF_TOKEN = os.getenv("HF_TOKEN")  # Reemplázalo con tu token de Hugging Face
MODEL_PATH = "./CopyrightSearchLM"  # Ruta al directorio de tu modelo
DATASET_PATH = "data_cleaning/final.json"  # Ruta al archivo JSON con el dataset
MODEL_REPO = "Franmartin09/CopyrightSearchLM-100M"  # Repositorio en Hugging Face para el modelo
DATASET_REPO = "Franmartin09/CopyrightSearch"  # Repositorio en Hugging Face para el dataset

# Autenticación
HfFolder.save_token(HF_TOKEN)
api = HfApi()

# Cargar el JSON del dataset
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset_data = json.load(f)

# Modificar cada elemento del array
for item in dataset_data:
    item.pop("file", None)
    if "comment" in item:
        item["input"] = item.pop("comment")  # Renombrar 'comment' a 'input'
    if "result" in item:
        item["output"] = item.pop("result")  # Renombrar 'result' a 'output'

# Guardar dataset como JSONL (formato estándar en HF)
dataset_jsonl_path = "dataset.jsonl"
with open(dataset_jsonl_path, "w", encoding="utf-8") as f:
    for entry in dataset_data:
        f.write(json.dumps(entry) + "\n")

# Subir el dataset
api.create_repo(DATASET_REPO, private=True, repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj=dataset_jsonl_path,
    path_in_repo="dataset.jsonl",
    repo_id=DATASET_REPO,
    repo_type="dataset",
)
print(f"Dataset subido exitosamente a {DATASET_REPO}")

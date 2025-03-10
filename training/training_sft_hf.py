# Import necessary libraries
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict
from trl import SFTConfig, SFTTrainer, setup_chat_format
from huggingface_hub import HfApi, HfFolder
import yaml

load_dotenv()

# Apply chat template to your dataset
def apply_template(example):
    instruction = f"Extract copyright from this text:\n{example['comment']}\n"
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": example['result']}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": formatted_prompt}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# Definir un token de padding dedicado
tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
model.resize_token_embeddings(len(tokenizer))  # Ajustar los embeddings tras añadir tokens especiales

# Configurar IDs de tokens especiales antes de `setup_chat_format`
model.config.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

# Aplicar el formato de chat (puede sobrescribir pad_token, así que lo arreglamos después)
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

# Volver a establecer correctamente el `pad_token` después de `setup_chat_format`
tokenizer.pad_token = "<|pad|>"
model.config.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")

# Verificar los IDs de los tokens
print("unk_token:", tokenizer.unk_token, "-> id:", tokenizer.convert_tokens_to_ids(tokenizer.unk_token))
print("pad_token:", tokenizer.pad_token, "-> id:", tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
print("eos_token:", tokenizer.eos_token, "-> id:", tokenizer.convert_tokens_to_ids(tokenizer.eos_token))

# Set our name for the finetune to be saved &/ uploaded to
finetune_name = "CopyrightSearchLM"

# Load dataset from JSONL file
dataset = load_dataset("json", data_files="final.json", split="train")

# Split dataset into 80% train and 20% test
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Create train and test datasets
ds = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"]
})

# Apply the formatting to the dataset
ds["train"] = ds["train"].map(apply_template)
ds["test"] = ds["test"].map(apply_template)

# Print dataset info
print(f"Train size: {len(ds['train'])}, Test size: {len(ds['test'])}")

# Configure the SFTTrainer
sft_config = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,  # Adjust based on dataset size and desired training duration
    per_device_train_batch_size=4,  # Set according to your GPU memory capacity
    learning_rate=5e-5,  # Common starting point for fine-tuning
    logging_steps=10,  # Frequency of logging training metrics
    save_steps=100,  # Frequency of saving model checkpoints
    eval_strategy="steps",  # Evaluate the model at regular intervals
    eval_steps=50,  # Frequency of evaluation
    use_mps_device=(
        True if device == "mps" else False
    ),  # Use MPS for mixed precision training
    hub_model_id=finetune_name,  # Set a unique name for your model
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds["train"],
    tokenizer=tokenizer,
    eval_dataset=ds["test"],
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(f"./{finetune_name}")

# Configuración
HF_TOKEN = os.getenv("HF_TOKEN")  # Reemplázalo con tu token de Hugging Face
MODEL_PATH = "./CopyrightSearchLM"  # Ruta al directorio de tu modelo
MODEL_REPO = "Franmartin09/CopyrightSearchLM-100M"  # Repositorio en Hugging Face para el modelo

# Autenticación
HfFolder.save_token(HF_TOKEN)
api = HfApi()

# Subir el modelo
api.create_repo(MODEL_REPO, private=True, exist_ok=True)
api.upload_folder(
    folder_path=MODEL_PATH,
    repo_id=MODEL_REPO,
    repo_type="model",
)
print(f"Modelo subido exitosamente a {MODEL_REPO}")
# Instalar dependencias necesarias
# pip install transformers datasets peft torch unsloth accelerate scikit-learn

import torch
import unsloth
import json
import random
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# Fix certificados en Windows (si es necesario)
from kostal_win_cert_fix.kostal_win_cert_fix.fix_certs import fix_windows_certs_for_requests_lib, get_cert_fix_context_manager
fix_windows_certs_for_requests_lib()
get_cert_fix_context_manager()

# 📌 Cargar modelo SmolLM2-135M con Unsloth para optimización
checkpoint = "HuggingFaceTB/SmolLM2-135M"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# 📌 Cargar dataset desde JSON
json_path = "data.json"  # Ruta del archivo JSON

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 📌 Convertir datos a lista de diccionarios
dataset = [{"input_text": d["comments"], "output_text": d["result"]} for d in data]

# 📌 Dividir dataset en train (80%) y test (20%)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# 📌 Convertir a formato `datasets.Dataset`
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# 📌 Función de tokenización
def tokenize_function(examples):
    return tokenizer(examples["input_text"], text_target=examples["output_text"], padding="max_length", truncation=True)

# Aplicar tokenización
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 📌 Definir argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True if device == "cuda" else False,  # Usar FP16 si hay GPU
    logging_dir="./logs",
    logging_steps=10
)

# 📌 Configurar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 🚀 Iniciar entrenamiento
trainer.train()

# 📌 Guardar modelo fine-tuneado
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

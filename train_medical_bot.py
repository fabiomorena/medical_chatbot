import os
import json
import torch
from datasets import Dataset
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    TrainingArguments
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer  # Nur SFTTrainer importieren

# Offline-Modus erzwingen
os.environ["HF_HUB_OFFLINE"] = "1"

# 1. Daten laden
print("Lade Datensatz...")
with open("medical_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
print(f"Daten geladen: {len(dataset)} Einträge")

# 2. Modell und Tokenizer laden (lokal)
print("Lade Modell...")
model_path = "/Users/fmorena/PycharmProjects/medical_chatbot/tinyllama_checkpoint/checkpoint-50"

tokenizer = LlamaTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

# Padding-Token setzen (wichtig!)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    device_map="auto",
    torch_dtype=torch.float16  # Speicher sparen
)

# 3. LoRA-Konfiguration (effizientes Training)
print("Konfiguriere LoRA...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=64,
    lora_alpha=16,
    lora_dropout=0.1
)

# 4. Training konfigurieren (klassische Methode)
training_args = TrainingArguments(
    output_dir="./medical-bot-checkpoint",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,  # Wichtig für Mac (deaktiviert)
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False  # macOS-Kompatibilität
)

# 5. Trainer erstellen (klassische Methode)
# Warnungen über veraltete Argumente können ignoriert werden,
# solange das Training startet.
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,              # Klassisch
    dataset_text_field="instruction", # Klassisch
    max_seq_length=512,               # Klassisch
    packing=False                     # Klassisch
)

# 6. Training starten
print("Starte Training...")
trainer.train()

# 7. Modell speichern
print("Speichere Modell...")
trainer.save_model("./medical-bot-final")
print("Training abgeschlossen! 🎉")


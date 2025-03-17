# ✅ Install dependencies
# !pip install -q transformers datasets accelerate torch peft

import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# ✅ Enable GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ✅ Load Dataset (JSON File with "code" and "explanation")
dataset_path = "/content/mern_dataset_v2.json"  # Change path if needed

with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ✅ Modify Dataset (Explanation → Code)
formatted_data = [
    {"text": f"### Explanation:\n{entry['explanation']}\n\n### Generate JavaScript code:\n"}
    for entry in data
]

dataset = Dataset.from_list(formatted_data)

# ✅ Load Tokenizer (from base model)
MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

# ✅ Tokenization Function (Add Labels for Loss Calculation)
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    tokens["labels"] = tokens["input_ids"].copy()  # ✅ Labels are required for training loss
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ✅ Split Dataset (90% Train, 10% Eval)
split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split["train"]
eval_dataset = split["test"]

# ✅ Load Model on GPU with FP16 for Speed
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)

# ✅ Apply LoRA for Efficient Training
config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, config)
model.print_trainable_parameters()

# ✅ Training Arguments
training_args = TrainingArguments(
    output_dir="./distilgpt2_code_gen",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    push_to_hub=False,
    fp16=True,  # ✅ Enable Mixed Precision
)

# ✅ Trainer Initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# ✅ Train Model on GPU
trainer.train()

# ✅ Save Fine-Tuned Model
trainer.save_model("distilgpt2_code_gen")
tokenizer.save_pretrained("distilgpt2_code_gen")

print("🎯 Fine-tuning complete! Model saved as 'distilgpt2_code_gen'.")

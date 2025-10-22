# LLM Fine-Tuning Script for Trip Planning
# Fine-tune a model for better trip planning responses.

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

# Load model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Dummy dataset (replace with real trip data)
train_dataset = [
    {"input": "Plan a trip to Paris.", "output": "Visit Eiffel Tower, Louvre, enjoy croissants."}
]

# Tokenize
inputs = tokenizer([ex["input"] for ex in train_dataset], return_tensors="pt", padding=True)
labels = tokenizer([ex["output"] for ex in train_dataset], return_tensors="pt", padding=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Use proper dataset here
)

# Train
trainer.train()

# Save
model.save_pretrained("./fine_tuned_trip_model")
tokenizer.save_pretrained("./fine_tuned_trip_model")

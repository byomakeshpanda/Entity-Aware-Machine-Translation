import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
import torch.nn.functional as F
import torch
from loss_function import CustomTrainer
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Flan-T5 model")
    parser.add_argument("--num_train_epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--output_dir", type=str, default="src/model/t5_large_finetuned", help="Output directory for the model")
    parser.add_argument("--train_data", type=str, default="data/train/train_data.json", help="Path to training data")
    return parser.parse_args()

def preprocess_data(data, output_path):
    formatted_data = []
    for sample in data:
        source_text = sample["source"]
        target_text = sample["target"]
        entities = sample.get("enriched_entities", [])

        entity_annotations = [f"{ent['entity_name']['en']} [{ent['entity_type']}]" for ent in entities]
        entity_text = ", ".join(entity_annotations) if entity_annotations else "None"

        # Reduce NER examples to avoid overfitting
        if len(formatted_data) % 3 == 0:  # Keep only 1/3 NER examples
            formatted_data.append({
                "task": "NER",
                "input": f"Recognize entities: {source_text}",
                "output": entity_text
            })

        # Keep more translation examples
        formatted_data.append({
            "task": "Entity-aware MT",
            "input": f"Entity translate (EN→FR): {source_text}",
            "output": target_text
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)

    print(f"Processed data saved to {output_path}")

def preprocess_function(samples):
    # Tokenize inputs and targets
    inputs = tokenizer(samples["input"], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(samples["output"], padding="max_length", truncation=True, max_length=128)

    # Set the labels for the inputs
    inputs["labels"] = targets["input_ids"]

    # Move tensors to CUDA (GPU) or CPU
    inputs = {key: torch.tensor(value).to(device) for key, value in inputs.items()}
    
    return inputs
def load_and_prepare_data(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Dataset.from_list(data)


current_path = os.getcwd()
print(current_path)
with open(current_path+r"/data/train/train_data.json", "r", encoding="utf-8") as f:
    data_train = json.load(f)
preprocess_data(data_train,  current_path+r"/data/processed/train_processed_data.json")

data_path = current_path+r"/data/processed/train_processed_data.json"
dataset = load_and_prepare_data(data_path)



tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

output_dir = os.path.abspath(current_path+r"/src/model/t5_large_finetuned")

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=15,
    weight_decay=0.01,
    save_total_limit=2,
    push_to_hub=False,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
import os
import json
import random
import evaluate
from tqdm import tqdm
from transformers import T5Tokenizer,T5ForConditionalGeneration

file_path = os.path.abspath(os.getcwd()+r"/src/inference/test_data.json")
print(file_path)
with open(file_path, "r", encoding="utf-8") as f:
    data_test = json.load(f)

def preprocess_data(data):
    formatted_data = []
    for sample in data:
        source_text = sample["source"]
        target_text = sample["target"]
        formatted_data.append({
            "task": "Entity-aware MT",
            "input": f"Entity translate (EN→FR): {source_text}",
            "output": target_text
        })

    return formatted_data


test_data = preprocess_data(data_test)

def restore_entities(text, entity_mapping):
    """Replace placeholders in translated text with original entity names."""
    for placeholder, original in entity_mapping.items():
        text = text.replace(placeholder, original)
    return text

current_path = os.getcwd()
# model_path = os.path.join(current_path, '..', '/model', 't5_large_finetuned') # Replace with the actual path to your saved model
model_path = os.path.abspath(os.getcwd()+r"/src/model/t5_large_finetuned")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

bleu = evaluate.load("sacrebleu")
def evaluate_bleu_on_random_subset(test_data, num_samples):
    random_samples = random.sample(test_data, num_samples)
    predictions, references = [], []

    for sample in tqdm(random_samples, desc="Evaluating Translations"):
        input_text = sample["input"]
        expected_output = sample["output"]
        entity_mapping = sample.get("enriched_entities", {})

        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(**inputs)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        final_output = restore_entities(decoded_output, entity_mapping)
        final_reference = restore_entities(expected_output, entity_mapping)

        predictions.append(final_output)
        references.append([final_reference])

    bleu_score = bleu.compute(predictions=predictions, references=references)
    return bleu_score

bleu_score = evaluate_bleu_on_random_subset(test_data,int(input("Enter the number of samples to test: ")))
print(f"BLEU Score: {bleu_score['score']:.2f}")
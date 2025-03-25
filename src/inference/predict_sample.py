from transformers import T5Tokenizer,T5ForConditionalGeneration
import torch
import os

current_path = os.getcwd()
model_path = os.path.abspath(os.getcwd()+r"/src/model/t5_large_finetuned")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Move the model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Example input text
input_text = "Entity translate (EN→FR): "+ input("Enter the english sentence:\n")

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

# Generate the output using the model
outputs = model.generate(**inputs)

# Decode the generated output
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the decoded output
print("Decoded Output:", decoded_output)
## Preprocessing the data to divide the tasks for training (Both NER and EA-MT required)

def preprocess_train(data, output_path):
    formatted_data = []
    for sample in data:
        source_text = sample["source"]
        target_text = sample["target"]
        entities = sample.get("enriched_entities", [])

        entity_annotations = [f"{ent['entity_name']['en']} [{ent['entity_type']}]" for ent in entities]
        entity_text = ", ".join(entity_annotations) if entity_annotations else "None"

        # Reduced NER examples to avoid overfitting
        if len(formatted_data) % 3 == 0: 
            formatted_data.append({
                "task": "NER",
                "input": f"Recognize entities: {source_text}",
                "output": entity_text
            })

        formatted_data.append({
            "task": "Entity-aware MT",
            "input": f"Entity translate (EN→FR): {source_text}",
            "output": target_text
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)

    print(f"Processed data saved to {output_path}")

# While testing only the entity-aware mt is required

def preprocess_test(data, output_path):
    formatted_data = []
    for sample in data:
        source_text = sample["source"]
        target_text = sample["target"]
        entities = sample.get("enriched_entities", [])

        entity_annotations = [f"{ent['entity_name']['en']} [{ent['entity_type']}]" for ent in entities]
        entity_text = ", ".join(entity_annotations) if entity_annotations else "None"

        formatted_data.append({
            "task": "Entity-aware MT",
            "input": f"Entity translate (EN→FR): {source_text}",
            "output": target_text
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)

    print(f"Processed data saved to {output_path}")
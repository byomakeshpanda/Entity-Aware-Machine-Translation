import torch
from transformers import Trainer
import torch.nn.functional as F

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss function to prioritize translation over NER."""
        labels = inputs.pop("labels")  # Extract target labels
        outputs = model(**inputs)
        logits = outputs.logits  # Get logits

        # Compute CrossEntropy loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        # Assign higher weight to translation (80%) and lower weight to NER (20%)
        ner_weight = 0.2
        translation_weight = 0.8

        # Get task type (default to Translation)
        task_type = inputs.get("task_type", ["Translation"] * logits.shape[0])

        # Convert task type to weight tensor
        task_weights = torch.tensor(
            [ner_weight if "NER" in task else translation_weight for task in task_type],
            device=logits.device,
            dtype=torch.float,
        )

        # Scale loss by task weights
        weighted_loss = loss * task_weights.mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss
    
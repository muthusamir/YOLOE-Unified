import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class DistilledCLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16", distill_ratio=0.5):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.text_encoder = AutoModel.from_pretrained(model_name)
        
        # Distillation: prune layers or use smaller backbone
        # Here using MobileCLIP as lightweight student
        if distill_ratio < 1.0:
            from mobileclip import MobileCLIP
            self.text_encoder = MobileCLIP(model_name="mobileclip_b").text_model

        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.text_encoder.device)
        with torch.no_grad():
            embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

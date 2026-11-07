# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

class CLMProjectionHead(nn.Module):
    """
    Exact CLM-8B Head Spec:
    Depth 3, Width 1536, Projection 512, LayerNorm, GELU, No residual.
    """
    def __init__(self, in_features: int, hidden_dim: int = 1536, out_dim: int = 512):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.block(x)
        # Final output is L2-normalized into unit hypersphere
        return F.normalize(projected, p=2, dim=-1)

class CLM8B(nn.Module):
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-7B",  # or Qwen/Qwen3-8B
        load_in_4bit: bool = True
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load backbone quantized to run on free 16GB GPUs (T4)
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.backbone = AutoModel.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.backbone = AutoModel.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

        # Freeze the 8B backbone completely
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        embed_dim = self.backbone.config.hidden_size

        # Exact separate dual heads
        self.state_head = CLMProjectionHead(in_features=embed_dim)
        self.action_head = CLMProjectionHead(in_features=embed_dim)

        # Logit scale (initialized to ln(100) ~ 4.6051)
        self.logit_scale = nn.Parameter(torch.tensor(4.6051))

    def extract_backbone_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs[0]  # [B, L, D]
            
            # Extract last active token or mean-pooling
            # CLM uses last non-padding token post-norm embedding
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            last_token_hidden = hidden_states[torch.arange(batch_size, device=input_ids.device), sequence_lengths]
            
            # L2-normalize before passing into head
            return F.normalize(last_token_hidden.float(), p=2, dim=-1)

    def encode_state(self, texts: list[str], max_length: int = 512) -> torch.Tensor:
        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        tok = {k: v.to(self.backbone.device) for k, v in tok.items()}
        emb = self.extract_backbone_embedding(tok["input_ids"], tok["attention_mask"])
        return self.state_head(emb)

    def encode_action(self, texts: list[str], max_length: int = 128) -> torch.Tensor:
        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        tok = {k: v.to(self.backbone.device) for k, v in tok.items()}
        emb = self.extract_backbone_embedding(tok["input_ids"], tok["attention_mask"])
        return self.action_head(emb)

    def forward(self, state_texts: list[str], action_texts: list[str]) -> torch.Tensor:
        """Computes InfoNCE loss across the batch."""
        s_vec = self.encode_state(state_texts)
        a_vec = self.encode_action(action_texts)

        # Cosine similarity matrix scaled by logit_scale
        scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        logits = torch.matmul(s_vec, a_vec.T) * scale

        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
        return loss
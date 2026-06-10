"""
DINOv2 backbone wrappers for semi-supervised learning.

CLS token (last_hidden_state[:, 0, :]) used as feature vector.

  dinov2_vitb14  —  facebook/dinov2-base,  feat_dim=768
  dinov2_vitl14  —  facebook/dinov2-large, feat_dim=1024

Optional LoRA mode (enable via cfg `use_lora: True`, `lora_dim: 16`):
  freezes the HF encoder and injects LoRA on Q,V of every attention block.
  Trainable parameters become: LoRA-AB on Q/V + classifier head.

The LoRA modules are instantiated INSIDE the attention layers (Q,V replaced
with LinearWithLoRA wrappers). The `lora_modules` ModuleList holds the same
objects to make state_dict() round-trips and deepcopy reference-sharing
behave predictably.
"""

import math

import torch
import torch.nn as nn
from transformers import Dinov2Model


# ── LoRA primitives ──────────────────────────────────────────────────────────

class LoRA(nn.Module):
    """Standard rank-r LoRA: y = (x A) B * (1/r)."""
    def __init__(self, in_dim, bottle_dim):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_dim, bottle_dim))
        self.lora_B = nn.Parameter(torch.zeros(bottle_dim, in_dim))
        self.scaling = 1.0 / bottle_dim
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LinearWithLoRA(nn.Module):
    """Wraps a frozen Linear and adds a trainable LoRA delta."""
    def __init__(self, linear, lora):
        super().__init__()
        self.linear = linear
        self.lora = lora

    def forward(self, x):
        return self.linear(x) + self.lora(x)


# ── Backbone wrapper ─────────────────────────────────────────────────────────

class DINOv2Wrapper(nn.Module):
    def __init__(self, hf_name, num_classes, feat_dim,
                 use_lora=False, lora_dim=16):
        super().__init__()
        self.encoder = Dinov2Model.from_pretrained(hf_name)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.feat_dim = feat_dim
        self.use_lora = bool(use_lora)

        if self.use_lora:
            # Freeze backbone
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            # Inject LoRA on Q,V of each transformer block
            n_layers = len(self.encoder.encoder.layer)
            self.lora_modules = nn.ModuleList()
            for i in range(n_layers):
                attn = self.encoder.encoder.layer[i].attention.attention
                lq = LoRA(feat_dim, lora_dim)
                lv = LoRA(feat_dim, lora_dim)
                attn.query = LinearWithLoRA(attn.query, lq)
                attn.value = LinearWithLoRA(attn.value, lv)
                # Keep aliases so state_dict / parameters() includes LoRA params
                # under a stable name regardless of HF internal reshuffling.
                self.lora_modules.append(nn.ModuleDict({"q": lq, "v": lv}))

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        out = self.encoder(pixel_values=x, return_dict=True)
        feat = out.last_hidden_state[:, 0, :]   # CLS token

        if only_feat:
            return feat
        logits = self.classifier(feat)
        if only_fc:
            return logits
        return {'logits': logits, 'feat': feat}

    def group_matcher(self, coarse=False, prefix=''):
        return dict(
            stem=r'^{}encoder.embeddings'.format(prefix),
            blocks=r'^{}encoder.encoder.layer\.(\d+)'.format(prefix),
        )

    def no_weight_decay(self):
        return []


# ── Factories ────────────────────────────────────────────────────────────────

def dinov2_vitb14(num_classes, pretrained=True,
                  pretrained_path='facebook/dinov2-base',
                  use_lora=False, lora_dim=16, **kwargs):
    return DINOv2Wrapper('facebook/dinov2-base', num_classes,
                         feat_dim=768, use_lora=use_lora, lora_dim=lora_dim)


def dinov2_vitl14(num_classes, pretrained=True,
                  pretrained_path='facebook/dinov2-large',
                  use_lora=False, lora_dim=16, **kwargs):
    return DINOv2Wrapper('facebook/dinov2-large', num_classes,
                         feat_dim=1024, use_lora=use_lora, lora_dim=lora_dim)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
CLIP ViT-B/16 visual encoder wrapper for the USB semi-supervised framework.

The vision encoder is extracted from CLIP and a linear classification head is added.
Weights are loaded from HuggingFace Hub automatically on first use.

Optional LoRA mode (enable via cfg `use_lora: True`, `lora_dim: 16`):
  freezes the CLIP vision encoder and injects LoRA on the q_proj / v_proj of
  every attention block. Trainable parameters: LoRA-AB + classifier head.

Supported net names (set net: <name> in config):
    clip_vit_base_patch16   — CLIP ViT-B/16, img_size=224, feat_dim=768
    clip_vit_large_patch14  — CLIP ViT-L/14, img_size=224, feat_dim=1024
"""

import math

import torch
import torch.nn as nn
from transformers import CLIPModel


# ── LoRA primitives (kept local so the file stays self-contained) ────────────

class LoRA(nn.Module):
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
    def __init__(self, linear, lora):
        super().__init__()
        self.linear = linear
        self.lora = lora

    def forward(self, x):
        return self.linear(x) + self.lora(x)


# ── Backbone wrapper ─────────────────────────────────────────────────────────

class CLIPVisionWrapper(nn.Module):
    """
    Wraps CLIP's vision encoder into the USB interface:
        forward(x) -> {'logits': [B, num_classes], 'feat': [B, feat_dim]}
    """
    def __init__(self, num_classes, pretrained=True,
                 pretrained_path='openai/clip-vit-base-patch16', feat_dim=768,
                 use_lora=False, lora_dim=16):
        super().__init__()
        clip = CLIPModel.from_pretrained(pretrained_path)
        self.encoder = clip.vision_model
        self.feat_dim = feat_dim
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.use_lora = bool(use_lora)

        if self.use_lora:
            # Freeze the CLIP vision encoder
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            # Inject LoRA on q_proj / v_proj of each transformer layer
            n_layers = len(self.encoder.encoder.layers)
            self.lora_modules = nn.ModuleList()
            for i in range(n_layers):
                attn = self.encoder.encoder.layers[i].self_attn
                lq = LoRA(feat_dim, lora_dim)
                lv = LoRA(feat_dim, lora_dim)
                attn.q_proj = LinearWithLoRA(attn.q_proj, lq)
                attn.v_proj = LinearWithLoRA(attn.v_proj, lv)
                self.lora_modules.append(nn.ModuleDict({"q": lq, "v": lv}))

    def forward(self, x):
        out = self.encoder(pixel_values=x, return_dict=True)
        # pooler_output: CLS token after projection, shape [B, feat_dim]
        feat = out.pooler_output
        logits = self.classifier(feat)
        return {'logits': logits, 'feat': feat}


def clip_vit_base_patch16(num_classes, pretrained=True,
                          pretrained_path='openai/clip-vit-base-patch16',
                          use_lora=False, lora_dim=16, **kwargs):
    """CLIP ViT-B/16: img_size=224, patch=16, feat_dim=768."""
    return CLIPVisionWrapper(num_classes, pretrained=pretrained,
                             pretrained_path=pretrained_path, feat_dim=768,
                             use_lora=use_lora, lora_dim=lora_dim)


def clip_vit_large_patch14(num_classes, pretrained=True,
                           pretrained_path='openai/clip-vit-large-patch14',
                           use_lora=False, lora_dim=16, **kwargs):
    """CLIP ViT-L/14: img_size=224, patch=14, feat_dim=1024."""
    return CLIPVisionWrapper(num_classes, pretrained=pretrained,
                             pretrained_path=pretrained_path, feat_dim=1024,
                             use_lora=use_lora, lora_dim=lora_dim)

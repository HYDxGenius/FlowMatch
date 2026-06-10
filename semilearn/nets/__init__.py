# Network registry for FlowMatch.
# Backbones used in the paper: CLIP ViT-B/16, DINOv2 ViT-B/14
# (plus ResNet/WRN/ViT for CIFAR / STL-10 conventional SSL experiments).

from .resnet   import resnet50
from .wrn      import wrn_28_2, wrn_28_8, wrn_var_37_2
from .vit      import (vit_base_patch16_224, vit_small_patch16_224,
                       vit_small_patch2_32, vit_tiny_patch2_32, vit_base_patch16_96)
from .clip_vit import clip_vit_base_patch16, clip_vit_large_patch14
from .dinov2   import dinov2_vitb14, dinov2_vitl14

"""
BioMedCLIP Vision Encoder Wrapper
===================================
Wraps microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (open_clip)
and exposes a clean encode_image() interface compatible with the SecureMed-LLM
pipeline.

Paper reference (Section 3.2.1):
  "BioMedCLIP is pre-trained on large-scale biomedical image-text pairs,
   enabling clinically meaningful joint representations directly transferable
   to radiology report generation."
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

BIOMED_CLIP_MODEL_TAG = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMED_CLIP_EMBED_DIM = 512   # ViT-B/16 output embedding size


class BioMedCLIPEncoder(nn.Module):
    """
    Thin wrapper around BioMedCLIP that:
      1. Loads the pretrained ViT-B/16 backbone via open_clip.
      2. Exposes encode_image() returning L2-normalised image embeddings.
      3. Freezes backbone weights by default (fine-tune only the projection).

    Args:
        pretrained_tag: open_clip model tag (default = BioMedCLIP HF hub path).
        freeze_backbone: If True, backbone parameters are frozen.
        device: Target device.
    """

    def __init__(
        self,
        pretrained_tag: str = BIOMED_CLIP_MODEL_TAG,
        freeze_backbone: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.embed_dim = BIOMED_CLIP_EMBED_DIM
        self.device = device or torch.device("cpu")

        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip is required for BioMedCLIPEncoder. "
                "Install with: pip install open-clip-torch"
            )

        logger.info("Loading BioMedCLIP from %s ...", pretrained_tag)
        model, _, preprocess = open_clip.create_model_and_transforms(pretrained_tag)
        self.backbone = model.visual       # ViT-B/16 visual branch
        self.preprocess = preprocess       # stored for external use if needed

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("BioMedCLIP backbone frozen.")

        self.backbone.to(self.device)

    # ------------------------------------------------------------------
    #  Forward / encode
    # ------------------------------------------------------------------

    def forward(self, images: Tensor) -> Tensor:
        """
        Encode a batch of images into L2-normalised embeddings.

        Args:
            images: Float tensor of shape (B, 3, H, W), values in [0,1]
                    normalised with ImageNet stats.

        Returns:
            Tensor of shape (B, embed_dim) — L2 normalised.
        """
        return self.encode_image(images)

    def encode_image(self, images: Tensor) -> Tensor:
        """
        Extract visual embeddings from a batch of images.

        Args:
            images: (B, 3, H, W) normalised image tensor.

        Returns:
            (B, embed_dim) L2-normalised embedding tensor.
        """
        images = images.to(self.device)
        features = self.backbone(images)      # (B, embed_dim)
        # L2 normalise — consistent with CLIP training objective
        features = nn.functional.normalize(features, p=2, dim=-1)
        return features

    # ------------------------------------------------------------------
    #  Utility
    # ------------------------------------------------------------------

    def get_embed_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self.embed_dim

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters (for full fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("BioMedCLIP backbone unfrozen.")

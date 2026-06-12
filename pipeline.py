"""
SecureMed-LLM Full Pipeline
==============================
Integrates the BioMedCLIP vision encoder and T5 text decoder into a single
end-to-end module for privacy-preserving clinical report generation.

Paper reference (Section 3.3 — Design Rationale of the Pipeline Integration):
  "The combination of BioMedCLIP and T5 is motivated by the need to balance
   three competing objectives: (i) clinical fidelity, (ii) privacy preservation,
   and (iii) deployability. BioMedCLIP ensures high-quality visual grounding,
   while T5 ensures structured and controllable text generation. Their integration
   enables a modular architecture where privacy-preserving mechanisms (Med-Guard,
   DP-SGD, validation, encryption) can operate independently without interfering
   with core representation learning or generation processes."

Pipeline data-flow (Section 5.1c):
  image (224×224, ImageNet-normalised)
      └─► BioMedCLIPEncoder.encode_image()  →  (B, 512) L2-normalised embedding
              └─► T5Decoder.forward() / .generate()
                      visual prefix (linear projection) + text token embeddings
                              └─► structured clinical report string

This module is the primary object passed to:
  - dp_training.py   (DP-SGD wrapping via Opacus)
  - adv_training.py  (adversarial fine-tuning)
  - inference.py     (end-to-end report generation)
  - evaluate.py      (BLEU, MIA, PHI leakage evaluation)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.models.biomed_clip_encoder import BioMedCLIPEncoder
from src.models.t5_decoder import T5Decoder

logger = logging.getLogger(__name__)


class SecureMedPipeline(nn.Module):
    """
    End-to-end BioMedCLIP + T5 pipeline for clinical report generation.

    Decouples perception (BioMedCLIP vision encoder) from generation (T5 decoder)
    as described in Section 3.4 — this separation allows each security layer
    (Med-Guard, DP-SGD, IDS-LLM, ECIES) to operate independently without
    interfering with core representation learning or generation.

    Args:
        freeze_encoder:    Freeze BioMedCLIP backbone weights (default: True).
                           Only the linear projection layer in T5Decoder is trained
                           unless this is set to False.
        freeze_decoder:    Freeze T5 backbone weights (default: False).
                           Setting True trains only the visual projection layer.
        t5_model_name:     HuggingFace T5 variant (default: 't5-base').
        visual_embed_dim:  BioMedCLIP output dimensionality (default: 512).
        num_visual_tokens: Visual prefix tokens injected into T5 encoder (default: 1).
        max_seq_length:    Maximum token length for report generation (default: 128).
        device:            Target device. Inferred from CUDA availability if None.
    """

    def __init__(
        self,
        freeze_encoder:    bool = True,
        freeze_decoder:    bool = False,
        t5_model_name:     str  = "t5-base",
        visual_embed_dim:  int  = 512,
        num_visual_tokens: int  = 1,
        max_seq_length:    int  = 128,
        device:            Optional[torch.device] = None,
    ):
        super().__init__()

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # ── Vision encoder ──────────────────────────────────────────────
        self.encoder = BioMedCLIPEncoder(
            freeze_backbone=freeze_encoder,
            device=self.device,
        )

        # ── Text decoder with visual projection ─────────────────────────
        self.decoder = T5Decoder(
            t5_model_name=t5_model_name,
            visual_embed_dim=visual_embed_dim,
            num_visual_tokens=num_visual_tokens,
            freeze_t5=freeze_decoder,
            max_seq_length=max_seq_length,
        )

        self.to(self.device)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "SecureMedPipeline ready | device=%s | trainable params=%s",
            self.device, f"{n_trainable:,}",
        )

    # ------------------------------------------------------------------
    #  Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        images:        Tensor,                    # (B, 3, H, W)
        input_ids:     Tensor,                    # (B, seq_len)
        attention_mask: Tensor,                   # (B, seq_len)
        labels:        Optional[Tensor] = None,   # (B, seq_len)
    ) -> Dict[str, Tensor]:
        """
        Full forward pass: image encoding → visual prefix → T5 generation.

        Teacher-forcing cross-entropy loss is computed when labels are provided,
        matching the training objective described in Section 5.1c:
          "The model was trained using teacher forcing with a standard
           cross-entropy loss."

        Args:
            images:         Normalised chest X-ray tensor (B, 3, H, W).
            input_ids:      Tokenised report input_ids (B, seq_len).
            attention_mask: Attention mask (B, seq_len).
            labels:         Target token ids for loss computation (B, seq_len).
                            Pass None for inference-only forward.

        Returns:
            Dict with keys:
              'loss'            – scalar CE loss (only when labels provided)
              'logits'          – (B, seq_len, vocab_size)
              'visual_embeds'   – (B, visual_embed_dim) L2-normalised embeddings
        """
        # Step 1 — encode image into L2-normalised visual embedding
        visual_embeds = self.encoder.encode_image(images)   # (B, 512)

        # Step 2 — decode via T5 with visual prefix
        decoder_out = self.decoder(
            visual_embeddings=visual_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        decoder_out["visual_embeds"] = visual_embeds
        return decoder_out

    # ------------------------------------------------------------------
    #  Inference — report generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_report(
        self,
        images:         Tensor,
        max_new_tokens: int  = 128,
        num_beams:      int  = 4,
        early_stopping: bool = True,
    ) -> List[str]:
        """
        Generate clinical reports from a batch of chest X-ray images.

        Implements the inference path described in Section 5.2.3:
          "The anonymised image-text pair is processed by the SecureMed-LLM
           model in an offline inference server without any external API calls."

        Args:
            images:         Normalised image tensor (B, 3, H, W).
            max_new_tokens: Maximum number of tokens to generate per report.
            num_beams:      Beam search width (paper uses greedy-equivalent; 4 is standard).
            early_stopping: Stop generation when all beams reach EOS.

        Returns:
            List of B decoded report strings.
        """
        self.eval()
        images = images.to(self.device)

        # Encode
        visual_embeds = self.encoder.encode_image(images)   # (B, 512)

        # Generate token ids
        token_ids = self.decoder.generate(
            visual_embeddings=visual_embeds,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=early_stopping,
        )

        # Decode tokens to strings
        reports = self.decoder.decode_tokens(token_ids)
        return reports

    # ------------------------------------------------------------------
    #  MIA evaluation support (used by metrics.py compute_mia_accuracy)
    # ------------------------------------------------------------------

    def compute_log_prob(
        self,
        images:         Tensor,   # (B, 3, H, W)
        input_ids:      Tensor,   # (B, seq_len)
        attention_mask: Tensor,   # (B, seq_len)
    ) -> Tensor:
        """
        Compute average per-token log-probability of ground-truth report tokens.

        Used by the black-box confidence-thresholding MIA (Section 5.1i):
          "A black-box confidence-thresholding MIA is used as the primary
           empirical privacy measure … values approaching 50% indicate strong
           privacy protection."

        Args:
            images:         Normalised image tensor (B, 3, H, W).
            input_ids:      Ground-truth report token ids (B, seq_len).
            attention_mask: Attention mask (B, seq_len).

        Returns:
            (B,) tensor of average log-probabilities.
            Higher = model has likely seen this sample (member).
        """
        images        = images.to(self.device)
        input_ids     = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        visual_embeds = self.encoder.encode_image(images)
        return self.decoder.compute_log_prob(
            visual_embeddings=visual_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # ------------------------------------------------------------------
    #  Cosine similarity (paper Section 5.1d)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_cosine_similarity(
        self,
        images:         Tensor,   # (B, 3, H, W)
        input_ids:      Tensor,   # (B, seq_len)
        attention_mask: Tensor,   # (B, seq_len)
    ) -> Tensor:
        """
        Compute cosine similarity between BioMedCLIP image embeddings and
        mean-pooled T5 text embeddings, as in paper Section 5.1d:

          CosSim(x, y) = (x · y) / (‖x‖ ‖y‖)

          "At σ=15 the model maintains cosine similarity of 0.84 and
           BLEU-4 = 0.70 with reduced PHI leakage."

        Args:
            images:         Normalised image tensor (B, 3, H, W).
            input_ids:      Report token ids (B, seq_len).
            attention_mask: Attention mask (B, seq_len).

        Returns:
            (B,) cosine similarity tensor.
        """
        images         = images.to(self.device)
        input_ids      = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Image embedding: (B, 512), already L2-normalised by BioMedCLIPEncoder
        img_embeds = self.encoder.encode_image(images)

        # Text embedding: mean-pool T5 encoder hidden states over sequence
        txt_embeds = self._encode_text(input_ids, attention_mask)

        return nn.functional.cosine_similarity(img_embeds, txt_embeds, dim=-1)

    # ------------------------------------------------------------------
    #  Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        output_dir: Union[str, Path],
        epoch:      int,
        step:       Optional[int] = None,
        extra_meta: Optional[dict] = None,
    ) -> Path:
        """
        Save model weights and training metadata to a checkpoint file.

        Args:
            output_dir: Directory to write the checkpoint.
            epoch:      Current epoch number (used in filename).
            step:       Optional step number within the epoch.
            extra_meta: Optional dict of additional metadata to store.

        Returns:
            Path to the saved checkpoint file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        step_tag = f"_step{step}" if step is not None else ""
        ckpt_name = f"securemed_epoch{epoch:03d}{step_tag}.pt"
        ckpt_path = output_dir / ckpt_name

        payload = {
            "epoch":         epoch,
            "step":          step,
            "encoder_state": self.encoder.state_dict(),
            "decoder_state": self.decoder.state_dict(),
            "model_config": {
                "t5_model_name":     self.decoder.t5.config._name_or_path,
                "visual_embed_dim":  self.encoder.embed_dim,
                "num_visual_tokens": self.decoder.num_visual_tokens,
                "max_seq_length":    self.decoder.max_seq_length,
            },
        }
        if extra_meta:
            payload.update(extra_meta)

        torch.save(payload, ckpt_path)
        logger.info("Checkpoint saved → %s", ckpt_path)
        return ckpt_path

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device:          Optional[torch.device] = None,
        strict:          bool = True,
    ) -> "SecureMedPipeline":
        """
        Instantiate a SecureMedPipeline from a saved checkpoint.

        Args:
            checkpoint_path: Path to a .pt file saved by save_checkpoint().
            device:          Target device (defaults to CUDA if available).
            strict:          Passed to load_state_dict (default: True).

        Returns:
            Loaded SecureMedPipeline instance in eval() mode.
        """
        device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        payload = torch.load(checkpoint_path, map_location=device)
        cfg     = payload.get("model_config", {})

        pipeline = cls(
            t5_model_name=     cfg.get("t5_model_name",     "t5-base"),
            visual_embed_dim=  cfg.get("visual_embed_dim",  512),
            num_visual_tokens= cfg.get("num_visual_tokens", 1),
            max_seq_length=    cfg.get("max_seq_length",    128),
            device=device,
        )

        pipeline.encoder.load_state_dict(payload["encoder_state"], strict=strict)
        pipeline.decoder.load_state_dict(payload["decoder_state"], strict=strict)
        pipeline.eval()

        epoch = payload.get("epoch", "?")
        step  = payload.get("step",  "?")
        logger.info(
            "Checkpoint loaded from %s (epoch=%s, step=%s)",
            checkpoint_path, epoch, step,
        )
        return pipeline

    # ------------------------------------------------------------------
    #  Trainable parameter helpers (used by dp_training.py and adv_training.py)
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Return only the parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def projection_parameters(self):
        """
        Return only the visual projection layer parameters.
        Used when freeze_encoder=True and freeze_decoder=True to train
        just the bridging projection layer.
        """
        return list(self.decoder.visual_projection.parameters())

    def encoder_parameters(self):
        """Return BioMedCLIP backbone parameters."""
        return list(self.encoder.backbone.parameters())

    def decoder_parameters(self):
        """Return T5 backbone parameters (excluding projection)."""
        t5_params  = list(self.decoder.t5.parameters())
        proj_params = set(id(p) for p in self.decoder.visual_projection.parameters())
        return [p for p in t5_params if id(p) not in proj_params]

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _encode_text(
        self,
        input_ids:      Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        Encode token ids into mean-pooled T5 encoder hidden states.

        Used for cosine similarity evaluation (Section 5.1d):
          "cosine similarity was computed between BioMedCLIP image embeddings
           and T5 text embeddings (mean-pooled final hidden states)."

        Returns:
            L2-normalised text embedding tensor (B, d_model).
        """
        # Run T5 encoder only (no decoder)
        encoder_outputs = self.decoder.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = encoder_outputs.last_hidden_state       # (B, seq_len, d_model)

        # Mean-pool over non-padding positions
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed        = (hidden * mask_expanded).sum(dim=1)
        counts        = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooled   = summed / counts                  # (B, d_model)

        return nn.functional.normalize(mean_pooled, p=2, dim=-1)

    def _move_batch_to_device(self, batch: dict) -> dict:
        """Move all tensor values in a batch dict to self.device."""
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

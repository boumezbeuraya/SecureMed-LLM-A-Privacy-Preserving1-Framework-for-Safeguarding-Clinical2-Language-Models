"""
T5 Text Decoder with Visual Projection Layer
=============================================
Implements the T5-base encoder-decoder conditioned on BioMedCLIP visual
embeddings injected as continuous prefix tokens (Section 3.2.2 / Section 5.1c).

Architecture:
  BioMedCLIP embedding (512-d)
      └─► Linear projection  →  T5 embedding space (512-d for t5-base)
              └─► Prepended as prefix tokens to the T5 encoder input
                      └─► T5 decoder generates the clinical report

Paper reference:
  "A learnable linear projection layer maps visual features into the
   embedding space of the T5 encoder. Projected visual embeddings are
   injected as continuous prefix embeddings, concatenated with textual
   token embeddings prior to encoding."
"""

import logging
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import T5ForConditionalGeneration, AutoTokenizer

logger = logging.getLogger(__name__)

T5_MODEL_NAME  = "t5-base"
T5_EMBED_DIM   = 512          # t5-base d_model
NUM_VIS_TOKENS = 1            # number of visual prefix tokens injected


class T5Decoder(nn.Module):
    """
    T5-base conditioned on visual prefix embeddings.

    Args:
        t5_model_name: HuggingFace model identifier (default: 't5-base').
        visual_embed_dim: Dimensionality of incoming visual embeddings (512 for BioMedCLIP).
        num_visual_tokens: How many visual prefix tokens to inject (default: 1).
        freeze_t5: If True freeze all T5 weights except the projection layer.
        max_seq_length: Maximum generation / target sequence length.
    """

    def __init__(
        self,
        t5_model_name: str = T5_MODEL_NAME,
        visual_embed_dim: int = 512,
        num_visual_tokens: int = NUM_VIS_TOKENS,
        freeze_t5: bool = False,
        max_seq_length: int = 128,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.num_visual_tokens = num_visual_tokens

        logger.info("Loading T5 model: %s ...", t5_model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        t5_dim = self.t5.config.d_model  # 512 for t5-base

        # Linear projection: visual_embed_dim → (num_visual_tokens × t5_dim)
        self.visual_projection = nn.Linear(
            visual_embed_dim, num_visual_tokens * t5_dim, bias=True
        )
        nn.init.xavier_uniform_(self.visual_projection.weight)
        nn.init.zeros_(self.visual_projection.bias)

        if freeze_t5:
            for param in self.t5.parameters():
                param.requires_grad = False
            logger.info("T5 backbone frozen; only projection layer is trainable.")

    # ------------------------------------------------------------------
    #  Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        visual_embeddings: Tensor,           # (B, visual_embed_dim)
        input_ids: Tensor,                   # (B, seq_len) — tokenised report
        attention_mask: Tensor,              # (B, seq_len)
        labels: Optional[Tensor] = None,    # (B, seq_len) for teacher-forcing loss
    ) -> Dict[str, Tensor]:
        """
        Forward pass with teacher-forcing cross-entropy loss.

        The visual embedding is projected and prepended as a prefix token to
        the T5 encoder input embeddings.

        Args:
            visual_embeddings: BioMedCLIP output, shape (B, visual_embed_dim).
            input_ids: Tokenised report input_ids (B, seq_len).
            attention_mask: Attention mask aligned with input_ids (B, seq_len).
            labels: Target token ids for cross-entropy loss (B, seq_len).

        Returns:
            dict with keys:
              'loss'   -- scalar cross-entropy loss (if labels provided)
              'logits' -- (B, seq_len, vocab_size)
        """
        encoder_inputs_embeds, encoder_attention_mask = self._build_encoder_input(
            visual_embeddings, input_ids, attention_mask
        )

        if labels is not None:
            outputs = self.t5(
                inputs_embeds=encoder_inputs_embeds,
                attention_mask=encoder_attention_mask,
                labels=labels,
            )
            return {"loss": outputs.loss, "logits": outputs.logits}
        else:
            outputs = self.t5(
                inputs_embeds=encoder_inputs_embeds,
                attention_mask=encoder_attention_mask,
            )
            return {"logits": outputs.logits}

    # ------------------------------------------------------------------
    #  Generation (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        visual_embeddings: Tensor,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        early_stopping: bool = True,
    ) -> Tensor:
        """
        Autoregressively generate a clinical report from visual embeddings.

        Args:
            visual_embeddings: (B, visual_embed_dim).
            max_new_tokens: Maximum tokens to generate.
            num_beams: Beam search width.
            early_stopping: Stop when all beams reach EOS.

        Returns:
            Tensor of generated token ids (B, seq_len).
        """
        B = visual_embeddings.size(0)
        device = visual_embeddings.device

        dummy_ids  = torch.zeros(B, 1, dtype=torch.long, device=device)
        dummy_mask = torch.ones(B, 1, dtype=torch.long, device=device)

        encoder_inputs_embeds, encoder_attention_mask = self._build_encoder_input(
            visual_embeddings, dummy_ids, dummy_mask
        )

        generated = self.t5.generate(
            inputs_embeds=encoder_inputs_embeds,
            attention_mask=encoder_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=early_stopping,
            no_repeat_ngram_size=3,
        )
        return generated

    def decode_tokens(self, token_ids: Tensor) -> list:
        """Decode a batch of token id tensors to strings."""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    #  Log-probability (for MIA evaluation)
    # ------------------------------------------------------------------

    def compute_log_prob(
        self,
        visual_embeddings: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        Compute average per-token log-probability of the ground-truth report.
        Used by the MIA evaluator as a membership confidence score.

        Returns:
            (B,) tensor of average log-probs.
        """
        with torch.no_grad():
            out = self.forward(
                visual_embeddings, input_ids, attention_mask, labels=input_ids
            )
            logits = out["logits"]           # (B, seq_len, vocab_size)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            vocab = log_probs.shape[-1]
            target = input_ids.unsqueeze(-1).clamp(0, vocab - 1)   # (B, seq_len, 1)
            token_log_probs = log_probs.gather(-1, target).squeeze(-1)  # (B, seq_len)

            # mask padding
            mask = (input_ids != self.tokenizer.pad_token_id).float()
            avg_log_prob = (token_log_probs * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        return avg_log_prob

    # ------------------------------------------------------------------
    #  Internal helper
    # ------------------------------------------------------------------

    def _build_encoder_input(
        self,
        visual_embeddings: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Project visual embedding and prepend as prefix tokens to T5 encoder inputs.

        Returns:
            encoder_inputs_embeds : (B, num_vis_tokens + seq_len, t5_dim)
            encoder_attention_mask: (B, num_vis_tokens + seq_len)
        """
        B      = visual_embeddings.size(0)
        device = visual_embeddings.device

        # Project: (B, visual_dim) → (B, num_vis_tokens, t5_dim)
        proj       = self.visual_projection(visual_embeddings)        # (B, V*D)
        vis_prefix = proj.view(B, self.num_visual_tokens, -1)         # (B, V, D)

        # Text token embeddings: (B, seq_len, t5_dim)
        token_embeds = self.t5.shared(input_ids)

        # Concatenate prefix + text
        encoder_inputs_embeds = torch.cat([vis_prefix, token_embeds], dim=1)

        # Extend attention mask for visual prefix (always attended)
        vis_mask = torch.ones(B, self.num_visual_tokens, dtype=torch.long, device=device)
        encoder_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        return encoder_inputs_embeds, encoder_attention_mask

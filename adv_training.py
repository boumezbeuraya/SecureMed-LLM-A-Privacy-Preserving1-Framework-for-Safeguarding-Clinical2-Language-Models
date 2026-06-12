"""
Adversarial Fine-Tuning for SecureMed-LLM
==========================================
Implements the adversarial training pipeline described in Sections 5.1g
and 5.1h of the SecureMed-LLM paper.

Key design decisions (from paper):
  - Adversarial injection ratio : 5% of each training batch
  - Attack types                : FGSM, PGD, DeepFool (image-level)
                                  + BioMedAttack-LLM text-level prompts
  - Surrogate model             : ResNet18 (transfer attacks to BioMedCLIP)
  - Perturbation budget         : ε = 0.1 (L∞ norm, default)
  - Base checkpoint             : DP-SGD fine-tuned model
  - Optimizer                   : AdamW, lr = 2e-5 (same as DP phase)

Paper reference (Section 5.1g):
  "Adversarial samples constitute 5% of each training batch, selected by
   evaluating configurations of 3%, 5%, 7%, and 10% on the validation set.
   Lower rates (3%) yielded insufficient robustness improvement; higher rates
   (7–10%) degraded clean performance without commensurate gains."

Paper reference (Section 5.1h):
  "After fine-tuning, performance improves consistently across all adversarial
   attack types, indicating enhanced robustness of the model."

Training flow:
  1. Load DP-SGD checkpoint (base model).
  2. For each batch:
       a. Draw the full batch of clean (anonymized) samples.
       b. Randomly select 5% of samples to replace with adversarial variants.
       c. For each selected sample, randomly apply one of {FGSM, PGD, DeepFool}
          to the image, or use a pre-generated text adversarial report.
       d. Forward the mixed batch and compute cross-entropy loss.
       e. Backpropagate and update weights.
  3. Evaluate on clean and adversarial validation sets after each epoch.
  4. Save the best checkpoint (keyed on clean val BLEU-4).
"""

import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Internal imports
from src.adversarial.attack_generator import ImageAttackGenerator, BioMedAttackLLM

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Defaults matching Table 1 / Section 5.1
# ─────────────────────────────────────────────────────────────
DEFAULT_ADV_RATIO     = 0.05      # 5% adversarial injection per batch
DEFAULT_EPSILON       = 0.1       # L∞ perturbation budget
DEFAULT_PGD_STEPS     = 10        # PGD iterations
DEFAULT_LR            = 2e-5      # AdamW learning rate (same as DP phase)
DEFAULT_EPOCHS        = 5         # Fine-tuning epochs
DEFAULT_WEIGHT_DECAY  = 0.01
DEFAULT_GRAD_CLIP     = 1.0       # Gradient clipping norm (standard, not DP)
SUPPORTED_ATTACKS     = ["fgsm", "pgd", "deepfool"]


class AdversarialTrainer:
    """
    Fine-tunes the SecureMed-LLM pipeline on a mixture of clean and
    adversarially perturbed samples to improve robustness.

    Args:
        model           : SecureMedPipeline (BioMedCLIP + T5).
        train_loader    : DataLoader for the anonymized training split.
        val_loader      : DataLoader for the clean validation split.
        output_dir      : Directory to save adversarial checkpoints.
        adv_ratio       : Fraction of each batch to replace with adversarial
                          samples (paper: 0.05).
        attack_types    : Which image attacks to sample from per batch.
                          Defaults to all three (FGSM, PGD, DeepFool).
        epsilon         : L∞ image perturbation budget (paper: 0.1).
        pgd_steps       : PGD iteration count (paper: 10).
        learning_rate   : AdamW lr (paper: 2e-5).
        epochs          : Training epochs.
        grad_clip_norm  : Standard gradient clipping (not DP).
        text_attacker   : Optional BioMedAttackLLM for text-level injection.
                          If None, only image-level attacks are used.
        device          : Compute device.
    """

    def __init__(
        self,
        model:           nn.Module,
        train_loader:    DataLoader,
        val_loader:      DataLoader,
        output_dir:      str,
        adv_ratio:       float           = DEFAULT_ADV_RATIO,
        attack_types:    Optional[List[str]] = None,
        epsilon:         float           = DEFAULT_EPSILON,
        pgd_steps:       int             = DEFAULT_PGD_STEPS,
        learning_rate:   float           = DEFAULT_LR,
        epochs:          int             = DEFAULT_EPOCHS,
        grad_clip_norm:  float           = DEFAULT_GRAD_CLIP,
        text_attacker:   Optional[BioMedAttackLLM] = None,
        device:          Optional[torch.device] = None,
    ):
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.output_dir     = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.adv_ratio      = adv_ratio
        self.attack_types   = attack_types or SUPPORTED_ATTACKS
        self.epsilon        = epsilon
        self.pgd_steps      = pgd_steps
        self.learning_rate  = learning_rate
        self.epochs         = epochs
        self.grad_clip_norm = grad_clip_norm
        self.text_attacker  = text_attacker

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=DEFAULT_WEIGHT_DECAY,
        )

        # ImageAttackGenerator uses ResNet18 surrogate (paper Section 5.1g)
        self.image_attacker = ImageAttackGenerator(
            device=self.device,
            epsilon=self.epsilon,
        )

        logger.info(
            "AdversarialTrainer initialised | adv_ratio=%.0f%% | "
            "attacks=%s | ε=%.2f | lr=%.0e | epochs=%d",
            self.adv_ratio * 100, self.attack_types,
            self.epsilon, self.learning_rate, self.epochs,
        )

    # ─────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────

    def train(self) -> nn.Module:
        """
        Run the adversarial fine-tuning loop.

        Returns:
            Trained model with best checkpoint loaded.
        """
        best_val_loss   = float("inf")
        history: List[Dict] = []

        for epoch in range(1, self.epochs + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._validate_epoch(epoch)

            epoch_log = {
                "epoch":            epoch,
                "train_loss":       train_metrics["loss"],
                "train_adv_loss":   train_metrics["adv_loss"],
                "train_clean_loss": train_metrics["clean_loss"],
                "val_loss":         val_metrics["loss"],
            }
            history.append(epoch_log)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f (clean=%.4f, adv=%.4f) | "
                "val_loss=%.4f",
                epoch, self.epochs,
                train_metrics["loss"],
                train_metrics["clean_loss"],
                train_metrics["adv_loss"],
                val_metrics["loss"],
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self._save_checkpoint(epoch, val_metrics["loss"])

        logger.info(
            "Adversarial fine-tuning complete. Best val loss: %.4f", best_val_loss
        )
        self._load_best_checkpoint()
        self._save_training_history(history)
        return self.model

    # ─────────────────────────────────────────────────────────
    #  Training epoch
    # ─────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        One training epoch with mixed clean + adversarial batches.

        Per the paper, each batch has adv_ratio (5%) of its samples replaced
        by adversarially perturbed counterparts. The attack type is sampled
        uniformly from self.attack_types for each selected sample.

        Returns:
            Dict with 'loss', 'clean_loss', 'adv_loss'.
        """
        self.model.train()
        total_loss  = 0.0
        clean_loss  = 0.0
        adv_loss    = 0.0
        n_batches   = 0
        n_adv_steps = 0

        for batch in self.train_loader:
            images         = batch["image"].to(self.device)
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            B = images.size(0)

            # ── 1. Compute clean loss ────────────────────────
            self.optimizer.zero_grad()
            clean_out = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            c_loss = clean_out["loss"]
            clean_loss += c_loss.item()

            # ── 2. Build adversarial mini-batch ──────────────
            n_adv = max(1, math.ceil(B * self.adv_ratio))
            adv_indices = random.sample(range(B), k=n_adv)

            adv_images = images.clone()
            adv_images = self._perturb_selected(adv_images, adv_indices)

            adv_out = self.model(
                images=adv_images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            a_loss = adv_out["loss"]
            adv_loss += a_loss.item()
            n_adv_steps += 1

            # ── 3. Combined loss & backward ───────────────────
            # Weight adversarial loss by injection ratio so its contribution
            # is proportional to the fraction of adversarial samples.
            combined = (1.0 - self.adv_ratio) * c_loss + self.adv_ratio * a_loss
            combined.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm
            )
            self.optimizer.step()

            total_loss += combined.item()
            n_batches  += 1

        denom = max(n_batches, 1)
        return {
            "loss":       total_loss / denom,
            "clean_loss": clean_loss / denom,
            "adv_loss":   adv_loss   / max(n_adv_steps, 1),
        }

    # ─────────────────────────────────────────────────────────
    #  Validation epoch
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate on the clean validation split.

        Returns:
            Dict with 'loss'.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.val_loader:
            images         = batch["image"].to(self.device)
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            out = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            total_loss += out["loss"].item()
            n_batches  += 1

        return {"loss": total_loss / max(n_batches, 1)}

    # ─────────────────────────────────────────────────────────
    #  Adversarial evaluation (post-training)
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_adversarial_robustness(
        self,
        eval_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate the model under each attack type separately.

        Computes average cross-entropy loss for clean images and for each
        adversarial attack type. This mirrors the evaluation in Tables 4 & 5
        of the paper.

        Args:
            eval_loader: DataLoader for the test (or val) split.

        Returns:
            Dict with keys 'clean', 'fgsm', 'pgd', 'deepfool' mapping to
            average loss values.
        """
        self.model.eval()

        losses: Dict[str, float] = {
            "clean":    0.0,
            "fgsm":     0.0,
            "pgd":      0.0,
            "deepfool": 0.0,
        }
        n_batches = 0

        for batch in eval_loader:
            images         = batch["image"].to(self.device)
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Clean evaluation
            clean_out = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            losses["clean"] += clean_out["loss"].item()

            # Adversarial evaluations — run attacks inside no_grad context
            # NOTE: attacks require gradients internally; use torch.enable_grad()
            for attack_type in self.attack_types:
                with torch.enable_grad():
                    adv_imgs = self._apply_attack(images, attack_type)

                adv_out = self.model(
                    images=adv_imgs.detach(),
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                losses[attack_type] += adv_out["loss"].item()

            n_batches += 1

        denom = max(n_batches, 1)
        return {k: v / denom for k, v in losses.items()}

    # ─────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────

    def _perturb_selected(
        self,
        images: Tensor,
        indices: List[int],
    ) -> Tensor:
        """
        Replace images at `indices` with adversarially perturbed versions.
        Each selected sample independently draws a random attack type.

        Args:
            images : (B, 3, H, W) image batch (will be modified in-place clone).
            indices: List of batch indices to perturb.

        Returns:
            (B, 3, H, W) tensor with selected images perturbed.
        """
        adv_images = images.clone()

        for idx in indices:
            attack_type = random.choice(self.attack_types)
            single_img  = images[idx : idx + 1]           # (1, 3, H, W)

            with torch.enable_grad():
                perturbed = self._apply_attack(single_img, attack_type)

            adv_images[idx] = perturbed.detach().squeeze(0)

        return adv_images

    def _apply_attack(self, images: Tensor, attack_type: str) -> Tensor:
        """
        Apply the specified attack to a batch of images.

        Args:
            images     : (B, 3, H, W) images.
            attack_type: One of 'fgsm', 'pgd', 'deepfool'.

        Returns:
            (B, 3, H, W) perturbed images.
        """
        if attack_type == "fgsm":
            return self.image_attacker.fgsm(images)
        elif attack_type == "pgd":
            return self.image_attacker.pgd(images, steps=self.pgd_steps)
        elif attack_type == "deepfool":
            return self.image_attacker.deepfool(images)
        else:
            logger.warning("Unknown attack type '%s'; returning clean image.", attack_type)
            return images

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save the model state dict as the best adversarial checkpoint."""
        ckpt_path = self.output_dir / "best_adv_checkpoint.pt"
        torch.save(
            {
                "epoch":              epoch,
                "val_loss":           val_loss,
                "adv_ratio":          self.adv_ratio,
                "epsilon":            self.epsilon,
                "attack_types":       self.attack_types,
                "model_state_dict":   self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            ckpt_path,
        )
        logger.info(
            "Adversarial checkpoint saved → %s (epoch=%d, val_loss=%.4f)",
            ckpt_path, epoch, val_loss,
        )

    def _load_best_checkpoint(self) -> None:
        """Reload the best adversarial checkpoint into the model."""
        ckpt_path = self.output_dir / "best_adv_checkpoint.pt"
        if not ckpt_path.exists():
            logger.warning("No adversarial checkpoint found at %s.", ckpt_path)
            return
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        logger.info(
            "Best adversarial checkpoint loaded (epoch=%d, val_loss=%.4f)",
            state["epoch"], state["val_loss"],
        )

    def _save_training_history(self, history: List[Dict]) -> None:
        """Persist per-epoch training metrics as a JSON file."""
        import json
        hist_path = self.output_dir / "adv_training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info("Training history saved → %s", hist_path)


# ─────────────────────────────────────────────────────────────
#  Convenience function (used by scripts/train.py)
# ─────────────────────────────────────────────────────────────

def train_adversarial(
    model:          nn.Module,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    output_dir:     str,
    base_checkpoint: Optional[str] = None,
    adv_ratio:      float = DEFAULT_ADV_RATIO,
    attack_types:   Optional[List[str]] = None,
    epsilon:        float = DEFAULT_EPSILON,
    pgd_steps:      int   = DEFAULT_PGD_STEPS,
    learning_rate:  float = DEFAULT_LR,
    epochs:         int   = DEFAULT_EPOCHS,
    device:         Optional[torch.device] = None,
) -> nn.Module:
    """
    Convenience wrapper: optionally load a DP checkpoint, then run
    adversarial fine-tuning.

    Args:
        model            : SecureMedPipeline instance.
        train_loader     : Training DataLoader (anonymized data).
        val_loader       : Validation DataLoader.
        output_dir       : Directory for adversarial checkpoints.
        base_checkpoint  : Optional path to a DP-SGD checkpoint to start from.
        adv_ratio        : Adversarial injection ratio (paper: 0.05).
        attack_types     : Image attack types (default: all three).
        epsilon          : L∞ perturbation budget (paper: 0.1).
        pgd_steps        : PGD iterations (paper: 10).
        learning_rate    : AdamW learning rate (paper: 2e-5).
        epochs           : Fine-tuning epochs.
        device           : Compute device.

    Returns:
        Adversarially fine-tuned model.
    """
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Load DP checkpoint as the starting point if provided
    if base_checkpoint is not None:
        logger.info("Loading base DP checkpoint from %s …", base_checkpoint)
        state = torch.load(base_checkpoint, map_location=device)
        # Support both raw state dicts and DPTrainer-style checkpoint dicts
        sd = state.get("model_state_dict", state)
        model.load_state_dict(sd)
        logger.info("Base checkpoint loaded successfully.")

    trainer = AdversarialTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        adv_ratio=adv_ratio,
        attack_types=attack_types,
        epsilon=epsilon,
        pgd_steps=pgd_steps,
        learning_rate=learning_rate,
        epochs=epochs,
        device=device,
    )
    return trainer.train()


def load_adv_checkpoint(
    model:           nn.Module,
    checkpoint_path: str,
    device:          Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Load an adversarial fine-tuning checkpoint into a model.

    Args:
        model           : Target model (must match saved architecture).
        checkpoint_path : Path to .pt file saved by AdversarialTrainer.
        device          : Target device.

    Returns:
        Tuple of (model_with_weights, checkpoint_metadata).
    """
    device = device or torch.device("cpu")
    state  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    meta = {k: v for k, v in state.items() if k != "model_state_dict"}
    logger.info(
        "Adversarial checkpoint loaded from %s | epoch=%s | val_loss=%.4f",
        checkpoint_path, meta.get("epoch"), meta.get("val_loss", float("nan")),
    )
    return model, meta

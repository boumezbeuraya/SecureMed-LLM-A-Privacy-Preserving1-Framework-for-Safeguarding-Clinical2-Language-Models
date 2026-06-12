"""
Differentially Private Training (DP-SGD) via Opacus
=====================================================
Implements the privacy-preserving fine-tuning pipeline described in
Section 5.1i and Table 1 of the SecureMed-LLM paper.

Training configuration (from paper):
  - Optimizer     : AdamW, lr = 2e-5
  - Batch size    : 16
  - Epochs        : 5
  - Clipping norm : C = 1.0
  - Noise mult.   : σ = 1.1
  - Privacy budget: (ε = 3.0, δ = 1e-5), Rényi DP accountant
  - Adversarial injection ratio: 5% per batch

Paper reference (Section 5.1i):
  "DP-SGD clips per-sample gradients to norm C=1.0 and adds calibrated
   Gaussian noise with multiplier σ=1.1 at each training step. Training
   for 5 epochs yields (ε=3.0, δ=1e-5) under the Rényi DP accountant."
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Privacy budget defaults (paper Table 1)
# ─────────────────────────────────────────────────────────────
DEFAULT_EPSILON        = 3.0
DEFAULT_DELTA          = 1e-5
DEFAULT_NOISE_MULT     = 1.1
DEFAULT_MAX_GRAD_NORM  = 1.0
DEFAULT_LR             = 2e-5
DEFAULT_EPOCHS         = 5
DEFAULT_BATCH_SIZE     = 16


class DPTrainer:
    """
    Wraps Opacus PrivacyEngine to enable DP-SGD training of the
    SecureMed-LLM BioMedCLIP+T5 pipeline.

    The trainer:
      1. Attaches an Opacus PrivacyEngine to the model and optimizer.
      2. Runs the training loop with per-sample gradient clipping and
         calibrated Gaussian noise injection.
      3. Reports the cumulative privacy budget (ε, δ) after every epoch
         using the Rényi DP accountant (Opacus default).
      4. Saves checkpoints whenever validation BLEU improves.

    Args:
        model          : The SecureMedPipeline (BioMedCLIP + T5).
        train_loader   : DataLoader for the (anonymized) training split.
        val_loader     : DataLoader for the validation split.
        output_dir     : Directory to write checkpoints.
        noise_multiplier: DP-SGD noise multiplier σ (default 1.1).
        max_grad_norm  : Per-sample gradient clipping norm C (default 1.0).
        target_epsilon : Target ε (used only for logging; budget is
                         computed analytically by Opacus).
        target_delta   : Target δ (default 1e-5).
        learning_rate  : AdamW learning rate (default 2e-5).
        epochs         : Number of training epochs (default 5).
        device         : torch.device; falls back to cuda if available.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        noise_multiplier: float = DEFAULT_NOISE_MULT,
        max_grad_norm:    float = DEFAULT_MAX_GRAD_NORM,
        target_epsilon:   float = DEFAULT_EPSILON,
        target_delta:     float = DEFAULT_DELTA,
        learning_rate:    float = DEFAULT_LR,
        epochs:           int   = DEFAULT_EPOCHS,
        device:           Optional[torch.device] = None,
    ):
        self.model           = model
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.output_dir      = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.noise_multiplier = noise_multiplier
        self.max_grad_norm    = max_grad_norm
        self.target_epsilon   = target_epsilon
        self.target_delta     = target_delta
        self.learning_rate    = learning_rate
        self.epochs           = epochs

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

        # Optimizer (will be replaced by Opacus-wrapped version in train())
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        self._privacy_engine = None   # initialized lazily in train()

    # ─────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────

    def train(self) -> nn.Module:
        """
        Run the full DP-SGD training loop.

        Returns:
            The trained model (with best checkpoint loaded).
        """
        try:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator
        except ImportError:
            raise ImportError(
                "Opacus is required for DP-SGD training. "
                "Install with: pip install opacus"
            )

        # Opacus requires all BatchNorm layers to be replaced with
        # GroupNorm (or similar) before attaching the PrivacyEngine.
        logger.info("Validating model compatibility with Opacus …")
        if not ModuleValidator.is_valid(self.model):
            self.model = ModuleValidator.fix(self.model)
            logger.info("Model fixed for Opacus compatibility.")

        privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )
        self._privacy_engine = privacy_engine

        logger.info(
            "DP-SGD configured | noise_multiplier=%.2f | max_grad_norm=%.2f | "
            "target ε=%.1f | δ=%.0e",
            self.noise_multiplier, self.max_grad_norm,
            self.target_epsilon, self.target_delta,
        )

        best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss   = self._validate_epoch(epoch)

            # Report privacy budget consumed so far
            epsilon = privacy_engine.get_epsilon(self.target_delta)
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                "ε=%.4f (δ=%.0e)",
                epoch, self.epochs, train_loss, val_loss,
                epsilon, self.target_delta,
            )

            if epsilon > self.target_epsilon:
                logger.warning(
                    "Privacy budget exhausted: ε=%.4f > target ε=%.1f. "
                    "Stopping training.",
                    epsilon, self.target_epsilon,
                )
                break

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, epsilon)

        logger.info("Training complete. Best val loss: %.4f", best_val_loss)
        self._load_best_checkpoint()
        return self.model

    def get_epsilon(self) -> float:
        """Return the current accumulated privacy budget ε."""
        if self._privacy_engine is None:
            raise RuntimeError("PrivacyEngine not initialised. Call train() first.")
        return self._privacy_engine.get_epsilon(self.target_delta)

    # ─────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.train_loader:
            images        = batch["image"].to(self.device)
            input_ids     = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass through the full pipeline
            # The pipeline's forward() must accept these three arguments
            # and return a dict with a 'loss' key.
            outputs = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,        # teacher-forcing target
            )
            loss = outputs["loss"]
            loss.backward()

            # Opacus handles per-sample gradient clipping internally;
            # standard optimizer.step() applies the noisy update.
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.val_loader:
            images         = batch["image"].to(self.device)
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            total_loss += outputs["loss"].item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, epoch: int, val_loss: float, epsilon: float) -> None:
        """Save model state dict to output_dir/best_checkpoint.pt."""
        ckpt_path = self.output_dir / "best_checkpoint.pt"
        # Opacus wraps the model in GradSampleModule; access _module for
        # the original model state.
        state = {
            "epoch":     epoch,
            "val_loss":  val_loss,
            "epsilon":   epsilon,
            "delta":     self.target_delta,
            "model_state_dict": self._unwrap_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path)
        logger.info(
            "Checkpoint saved → %s (epoch=%d, val_loss=%.4f, ε=%.4f)",
            ckpt_path, epoch, val_loss, epsilon,
        )

    def _load_best_checkpoint(self) -> None:
        """Load the best checkpoint back into the model."""
        ckpt_path = self.output_dir / "best_checkpoint.pt"
        if not ckpt_path.exists():
            logger.warning("No checkpoint found at %s; keeping current weights.", ckpt_path)
            return
        state = torch.load(ckpt_path, map_location=self.device)
        self._unwrap_model().load_state_dict(state["model_state_dict"])
        logger.info(
            "Best checkpoint loaded (epoch=%d, val_loss=%.4f, ε=%.4f)",
            state["epoch"], state["val_loss"], state["epsilon"],
        )

    def _unwrap_model(self) -> nn.Module:
        """Return the underlying model, unwrapping Opacus GradSampleModule if needed."""
        # Opacus ≥ 1.0 wraps the model in GradSampleModule
        if hasattr(self.model, "_module"):
            return self.model._module
        return self.model


# ─────────────────────────────────────────────────────────────
#  Standalone training function (used by scripts/train.py)
# ─────────────────────────────────────────────────────────────

def train_with_dp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str,
    noise_multiplier: float = DEFAULT_NOISE_MULT,
    max_grad_norm:    float = DEFAULT_MAX_GRAD_NORM,
    target_epsilon:   float = DEFAULT_EPSILON,
    target_delta:     float = DEFAULT_DELTA,
    learning_rate:    float = DEFAULT_LR,
    epochs:           int   = DEFAULT_EPOCHS,
    device:           Optional[torch.device] = None,
) -> Tuple[nn.Module, float]:
    """
    Convenience function: construct a DPTrainer and run training.

    Args:
        model          : SecureMedPipeline instance.
        train_loader   : Training DataLoader (anonymized data).
        val_loader     : Validation DataLoader.
        output_dir     : Directory for checkpoints.
        noise_multiplier: DP-SGD noise multiplier (paper: 1.1).
        max_grad_norm  : Per-sample clip norm (paper: 1.0).
        target_epsilon : Target privacy budget ε (paper: 3.0).
        target_delta   : Target δ (paper: 1e-5).
        learning_rate  : AdamW learning rate (paper: 2e-5).
        epochs         : Training epochs (paper: 5).
        device         : Compute device.

    Returns:
        Tuple of (trained_model, final_epsilon).
    """
    trainer = DPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        learning_rate=learning_rate,
        epochs=epochs,
        device=device,
    )
    trained_model = trainer.train()
    final_epsilon = trainer.get_epsilon()
    logger.info("Final privacy budget: ε=%.4f (δ=%.0e)", final_epsilon, target_delta)
    return trained_model, final_epsilon


def load_dp_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, dict]:
    """
    Load a DP-SGD checkpoint into an existing model.

    Args:
        model           : Target model instance (architecture must match).
        checkpoint_path : Path to .pt checkpoint file saved by DPTrainer.
        device          : Target device.

    Returns:
        Tuple of (model_with_loaded_weights, checkpoint_metadata_dict).
    """
    device = device or torch.device("cpu")
    state  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    meta = {
        "epoch":    state.get("epoch"),
        "val_loss": state.get("val_loss"),
        "epsilon":  state.get("epsilon"),
        "delta":    state.get("delta"),
    }
    logger.info(
        "DP checkpoint loaded from %s | epoch=%s | ε=%s | δ=%s",
        checkpoint_path, meta["epoch"], meta["epsilon"], meta["delta"],
    )
    return model, meta

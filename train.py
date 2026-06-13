#!/usr/bin/env python3
"""
SecureMed-LLM Training Script
================================
Orchestrates the two-phase offline fine-tuning pipeline described in
Sections 5.1c, 5.1i, and 5.1g of the SecureMed-LLM paper.

Training phases (paper Section 5.1 / Table 1):
  Phase 1 — DP-SGD Fine-Tuning (Section 5.1i):
    Fine-tunes BioMedCLIP+T5 on the anonymized Open-I training split using
    Differentially Private Stochastic Gradient Descent (Opacus).
      - Optimizer      : AdamW, lr = 2e-5
      - Batch size     : 16
      - Epochs         : 5
      - Clipping norm  : C = 1.0
      - Noise mult.    : σ = 1.1
      - Privacy budget : (ε = 3.0, δ = 1e-5), Rényi DP accountant

  Phase 2 — Adversarial Fine-Tuning (Section 5.1g):
    Continues training from the DP-SGD checkpoint, injecting adversarially
    perturbed samples (FGSM, PGD, DeepFool) at a 5% ratio per batch.
      - Base model     : best DP-SGD checkpoint from Phase 1
      - Adv. ratio     : 5% of each batch
      - Attacks        : FGSM, PGD (ε=0.1, L∞), DeepFool
      - Epochs         : 5 (same as DP phase)

Both phases operate fully offline; no external API calls are made during
training (paper Section 5.1 — "all training and fine-tuning operations were
performed in an offline execution environment").

Usage:
  # Phase 1 only (DP-SGD):
  python scripts/train.py --config configs/default.yaml --phase dp

  # Phase 2 only (adversarial fine-tuning, requires a DP checkpoint):
  python scripts/train.py --config configs/default.yaml --phase adv \\
      --dp_checkpoint outputs/dp/best_checkpoint.pt

  # Full two-phase training:
  python scripts/train.py --config configs/default.yaml --phase both

  # Override any config key from the command line:
  python scripts/train.py --config configs/default.yaml --phase both \\
      --override training.epochs=3 differential_privacy.epsilon=2.0

Expected config.yaml structure (configs/default.yaml):
  data:
    train_dir:   data/open-i/train
    val_dir:     data/open-i/val
    num_workers: 4

  model:
    t5_model_name:     t5-base
    visual_embed_dim:  512
    num_visual_tokens: 1
    max_seq_length:    128
    freeze_encoder:    true
    freeze_decoder:    false

  training:
    batch_size:   16
    epochs:       5
    learning_rate: 2.0e-5
    grad_clip:    1.0

  differential_privacy:
    noise_multiplier: 1.1
    max_grad_norm:    1.0
    target_epsilon:   3.0
    target_delta:     1.0e-5

  adversarial:
    adv_ratio:    0.05
    epsilon:      0.1
    pgd_steps:    10
    attack_types: [fgsm, pgd, deepfool]

  output:
    dp_dir:  outputs/dp
    adv_dir: outputs/adv
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

# ── SecureMed-LLM internal imports ──────────────────────────────────────────
from src.utils.config import load_config, Config
from src.utils.data_loader import get_dataloader
from src.models.pipeline import SecureMedPipeline
from src.privacy.dp_training import train_with_dp, load_dp_checkpoint
from src.adversarial.adv_training import train_adversarial, load_adv_checkpoint

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("securemed.train")


# ─────────────────────────────────────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SecureMed-LLM: two-phase offline training (DP-SGD + adversarial).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )

    # Training phase selection
    parser.add_argument(
        "--phase",
        type=str,
        choices=["dp", "adv", "both"],
        default="both",
        help=(
            "Training phase to execute. "
            "'dp' = Phase 1 DP-SGD only; "
            "'adv' = Phase 2 adversarial only (requires --dp_checkpoint); "
            "'both' = run Phase 1 then Phase 2 sequentially."
        ),
    )

    # Optional starting checkpoint for Phase 2
    parser.add_argument(
        "--dp_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a DP-SGD checkpoint (.pt) to use as the starting model "
            "for Phase 2 adversarial fine-tuning. Required when --phase adv; "
            "when --phase both, the best Phase 1 checkpoint is used automatically."
        ),
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: 'cpu', 'cuda', 'cuda:0', etc. Auto-detected if omitted.",
    )

    # YAML overrides: key=value pairs (dot-separated nested keys)
    parser.add_argument(
        "--override",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Zero or more config overrides in KEY=VALUE format. "
            "Dot-separated keys address nested config fields. "
            "Example: --override training.epochs=3 differential_privacy.epsilon=2.0"
        ),
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args(argv)


def _parse_overrides(override_list: List[str]) -> Dict[str, str]:
    """Convert ['key=value', ...] list to a dict for load_config()."""
    overrides: Dict[str, str] = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(
                f"Override '{item}' is not in KEY=VALUE format. "
                "Example: training.epochs=10"
            )
        key, _, value = item.partition("=")
        overrides[key.strip()] = value.strip()
    return overrides


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_dataloaders(cfg: Config, tokenizer):
    """
    Construct train and validation DataLoaders from the config.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    data_cfg = cfg.data
    train_cfg = cfg.training

    train_loader = get_dataloader(
        split_dir=data_cfg.train_dir,
        tokenizer=tokenizer,
        batch_size=train_cfg.batch_size,
        image_resolution=224,               # BioMedCLIP expects 224×224 (Table 1)
        max_seq_length=cfg.model.max_seq_length,
        split="train",
        num_workers=data_cfg.num_workers,
        return_raw_text=False,
    )

    val_loader = get_dataloader(
        split_dir=data_cfg.val_dir,
        tokenizer=tokenizer,
        batch_size=train_cfg.batch_size,
        image_resolution=224,
        max_seq_length=cfg.model.max_seq_length,
        split="val",
        num_workers=data_cfg.num_workers,
        return_raw_text=False,
    )

    logger.info(
        "DataLoaders ready | train_batches=%d | val_batches=%d | batch_size=%d",
        len(train_loader), len(val_loader), train_cfg.batch_size,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1: DP-SGD fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def run_dp_phase(
    cfg: Config,
    model: SecureMedPipeline,
    train_loader,
    val_loader,
    device: torch.device,
) -> Path:
    """
    Execute Phase 1: DP-SGD fine-tuning via Opacus.

    Implements the training configuration from paper Table 1:
      noise_multiplier=1.1, max_grad_norm=1.0, ε=3.0, δ=1e-5, epochs=5.

    Args:
        cfg:          Loaded Config object.
        model:        SecureMedPipeline instance to train.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        device:       Compute device.

    Returns:
        Path to the best DP-SGD checkpoint file.
    """
    dp_cfg   = cfg.differential_privacy
    train_cfg = cfg.training
    out_dir  = Path(cfg.output.dp_dir)

    logger.info(
        "=" * 60 + "\n"
        "  Phase 1: DP-SGD Fine-Tuning\n"
        "  noise_mult=%.2f | grad_norm=%.1f | ε=%.1f | δ=%.0e | epochs=%d\n"
        + "=" * 60,
        dp_cfg.noise_multiplier, dp_cfg.max_grad_norm,
        dp_cfg.target_epsilon, dp_cfg.target_delta, train_cfg.epochs,
    )

    t0 = time.time()
    trained_model, final_epsilon = train_with_dp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(out_dir),
        noise_multiplier=dp_cfg.noise_multiplier,
        max_grad_norm=dp_cfg.max_grad_norm,
        target_epsilon=dp_cfg.target_epsilon,
        target_delta=dp_cfg.target_delta,
        learning_rate=train_cfg.learning_rate,
        epochs=train_cfg.epochs,
        device=device,
    )
    elapsed = time.time() - t0

    logger.info(
        "Phase 1 complete | final_ε=%.4f (δ=%.0e) | wall_time=%.1f s",
        final_epsilon, dp_cfg.target_delta, elapsed,
    )

    # Persist a summary alongside the checkpoint
    summary = {
        "phase": "dp",
        "final_epsilon": final_epsilon,
        "target_delta": dp_cfg.target_delta,
        "noise_multiplier": dp_cfg.noise_multiplier,
        "max_grad_norm": dp_cfg.max_grad_norm,
        "epochs": train_cfg.epochs,
        "wall_time_seconds": elapsed,
    }
    summary_path = out_dir / "dp_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("DP training summary written → %s", summary_path)

    best_ckpt = out_dir / "best_checkpoint.pt"
    return best_ckpt


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2: Adversarial fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def run_adv_phase(
    cfg: Config,
    model: SecureMedPipeline,
    train_loader,
    val_loader,
    device: torch.device,
    base_checkpoint: Optional[str] = None,
) -> Path:
    """
    Execute Phase 2: adversarial fine-tuning.

    Loads a DP-SGD checkpoint as the base model (paper Section 5.1g:
    "Base checkpoint: DP-SGD fine-tuned model"), then trains with a
    5% adversarial injection ratio using FGSM, PGD, and DeepFool.

    Args:
        cfg:             Loaded Config object.
        model:           SecureMedPipeline to fine-tune.
        train_loader:    Training DataLoader.
        val_loader:      Validation DataLoader.
        device:          Compute device.
        base_checkpoint: Path to DP-SGD checkpoint. If None, falls back to
                         cfg.output.dp_dir/best_checkpoint.pt.

    Returns:
        Path to the best adversarial checkpoint file.
    """
    adv_cfg  = cfg.adversarial
    train_cfg = cfg.training
    out_dir  = Path(cfg.output.adv_dir)

    # Resolve base checkpoint path
    if base_checkpoint is None:
        base_checkpoint = str(Path(cfg.output.dp_dir) / "best_checkpoint.pt")
    base_checkpoint = str(base_checkpoint)

    attack_types: List[str] = list(adv_cfg.attack_types) if hasattr(adv_cfg, "attack_types") \
        else ["fgsm", "pgd", "deepfool"]

    logger.info(
        "=" * 60 + "\n"
        "  Phase 2: Adversarial Fine-Tuning\n"
        "  base_ckpt=%s\n"
        "  adv_ratio=%.0f%% | ε=%.2f | attacks=%s | epochs=%d\n"
        + "=" * 60,
        base_checkpoint,
        adv_cfg.adv_ratio * 100, adv_cfg.epsilon, attack_types, train_cfg.epochs,
    )

    t0 = time.time()
    trained_model = train_adversarial(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(out_dir),
        base_checkpoint=base_checkpoint,
        adv_ratio=adv_cfg.adv_ratio,
        attack_types=attack_types,
        epsilon=adv_cfg.epsilon,
        pgd_steps=adv_cfg.pgd_steps,
        learning_rate=train_cfg.learning_rate,
        epochs=train_cfg.epochs,
        device=device,
    )
    elapsed = time.time() - t0

    logger.info(
        "Phase 2 complete | wall_time=%.1f s", elapsed,
    )

    # Persist summary
    summary = {
        "phase": "adversarial",
        "base_checkpoint": base_checkpoint,
        "adv_ratio": adv_cfg.adv_ratio,
        "epsilon": adv_cfg.epsilon,
        "pgd_steps": adv_cfg.pgd_steps,
        "attack_types": attack_types,
        "epochs": train_cfg.epochs,
        "wall_time_seconds": elapsed,
    }
    summary_path = out_dir / "adv_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Adversarial training summary written → %s", summary_path)

    best_ckpt = out_dir / "best_adv_checkpoint.pt"
    return best_ckpt


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    """
    Main training entry-point.

    Returns:
        0 on success, non-zero on error.
    """
    args = _parse_args(argv)

    # ── Load configuration ────────────────────────────────────────────────
    try:
        overrides = _parse_overrides(args.override or [])
        cfg = load_config(args.config, overrides=overrides or None)
    except FileNotFoundError as exc:
        logger.error("Configuration file not found: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("Invalid --override argument: %s", exc)
        return 1

    logger.info("Configuration loaded from %s", args.config)
    if overrides:
        logger.info("Active overrides: %s", overrides)

    # ── Reproducibility ───────────────────────────────────────────────────
    _set_seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA detected — using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.warning(
            "CUDA not available — training on CPU. "
            "DP-SGD training on CPU can be very slow. "
            "Consider using a GPU-enabled environment (Colab / Kaggle)."
        )

    # ── Validate phase-specific requirements ────────────────────────────
    if args.phase == "adv" and args.dp_checkpoint is None:
        # Allow falling back to default path; warn if it does not exist yet
        default_ckpt = Path(cfg.output.dp_dir) / "best_checkpoint.pt"
        if not default_ckpt.exists():
            logger.error(
                "Phase 'adv' requires a DP-SGD checkpoint. "
                "Provide --dp_checkpoint or run --phase dp first. "
                "Expected default path: %s (not found).",
                default_ckpt,
            )
            return 1

    # ── Tokenizer (T5 — consistent with paper Table 1) ───────────────────
    model_cfg = cfg.model
    logger.info("Loading T5 tokenizer: %s", model_cfg.t5_model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.t5_model_name)

    # ── DataLoaders ───────────────────────────────────────────────────────
    try:
        train_loader, val_loader = _build_dataloaders(cfg, tokenizer)
    except RuntimeError as exc:
        logger.error("Failed to build DataLoaders: %s", exc)
        return 1

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("Initialising SecureMedPipeline …")
    model = SecureMedPipeline(
        freeze_encoder=model_cfg.freeze_encoder,
        freeze_decoder=model_cfg.freeze_decoder,
        t5_model_name=model_cfg.t5_model_name,
        visual_embed_dim=model_cfg.visual_embed_dim,
        num_visual_tokens=model_cfg.num_visual_tokens,
        max_seq_length=model_cfg.max_seq_length,
        device=device,
    )

    # ── Training phases ───────────────────────────────────────────────────
    dp_checkpoint_path: Optional[Path] = None

    if args.phase in ("dp", "both"):
        try:
            dp_checkpoint_path = run_dp_phase(
                cfg=cfg,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
            )
        except Exception as exc:
            logger.exception("Phase 1 (DP-SGD) failed: %s", exc)
            return 1

    if args.phase in ("adv", "both"):
        # For 'both', the model returned from Phase 1 already carries the
        # best DP weights (train_with_dp reloads the best checkpoint).
        # For 'adv' standalone, load from the provided/default checkpoint.
        base_ckpt_for_adv = (
            str(dp_checkpoint_path)
            if (args.phase == "both" and dp_checkpoint_path is not None)
            else args.dp_checkpoint
        )
        try:
            adv_checkpoint_path = run_adv_phase(
                cfg=cfg,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                base_checkpoint=base_ckpt_for_adv,
            )
        except Exception as exc:
            logger.exception("Phase 2 (adversarial) failed: %s", exc)
            return 1

        logger.info("Final checkpoint: %s", adv_checkpoint_path)

    elif args.phase == "dp":
        logger.info("Final checkpoint: %s", dp_checkpoint_path)

    logger.info("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

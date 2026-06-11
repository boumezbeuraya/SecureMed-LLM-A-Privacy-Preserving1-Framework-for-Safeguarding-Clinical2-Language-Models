"""
Evaluation metrics for SecureMed-LLM.

Implements:
  - BLEU-1 and BLEU-4 (corpus-level, sacrebleu)
  - SSIM and PSNR (image quality after anonymisation)
  - MIA accuracy (membership inference attack)
  - PHI leakage rate (Presidio re-identification proxy)
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Text quality
# ─────────────────────────────────────────────

def compute_bleu(hypotheses: List[str], references: List[str]) -> dict:
    """
    Corpus-level BLEU-1 and BLEU-4 using sacrebleu.

    Args:
        hypotheses: List of generated report strings.
        references: List of ground-truth report strings (same length).

    Returns:
        dict with keys 'bleu1' and 'bleu4' (0–100 scale from sacrebleu).
    """
    try:
        import sacrebleu
    except ImportError:
        raise ImportError("Install sacrebleu: pip install sacrebleu")

    # sacrebleu expects references as a list of lists (one list per reference set)
    refs_wrapped = [references]

    bleu1 = sacrebleu.corpus_bleu(hypotheses, refs_wrapped, max_ngram_order=1).score
    bleu4 = sacrebleu.corpus_bleu(hypotheses, refs_wrapped, max_ngram_order=4).score

    # convert to 0-1 scale for consistency with paper notation
    return {"bleu1": bleu1 / 100.0, "bleu4": bleu4 / 100.0}


# ─────────────────────────────────────────────
#  Image quality
# ─────────────────────────────────────────────

def compute_ssim(original: np.ndarray, perturbed: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) between two images.

    Args:
        original:  HxWxC numpy array (uint8 or float in [0,1]).
        perturbed: HxWxC numpy array (same shape/dtype).

    Returns:
        SSIM value in [-1, 1] (1 = identical).
    """
    from skimage.metrics import structural_similarity as ssim

    # ensure float in [0,1]
    if original.dtype == np.uint8:
        original = original.astype(np.float64) / 255.0
        perturbed = perturbed.astype(np.float64) / 255.0

    if original.ndim == 3:
        return ssim(original, perturbed, channel_axis=2, data_range=1.0)
    return ssim(original, perturbed, data_range=1.0)


def compute_psnr(original: np.ndarray, perturbed: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in dB.

    Args:
        original:  HxWxC numpy array.
        perturbed: HxWxC numpy array (same shape).

    Returns:
        PSNR in dB. Returns inf if images are identical.
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr

    if original.dtype == np.uint8:
        original = original.astype(np.float64) / 255.0
        perturbed = perturbed.astype(np.float64) / 255.0

    return psnr(original, perturbed, data_range=1.0)


# ─────────────────────────────────────────────
#  Privacy — Membership Inference Attack
# ─────────────────────────────────────────────

def compute_mia_accuracy(
    model,
    member_loader,
    nonmember_loader,
    device: torch.device,
    threshold: float = 0.5,
) -> float:
    """
    Black-box confidence-thresholding membership inference attack (MIA).

    For each sample we compute the average token-level log-probability of the
    ground-truth report under the model.  Members (training samples) tend to
    have higher log-prob than non-members.  We sweep a threshold and report
    the balanced accuracy.

    Args:
        model: The SecureMedPipeline (must support compute_log_prob()).
        member_loader: DataLoader of training (member) samples.
        nonmember_loader: DataLoader of held-out (non-member) samples.
        device: torch.device.
        threshold: Decision threshold on log-prob score.

    Returns:
        MIA accuracy in [0, 1].
    """
    model.eval()
    member_scores = _collect_scores(model, member_loader, device)
    nonmember_scores = _collect_scores(model, nonmember_loader, device)

    # find threshold maximising accuracy (sweep over candidate thresholds)
    all_scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate(
        [np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    )

    best_acc = 0.0
    for thr in np.percentile(all_scores, np.linspace(0, 100, 200)):
        preds = (all_scores >= thr).astype(int)
        acc = (preds == labels).mean()
        best_acc = max(best_acc, acc)

    logger.info("MIA accuracy: %.4f (threshold sweep over %d samples)", best_acc, len(all_scores))
    return float(best_acc)


@torch.no_grad()
def _collect_scores(model, loader, device: torch.device) -> np.ndarray:
    """Collect per-sample average log-probability scores."""
    scores = []
    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        log_probs = model.compute_log_prob(images, input_ids, attention_mask)
        scores.extend(log_probs.cpu().numpy().tolist())
    return np.array(scores)


# ─────────────────────────────────────────────
#  Privacy — PHI Leakage Rate
# ─────────────────────────────────────────────

def compute_phi_leakage(
    generated_reports: List[str],
    original_reports: List[str],
    phi_entities: Optional[List[str]] = None,
) -> float:
    """
    PHI leakage rate: proportion of source PHI entities that survive in
    generated reports, measured via Presidio re-identification.

    This is an approximation of re-identification risk, not a formal guarantee.

    Args:
        generated_reports: List of model-generated report strings.
        original_reports:  List of corresponding original (un-anonymised) reports.
        phi_entities: Presidio entity types to check (default: paper's list).

    Returns:
        PHI leakage rate in [0, 1].
    """
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        raise ImportError("Install presidio-analyzer: pip install presidio-analyzer")

    if phi_entities is None:
        phi_entities = [
            "PERSON", "DATE_TIME", "LOCATION", "MEDICAL_LICENSE",
            "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN",
        ]

    analyzer = AnalyzerEngine()

    total_phi = 0
    leaked_phi = 0

    for orig, gen in zip(original_reports, generated_reports):
        # Extract PHI spans from original report
        orig_results = analyzer.analyze(text=orig, entities=phi_entities, language="en")
        source_values = {orig[r.start: r.end].lower() for r in orig_results}

        if not source_values:
            continue

        total_phi += len(source_values)

        # Check how many appear in generated report
        for val in source_values:
            if val in gen.lower():
                leaked_phi += 1

    if total_phi == 0:
        return 0.0

    rate = leaked_phi / total_phi
    logger.info("PHI leakage: %d/%d = %.4f", leaked_phi, total_phi, rate)
    return rate


# ─────────────────────────────────────────────
#  Cosine Similarity (image–text alignment)
# ─────────────────────────────────────────────

def cosine_similarity(a: Tensor, b: Tensor, dim: int = -1) -> Tensor:
    """Batch cosine similarity along specified dimension."""
    return torch.nn.functional.cosine_similarity(a, b, dim=dim)

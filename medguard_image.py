"""
Med-Guard Image Anonymization Module
======================================
Implements Gaussian noise-based image obfuscation for chest X-ray anonymisation,
as described in Sections 5.1a and 5.1b of the paper.

Paper reference:
  "Additive Gaussian noise with standard deviation σ is applied to pixel values
   of each chest X-ray. Its role is to reduce the visual identifiability of
   patient-specific features in a computationally lightweight manner."

  "σ=15 was selected as the operational configuration — SSIM ≈ 0.81,
   BLEU-4 = 0.70, PHI leakage ≈ 2.1% (full Med-Guard). Best empirical trade-off."

IMPORTANT — paper caveat (reproduced verbatim):
  "Gaussian noise injection does NOT provide formal re-identification guarantees
   equivalent to differential privacy or k-anonymity. The noise level selection
   is empirical and specific to this dataset."

All operations are executed locally on the clinician's device (Level 2 of the
SecureMed-LLM pipeline), before any data leaves the local environment.

Noise level selection results from the paper (Table S6 / Section 5.1b):
  σ= 5  → SSIM≈0.96, BLEU-4=0.82, PHI leakage≈15%  (insufficient)
  σ=15  → SSIM≈0.81, BLEU-4=0.70, PHI leakage≈2.1% (SELECTED)
  σ=25  → SSIM≈0.65, BLEU-4=0.58, PHI leakage≈6%
  σ=50  → SSIM≈0.41, BLEU-4=0.34, PHI leakage≈3%   (severe quality loss)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Paper-defined operational constants
# ---------------------------------------------------------------------------

SELECTED_SIGMA: int = 15          # operational noise level from paper Section 5.1b
SIGMA_CANDIDATES: List[int] = [5, 15, 25, 50]   # evaluated in ablation (Table S6)

# ImageNet normalisation statistics — used by BioMedCLIP preprocessing
IMAGENET_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, ...] = (0.229, 0.224, 0.225)


class MedGuardImageAnonymizer:
    """
    Gaussian noise-based image anonymiser for chest X-ray PHI obfuscation.

    Applies additive Gaussian noise N(0, σ²) to pixel values, clipping the
    result to the valid uint8 range [0, 255] to preserve image format
    compatibility downstream (BioMedCLIP preprocessing expects PIL Images).

    This module operates on raw (un-normalised) pixel values so that the
    noise magnitude σ is directly interpretable in the [0, 255] scale,
    consistent with SSIM/PSNR reporting in the paper.

    Args:
        sigma: Gaussian noise standard deviation in pixel units [0, 255].
               Paper selected σ=15 as the optimal privacy–utility trade-off.
        seed:  Optional random seed for reproducibility across evaluation runs.
               Set to None (default) for non-deterministic inference-time noise.
    """

    def __init__(
        self,
        sigma: float = SELECTED_SIGMA,
        seed:  Optional[int] = None,
    ):
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative; got {sigma}.")
        self.sigma = sigma
        self.seed  = seed
        self._rng  = np.random.default_rng(seed)

        logger.info(
            "MedGuardImageAnonymizer initialised | sigma=%.1f | seed=%s",
            self.sigma, self.seed,
        )

    # ------------------------------------------------------------------
    #  Core anonymisation API
    # ------------------------------------------------------------------

    def anonymize(self, image: Image.Image) -> Image.Image:
        """
        Apply Gaussian noise to a single PIL Image.

        Args:
            image: Input PIL Image (any mode; converted to RGB internally).

        Returns:
            Anonymised PIL Image (RGB, same size, uint8 pixel values).
        """
        img_rgb  = image.convert("RGB")
        img_arr  = np.array(img_rgb, dtype=np.float32)        # (H, W, 3) in [0, 255]
        noisy    = self._add_noise(img_arr)
        return Image.fromarray(noisy, mode="RGB")

    def anonymize_array(self, image_array: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian noise to a numpy image array.

        Args:
            image_array: HxW or HxWxC array.
                         - uint8  → treated as [0, 255] pixel values.
                         - float  → assumed [0, 1]; scaled to [0, 255] internally,
                                    noise applied, then re-scaled back to [0, 1].

        Returns:
            Numpy array of same shape and dtype as input.
        """
        float_input = image_array.dtype in (np.float32, np.float64)

        if float_input:
            arr = (image_array * 255.0).astype(np.float32)
        else:
            arr = image_array.astype(np.float32)

        noisy = self._add_noise(arr).astype(np.float32)

        if float_input:
            return (noisy / 255.0).astype(image_array.dtype)
        return noisy

    def anonymize_batch(
        self, images: List[Image.Image]
    ) -> List[Image.Image]:
        """
        Anonymise a list of PIL Images.

        Args:
            images: List of PIL Images.

        Returns:
            List of anonymised PIL Images (same length and order).
        """
        return [self.anonymize(img) for img in images]

    def anonymize_file(
        self,
        input_path:  Union[str, Path],
        output_path: Union[str, Path],
    ) -> None:
        """
        Load an image file, anonymise it, and save the result.

        Args:
            input_path:  Path to the source image (PNG, JPEG, …).
            output_path: Path to write the anonymised image.
                         Parent directories are created if needed.
        """
        input_path  = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image   = Image.open(input_path)
        noisy   = self.anonymize(image)
        noisy.save(output_path)
        logger.debug("Anonymised %s → %s", input_path.name, output_path.name)

    def anonymize_directory(
        self,
        input_dir:  Union[str, Path],
        output_dir: Union[str, Path],
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> int:
        """
        Batch-anonymise all images in a directory.

        Args:
            input_dir:  Source directory containing chest X-ray images.
            output_dir: Destination directory for anonymised images.
            extensions: File extensions to process.

        Returns:
            Number of images processed.
        """
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        for ext in extensions:
            for img_path in sorted(input_dir.glob(f"*{ext}")):
                out_path = output_dir / img_path.name
                self.anonymize_file(img_path, out_path)
                processed += 1

        logger.info(
            "Image anonymisation complete: %d files → %s", processed, output_dir
        )
        return processed

    # ------------------------------------------------------------------
    #  Quality metrics (paper Table S6 reproduced for validation)
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        original: Union[Image.Image, np.ndarray],
        anonymised: Union[Image.Image, np.ndarray],
    ) -> dict:
        """
        Compute SSIM and PSNR between original and anonymised images.

        Used to validate that σ=15 reproduces the paper's reported values:
          SSIM ≈ 0.81,  PSNR ≈ 20 dB  (approximate, dataset-dependent).

        Args:
            original:   Original PIL Image or numpy array.
            anonymised: Anonymised PIL Image or numpy array.

        Returns:
            Dict with keys 'ssim' and 'psnr'.
        """
        from src.utils.metrics import compute_ssim, compute_psnr

        orig_arr = _to_float_array(original)
        anon_arr = _to_float_array(anonymised)

        return {
            "ssim": compute_ssim(orig_arr, anon_arr),
            "psnr": compute_psnr(orig_arr, anon_arr),
        }

    def evaluate_noise_levels(
        self,
        image: Image.Image,
        sigmas: Optional[List[float]] = None,
    ) -> List[dict]:
        """
        Evaluate SSIM and PSNR for each candidate noise level.
        Reproduces the ablation study from paper Section 5.1b / Table S6.

        Args:
            image:  A reference PIL Image (single sample is sufficient).
            sigmas: List of σ values to evaluate.
                    Defaults to paper's candidates [5, 15, 25, 50].

        Returns:
            List of dicts: [{'sigma': σ, 'ssim': …, 'psnr': …}, …]
        """
        sigmas = sigmas or SIGMA_CANDIDATES
        results = []

        orig_arr = _to_float_array(image.convert("RGB"))

        for s in sigmas:
            temp_anon = MedGuardImageAnonymizer(sigma=s, seed=self.seed)
            anon_img  = temp_anon.anonymize(image)
            anon_arr  = _to_float_array(anon_img)

            from src.utils.metrics import compute_ssim, compute_psnr
            results.append({
                "sigma": s,
                "ssim":  compute_ssim(orig_arr, anon_arr),
                "psnr":  compute_psnr(orig_arr, anon_arr),
            })
            logger.info(
                "σ=%3d → SSIM=%.4f | PSNR=%.2f dB", s,
                results[-1]["ssim"], results[-1]["psnr"],
            )

        return results

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _add_noise(self, img_float: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise N(0, σ²) and clip to [0, 255].

        Args:
            img_float: Float32 array with values in [0, 255].

        Returns:
            uint8 array clipped to [0, 255].
        """
        if self.sigma == 0:
            return np.clip(img_float, 0, 255).astype(np.uint8)

        noise = self._rng.normal(loc=0.0, scale=self.sigma, size=img_float.shape)
        noisy = img_float + noise.astype(np.float32)
        return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
#  Tensor-level interface (used by data_loader and pipeline during training)
# ---------------------------------------------------------------------------

def anonymize_tensor(
    image_tensor,          # torch.Tensor  (C, H, W) normalised with ImageNet stats
    sigma: float = SELECTED_SIGMA,
    rng: Optional[np.random.Generator] = None,
):
    """
    Apply Gaussian noise to a normalised PyTorch image tensor (C, H, W).

    The tensor is expected to be normalised with ImageNet mean/std
    (as produced by data_loader.build_transform). Noise is added in
    un-normalised pixel space for σ interpretability, then re-normalised.

    Args:
        image_tensor: Normalised float tensor (C, H, W), values ≈ [-2, 2].
        sigma:        Noise level in pixel units (paper uses σ=15 out of 255).
        rng:          Optional numpy Generator for reproducibility.

    Returns:
        Noisy normalised tensor of the same shape and dtype.

    Note:
        σ=15 in pixel space [0, 255] ≈ σ_norm = 15/255 ≈ 0.059 in [0, 1] space.
        We apply noise in [0, 1] space after un-normalising for consistency
        with SSIM/PSNR evaluation.
    """
    import torch

    rng = rng or np.random.default_rng()

    mean = torch.tensor(IMAGENET_MEAN, dtype=image_tensor.dtype).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=image_tensor.dtype).view(3, 1, 1)

    # Un-normalise → [0, 1]
    img_01 = image_tensor * std + mean

    # Add noise in [0, 1] space  (σ / 255 converts pixel σ to [0,1] σ)
    sigma_01 = sigma / 255.0
    noise    = torch.from_numpy(
        rng.normal(0.0, sigma_01, size=image_tensor.shape).astype(np.float32)
    )
    noisy_01 = torch.clamp(img_01 + noise, 0.0, 1.0)

    # Re-normalise
    return (noisy_01 - mean) / std


# ---------------------------------------------------------------------------
#  Helper utilities
# ---------------------------------------------------------------------------

def _to_float_array(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Convert PIL Image or uint8/float array to float64 in [0, 1]."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.float64)
        return arr / 255.0
    if image.dtype == np.uint8:
        return image.astype(np.float64) / 255.0
    return image.astype(np.float64)


# ---------------------------------------------------------------------------
#  Convenience function
# ---------------------------------------------------------------------------

def anonymize_image(
    image: Image.Image,
    sigma: float = SELECTED_SIGMA,
    seed:  Optional[int] = None,
) -> Image.Image:
    """
    Stateless convenience wrapper for single-image anonymisation.

    Args:
        image: Input PIL Image.
        sigma: Gaussian noise standard deviation (pixel units, default=15).
        seed:  Optional random seed.

    Returns:
        Anonymised PIL Image.
    """
    return MedGuardImageAnonymizer(sigma=sigma, seed=seed).anonymize(image)


# ---------------------------------------------------------------------------
#  CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Med-Guard: anonymise chest X-ray images with Gaussian noise."
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Directory containing source .png/.jpg images.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write anonymised images.")
    parser.add_argument("--sigma",  type=float, default=SELECTED_SIGMA,
                        help=f"Gaussian noise σ (default: {SELECTED_SIGMA}).")
    parser.add_argument("--seed",   type=int,   default=None,
                        help="Random seed for reproducibility (default: None).")
    parser.add_argument("--evaluate_levels", action="store_true",
                        help="Run σ ablation on the first image and print metrics.")
    parser.add_argument("--metrics_json", default=None,
                        help="Optional path to write SSIM/PSNR metrics JSON.")
    args = parser.parse_args()

    anonymizer = MedGuardImageAnonymizer(sigma=args.sigma, seed=args.seed)
    n = anonymizer.anonymize_directory(args.input_dir, args.output_dir)
    logger.info("Processed %d images.", n)

    if args.evaluate_levels or args.metrics_json:
        sample_paths = sorted(Path(args.input_dir).glob("*.png"))
        if not sample_paths:
            sample_paths = sorted(Path(args.input_dir).glob("*.jpg"))
        if sample_paths:
            sample_img = Image.open(sample_paths[0])
            results    = anonymizer.evaluate_noise_levels(sample_img)
            print("\nNoise level ablation (reproduces paper Table S6):")
            print(f"  {'σ':>4}  {'SSIM':>7}  {'PSNR (dB)':>10}")
            for r in results:
                print(f"  {r['sigma']:>4}  {r['ssim']:>7.4f}  {r['psnr']:>10.2f}")

            if args.metrics_json:
                with open(args.metrics_json, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info("Metrics written to %s", args.metrics_json)

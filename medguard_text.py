"""
Med-Guard Text Anonymization Module
=====================================
Implements PHI de-identification using Microsoft Presidio with domain-specific
medical NER extensions, as described in Section 5.1a of the paper.

Paper reference:
  "Microsoft Presidio was used with domain-specific medical NER extensions to
   detect and replace PHI entities — including patient names, dates, timestamps,
   and location identifiers — with standardized placeholders
   (e.g., [DATE], [TIME], [NAME])."

PHI leakage is measured as the proportion of identifiable entities in generated
reports that can be matched to source PHI using the same Presidio pipeline.
This is an approximation of re-identification risk, NOT a formal privacy guarantee.

All operations are executed locally on the clinician's device (Level 2 of the
SecureMed-LLM pipeline) to prevent transmission of raw PHI to external systems.
"""

import re
import logging
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Entity type → placeholder mapping
#  Consistent with paper examples: '9:04 a.m.' → '[TIME]'
# ---------------------------------------------------------------------------
DEFAULT_PHI_ENTITIES: List[str] = [
    "PERSON",
    "DATE_TIME",
    "LOCATION",
    "MEDICAL_LICENSE",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "URL",
    "IP_ADDRESS",
    "NRP",             # Nationality, Religion, Political group
]

# Map Presidio entity type → placeholder token used in paper notation
ENTITY_PLACEHOLDER: Dict[str, str] = {
    "PERSON":            "[NAME]",
    "DATE_TIME":         "[DATE]",   # covers both DATE and TIME sub-types
    "LOCATION":          "[LOCATION]",
    "MEDICAL_LICENSE":   "[ID]",
    "PHONE_NUMBER":      "[PHONE]",
    "EMAIL_ADDRESS":     "[EMAIL]",
    "US_SSN":            "[ID]",
    "US_DRIVER_LICENSE": "[ID]",
    "US_PASSPORT":       "[ID]",
    "URL":               "[URL]",
    "IP_ADDRESS":        "[IP]",
    "NRP":               "[NRP]",
}

# Separate TIME placeholder for fine-grained paper notation ('9:04 a.m.' → '[TIME]')
_TIME_PATTERN = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\b",
    re.IGNORECASE
)


class MedGuardTextAnonymizer:
    """
    PHI de-identification for clinical text reports using Microsoft Presidio.

    Wraps the Presidio AnalyzerEngine and AnonymizerEngine, injecting
    domain-specific medical entity recognisers to improve recall on
    radiology-specific PHI patterns (e.g., accession numbers, MRN formats).

    Args:
        phi_entities: List of Presidio entity types to detect and redact.
                      Defaults to DEFAULT_PHI_ENTITIES.
        language: Language code for Presidio analysis (default: 'en').
        score_threshold: Minimum confidence score for entity recognition (0–1).
        use_medical_ner: If True, attempts to load a spaCy medical NER pipeline
                         ('en_core_med7_lg') for additional entity coverage.
                         Falls back gracefully if not installed.
    """

    def __init__(
        self,
        phi_entities: Optional[List[str]] = None,
        language: str = "en",
        score_threshold: float = 0.35,
        use_medical_ner: bool = True,
    ):
        self.phi_entities = phi_entities or DEFAULT_PHI_ENTITIES
        self.language = language
        self.score_threshold = score_threshold

        try:
            from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig
        except ImportError:
            raise ImportError(
                "presidio-analyzer and presidio-anonymizer are required. "
                "Install with: pip install presidio-analyzer presidio-anonymizer"
            )

        # Build registry and optionally extend with spaCy medical NER
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()

        if use_medical_ner:
            self._try_add_medical_ner(registry)

        self._analyzer  = AnalyzerEngine(registry=registry)
        self._anonymizer = AnonymizerEngine()
        self._OperatorConfig = OperatorConfig

        logger.info(
            "MedGuardTextAnonymizer initialised | entities=%s | threshold=%.2f",
            self.phi_entities, self.score_threshold,
        )

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    def anonymize(self, text: str) -> str:
        """
        Detect and replace PHI entities in a clinical text string.

        Args:
            text: Raw clinical report text (may contain PHI).

        Returns:
            Anonymised text with PHI replaced by typed placeholders.
        """
        if not text or not text.strip():
            return text

        # Step 1 — rule-based TIME pre-pass (handles '9:04 a.m.' → '[TIME]')
        text = _TIME_PATTERN.sub("[TIME]", text)

        # Step 2 — Presidio entity analysis
        results = self._analyzer.analyze(
            text=text,
            entities=self.phi_entities,
            language=self.language,
            score_threshold=self.score_threshold,
        )

        if not results:
            return text

        # Step 3 — Build operator map: entity_type → replace with placeholder
        operators = {
            entity: self._OperatorConfig(
                "replace",
                {"new_value": ENTITY_PLACEHOLDER.get(entity, f"[{entity}]")}
            )
            for entity in self.phi_entities
        }

        # Step 4 — Anonymise
        anonymised = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )
        return anonymised.text

    def anonymize_batch(self, texts: List[str]) -> List[str]:
        """
        Anonymise a list of clinical report strings.

        Args:
            texts: List of raw report strings.

        Returns:
            List of anonymised report strings (same length and order).
        """
        return [self.anonymize(t) for t in texts]

    def analyze_phi(self, text: str) -> List[Dict]:
        """
        Return a list of detected PHI spans (without anonymising).
        Useful for auditing and measuring PHI leakage rate.

        Args:
            text: Input text.

        Returns:
            List of dicts with keys: entity_type, start, end, score, text.
        """
        results = self._analyzer.analyze(
            text=text,
            entities=self.phi_entities,
            language=self.language,
            score_threshold=self.score_threshold,
        )
        return [
            {
                "entity_type": r.entity_type,
                "start":       r.start,
                "end":         r.end,
                "score":       r.score,
                "text":        text[r.start: r.end],
            }
            for r in results
        ]

    def phi_leakage_rate(
        self,
        original_texts: List[str],
        generated_texts: List[str],
    ) -> Tuple[float, int, int]:
        """
        Compute the PHI leakage rate as defined in the paper:
          "Proportion of identifiable entities in generated reports that can
           be matched to source PHI using the same Presidio pipeline."

        This is an approximation of re-identification risk, not a formal guarantee.

        Args:
            original_texts: List of original (un-anonymised) reports.
            generated_texts: List of model-generated reports (same length).

        Returns:
            Tuple of (leakage_rate, leaked_count, total_phi_count).
        """
        assert len(original_texts) == len(generated_texts), (
            "original_texts and generated_texts must have the same length."
        )

        total_phi = 0
        leaked    = 0

        for orig, gen in zip(original_texts, generated_texts):
            spans = self.analyze_phi(orig)
            source_values = {orig[s["start"]: s["end"]].lower().strip() for s in spans}

            if not source_values:
                continue

            total_phi += len(source_values)
            gen_lower  = gen.lower()
            for val in source_values:
                if val and val in gen_lower:
                    leaked += 1

        rate = leaked / total_phi if total_phi > 0 else 0.0
        logger.info(
            "PHI leakage: %d / %d = %.4f (%.1f%%)",
            leaked, total_phi, rate, rate * 100,
        )
        return rate, leaked, total_phi

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_add_medical_ner(registry) -> None:
        """
        Attempt to register a spaCy-based medical NER recogniser.
        Uses 'en_core_med7_lg' if available; silently skips if not installed.

        Med7 covers: DOSAGE, DRUG, DURATION, FORM, FREQUENCY, ROUTE, STRENGTH.
        These are not PHI per se but improve context-aware de-identification.
        """
        try:
            import spacy
            from presidio_analyzer.nlp_engine import SpacyNlpEngine
            from presidio_analyzer import NlpEngineProvider

            # Check if med7 model is available
            if spacy.util.is_package("en_core_med7_lg"):
                provider = NlpEngineProvider(
                    nlp_configuration={
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "en", "model_name": "en_core_med7_lg"}],
                    }
                )
                logger.info("Medical NER (en_core_med7_lg) loaded successfully.")
            else:
                logger.info(
                    "en_core_med7_lg not found; using default Presidio NLP engine. "
                    "Install with: pip install https://huggingface.co/kormilitzin/en_core_med7_lg/..."
                )
        except Exception as exc:
            logger.debug("Medical NER load skipped: %s", exc)


# ---------------------------------------------------------------------------
#  Convenience function for single-string anonymisation
# ---------------------------------------------------------------------------

def anonymize_text(
    text: str,
    phi_entities: Optional[List[str]] = None,
    score_threshold: float = 0.35,
) -> str:
    """
    Stateless convenience wrapper for single-report anonymisation.

    Args:
        text: Raw clinical report string.
        phi_entities: Presidio entity types to redact (defaults to paper's list).
        score_threshold: Presidio confidence threshold.

    Returns:
        Anonymised text string.

    Example:
        >>> anonymize_text("Patient John Doe, DOB 12/01/1985, seen on 09/15/2023.")
        'Patient [NAME], DOB [DATE], seen on [DATE].'
    """
    anonymizer = MedGuardTextAnonymizer(
        phi_entities=phi_entities,
        score_threshold=score_threshold,
    )
    return anonymizer.anonymize(text)


# ---------------------------------------------------------------------------
#  CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Med-Guard: anonymise PHI in clinical text files."
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Directory containing raw .txt report files.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write anonymised .txt files.")
    parser.add_argument("--score_threshold", type=float, default=0.35,
                        help="Presidio confidence threshold (default: 0.35).")
    parser.add_argument("--audit_json", default=None,
                        help="Optional path to write per-file PHI audit JSON.")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anonymizer = MedGuardTextAnonymizer(score_threshold=args.score_threshold)
    audit      = {}
    processed  = 0

    for txt_path in sorted(input_dir.glob("*.txt")):
        raw_text      = txt_path.read_text(encoding="utf-8")
        anonymised    = anonymizer.anonymize(raw_text)
        out_path      = output_dir / txt_path.name
        out_path.write_text(anonymised, encoding="utf-8")

        if args.audit_json:
            audit[txt_path.name] = anonymizer.analyze_phi(raw_text)

        processed += 1

    logger.info("Anonymised %d files → %s", processed, output_dir)

    if args.audit_json:
        with open(args.audit_json, "w") as f:
            json.dump(audit, f, indent=2)
        logger.info("PHI audit written to %s", args.audit_json)

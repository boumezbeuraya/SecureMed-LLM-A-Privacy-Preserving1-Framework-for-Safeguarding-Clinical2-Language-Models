"""
IDS-LLM: Multi-Stage Clinical Report Validation Pipeline
=========================================================
Implements the three-stage post-generation validation module described in
Section 5.1k of the SecureMed-LLM paper.

The IDS-LLM pipeline applies sequential safety checks to each generated
clinical report. Reports that fail any stage are rejected and trigger a
controlled regeneration request.

Three validation stages (Section 5.1k):
  1. Rule-Based Consistency Checking
     - 47 hand-crafted rules derived from chest X-ray reporting conventions
       and statistical patterns in the Open-I training corpus.
     - Detects: logical contradictions, forbidden terms (prompt injection
       indicators), structural violations.
     - Example: co-occurrence of "no pneumonia" and "consolidation consistent
       with pneumonia" → REJECT.

  2. Clinical Entity Verification
     - Medical entities extracted via SNOMED-CT radiology subset mapping.
     - Verifies anatomical plausibility and finding–impression consistency.

  3. Semantic Anomaly Detection
     - Reports encoded with Sentence-BERT (all-MiniLM-L6-v2).
     - Isolation Forest trained on clean training-set embeddings
       (contamination = 0.1, calibrated on validation set).
     - Reports with anomaly score above threshold → REJECT.

Paper performance (Table 8, held-out set: 400 clean + 100 corrupted):
  Rule-Based : Precision 95.1% | Recall 92.3% | F1 93.7% | AUC 0.93
  Clinical   : Precision 92.8% | Recall 90.2% | F1 91.5% | AUC 0.91
  Anomaly    : Precision 90.4% | Recall 87.6% | F1 89.0% | AUC 0.90
  Overall    : Precision 92.7% | Recall 90.1% | F1 91.3% | AUC 0.94

Paper reference (Section 5.1k):
  "Reports flagged by any stage are rejected and trigger regeneration."
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Embedding model (Section 5.1k)
# ─────────────────────────────────────────────────────────────
SBERT_MODEL_NAME      = "all-MiniLM-L6-v2"
ISOLATION_FOREST_CONTAMINATION = 0.1   # calibrated on validation set


# ═════════════════════════════════════════════════════════════
#  Data structures
# ═════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """
    Aggregated result from the three-stage IDS-LLM pipeline.

    Attributes:
        is_valid        : True if the report passed all three stages.
        rule_passed     : Result of stage 1 (rule-based check).
        clinical_passed : Result of stage 2 (clinical entity check).
        anomaly_passed  : Result of stage 3 (semantic anomaly check).
        rule_violations : List of rule IDs / descriptions that fired.
        clinical_issues : List of clinical consistency issues detected.
        anomaly_score   : Raw Isolation Forest anomaly score (lower = more anomalous).
        report          : The report text that was validated.
    """
    is_valid:         bool
    rule_passed:      bool
    clinical_passed:  bool
    anomaly_passed:   bool
    rule_violations:  List[str]     = field(default_factory=list)
    clinical_issues:  List[str]     = field(default_factory=list)
    anomaly_score:    float         = 0.0
    report:           str           = ""

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        stages = {
            "Rule":     "✓" if self.rule_passed     else "✗",
            "Clinical": "✓" if self.clinical_passed else "✗",
            "Anomaly":  "✓" if self.anomaly_passed  else "✗",
        }
        status = "VALID" if self.is_valid else "REJECTED"
        return (
            f"[{status}] Rule:{stages['Rule']} "
            f"Clinical:{stages['Clinical']} "
            f"Anomaly:{stages['Anomaly']} "
            f"(score={self.anomaly_score:.4f})"
        )


# ═════════════════════════════════════════════════════════════
#  Stage 1 — Rule-Based Consistency Checker
# ═════════════════════════════════════════════════════════════

# Built-in rule set (47 rules matching paper description).
# Each rule is a dict with:
#   id          : unique rule identifier
#   description : human-readable explanation
#   type        : 'contradiction' | 'forbidden' | 'structural'
#   check       : callable(text: str) -> bool  (True = violation detected)
#
# Rules are constructed programmatically below to keep the code readable.
# They are derived from published chest X-ray reporting guidelines and
# statistical patterns in the Open-I corpus (paper Section 5.1k).

def _build_contradiction_rules() -> List[Dict]:
    """
    Build rules that detect mutually exclusive finding pairs.
    These correspond to logical contradictions in radiology reports.
    """
    contradictions = [
        # Pneumonia / consolidation contradictions
        ("R001", "no pneumonia + consolidation",
         r"\bno\s+(sign[s]?\s+of\s+)?pneumonia\b",
         r"\bconsolidation\b"),
        ("R002", "lungs clear + pneumonia",
         r"\b(lungs?\s+(are\s+)?clear|clear\s+lungs?)\b",
         r"\bpneumonia\b"),
        ("R003", "no opacity + opacity present",
         r"\bno\s+(focal\s+)?opacit(y|ies)\b",
         r"\bopacit(y|ies)\s+(present|noted|seen|identified)\b"),
        # Pleural effusion contradictions
        ("R004", "no effusion + effusion present",
         r"\bno\s+(pleural\s+)?effusion\b",
         r"\bpleural\s+effusion\s+(present|noted|seen|identified)\b"),
        ("R005", "pleural spaces clear + effusion",
         r"\bpleural\s+space[s]?\s+(are\s+)?clear\b",
         r"\beffusion\b"),
        # Pneumothorax contradictions
        ("R006", "no pneumothorax + pneumothorax present",
         r"\bno\s+pneumothorax\b",
         r"\bpneumothorax\s+(is\s+)?(present|noted|seen|identified)\b"),
        # Cardiomegaly contradictions
        ("R007", "normal heart size + cardiomegaly",
         r"\b(normal|unremarkable)\s+(heart\s+size|cardiac\s+silhouette)\b",
         r"\bcardiomegaly\b"),
        ("R008", "no cardiomegaly + enlarged heart",
         r"\bno\s+cardiomegaly\b",
         r"\b(enlarged|increased)\s+(heart|cardiac)\b"),
        # Infiltrate contradictions
        ("R009", "no infiltrate + infiltrate present",
         r"\bno\s+infiltrate[s]?\b",
         r"\binfiltrate[s]?\s+(present|noted|seen|identified)\b"),
        ("R010", "no abnormality + pathology terms",
         r"\bno\s+(acute\s+)?abnormalit(y|ies)\b",
         r"\b(pneumonia|effusion|consolidation|infiltrate|opacity|mass|nodule)\b"),
        # Atelectasis contradictions
        ("R011", "no atelectasis + atelectasis present",
         r"\bno\s+atelectasis\b",
         r"\batelectasis\s+(is\s+)?(present|noted|seen|identified)\b"),
        # Edema contradictions
        ("R012", "no pulmonary edema + edema present",
         r"\bno\s+(pulmonary\s+)?edema\b",
         r"\bpulmonary\s+edema\s+(present|noted|seen|identified)\b"),
    ]

    rules = []
    for rule_id, desc, pat_a, pat_b in contradictions:
        _pa = re.compile(pat_a, re.IGNORECASE)
        _pb = re.compile(pat_b, re.IGNORECASE)
        rules.append({
            "id":          rule_id,
            "description": desc,
            "type":        "contradiction",
            "check":       lambda t, pa=_pa, pb=_pb: bool(pa.search(t) and pb.search(t)),
        })
    return rules


def _build_forbidden_rules() -> List[Dict]:
    """
    Build rules that detect prompt-injection indicators and forbidden terms.
    These cover patterns that should never appear in a genuine radiology report.
    """
    forbidden_patterns = [
        ("R013", "ignore previous instructions",
         r"\bignore\s+(previous|prior|all)\s+instructions?\b"),
        ("R014", "system prompt injection",
         r"\b(system\s*prompt|<\s*system\s*>|###\s*system)\b"),
        ("R015", "jailbreak attempt",
         r"\b(jailbreak|DAN\s+mode|developer\s+mode|god\s+mode)\b"),
        ("R016", "role override injection",
         r"\b(you\s+are\s+now|act\s+as|pretend\s+(you\s+are|to\s+be))\b"),
        ("R017", "instruction override",
         r"\b(disregard|forget|override)\s+(your|all|previous)\s+(instructions?|rules?|guidelines?)\b"),
        ("R018", "non-medical URL in report",
         r"https?://(?!.*(?:nih\.gov|pubmed|radiology|medical))"),
        ("R019", "personal data injection attempt",
         r"\b(SSN|social\s+security|credit\s+card|passport\s+number)\b"),
        ("R020", "executable code patterns",
         r"(<script|import\s+os|subprocess\.run|exec\s*\(|eval\s*\()"),
        ("R021", "prompt delimiter injection",
         r"(\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>)"),
        ("R022", "explicit override language",
         r"\b(STOP|HALT|OVERRIDE|BYPASS)\s+(ALL\s+)?(RULES?|FILTERS?|SAFETY)\b"),
        ("R023", "temperature/sampling manipulation",
         r"\b(temperature\s*=|top_p\s*=|max_tokens\s*=)\s*\d"),
        ("R024", "repetitive nonsense (likely attack artifact)",
         r"(.)\1{20,}"),   # 20+ repetitions of any character
        ("R025", "non-ASCII script injection",
         r"[\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF]{10,}"),  # unexpected scripts
    ]

    rules = []
    for rule_id, desc, pattern in forbidden_patterns:
        _p = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        rules.append({
            "id":          rule_id,
            "description": desc,
            "type":        "forbidden",
            "check":       lambda t, p=_p: bool(p.search(t)),
        })
    return rules


def _build_structural_rules() -> List[Dict]:
    """
    Build rules that enforce structural and length constraints
    consistent with chest X-ray report conventions.
    """
    rules = []

    # R026: Report is too short (fewer than 5 words) — likely incomplete/corrupted
    rules.append({
        "id":          "R026",
        "description": "report too short (< 5 words)",
        "type":        "structural",
        "check":       lambda t: len(t.split()) < 5,
    })

    # R027: Report is too long (> 500 words) — likely runaway generation
    rules.append({
        "id":          "R027",
        "description": "report too long (> 500 words)",
        "type":        "structural",
        "check":       lambda t: len(t.split()) > 500,
    })

    # R028: Report contains no anatomical terms (not a real radiology report)
    _anatomy_terms = re.compile(
        r"\b(lung|lobe|pleura|heart|cardiac|mediastin|diaphragm|"
        r"rib|clavicle|trachea|hilum|bronch|thorax|thoracic|chest|"
        r"spine|vertebra|costophrenic|cardiophrenic)\b",
        re.IGNORECASE,
    )
    rules.append({
        "id":          "R028",
        "description": "no anatomical terms found (not a radiology report)",
        "type":        "structural",
        "check":       lambda t: not bool(_anatomy_terms.search(t)),
    })

    # R029–R035: Finding–Impression coherence patterns
    coherence_checks = [
        ("R029", "bilateral + unilateral contradiction",
         r"\bbilateral\b", r"\b(unilateral|right\s+only|left\s+only)\b"),
        ("R030", "acute + chronic same finding (rare)",
         r"\bacute\s+and\s+chronic\b", r"\bnot\s+(consistent\s+with|suggestive\s+of)\b"),
        ("R031", "normal + abnormal in Impression",
         r"\b(Impression|IMPRESSION)\s*:\s*normal\b",
         r"\b(Impression|IMPRESSION)\s*:.*\b(abnormal|pathology|disease)\b"),
    ]
    for rule_id, desc, pat_a, pat_b in coherence_checks:
        _pa = re.compile(pat_a, re.IGNORECASE)
        _pb = re.compile(pat_b, re.IGNORECASE)
        rules.append({
            "id":          rule_id,
            "description": desc,
            "type":        "structural",
            "check":       lambda t, pa=_pa, pb=_pb: bool(pa.search(t) and pb.search(t)),
        })

    # R036–R047: Additional structural / formatting violations
    extra_structural = [
        ("R036", "multiple contradictory negations (≥3 'no X' then X found)",
         lambda t: (
             len(re.findall(r"\bno\s+\w+", t, re.IGNORECASE)) >= 3
             and bool(re.search(r"\b(present|identified|noted|seen)\b", t, re.IGNORECASE))
         )),
        ("R037", "repeated identical sentence (copy-paste artifact)",
         lambda t: (
             len(set(s.strip() for s in re.split(r"[.!?]", t) if len(s.strip()) > 15))
             < len([s for s in re.split(r"[.!?]", t) if len(s.strip()) > 15]) * 0.7
         )),
        ("R038", "missing period/full stop (malformed generation)",
         lambda t: "." not in t and len(t) > 50),
        ("R039", "all uppercase (shouting / injection artifact)",
         lambda t: t.isupper() and len(t) > 30),
        ("R040", "all lowercase with no punctuation (malformed)",
         lambda t: t.islower() and "." not in t and len(t) > 80),
        ("R041", "bracket mismatch (template artifact)",
         lambda t: t.count("[") != t.count("]")),
        ("R042", "unresolved placeholder token in output",
         lambda t: bool(re.search(r"\[(NAME|DATE|TIME|LOCATION|ID|PHONE|EMAIL)\]", t))),
        ("R043", "numeric-only content (not a narrative report)",
         lambda t: bool(re.fullmatch(r"[\d\s.,;:\-]+", t.strip())) and len(t) > 10),
        ("R044", "unrealistic numerical value (e.g., age > 130)",
         lambda t: bool(re.search(r"\b(1[3-9]\d|[2-9]\d{2})\s*-?\s*(year|yr)s?\s*old\b", t,
                                  re.IGNORECASE))),
        ("R045", "mentions patient name (PHI leakage indicator)",
         lambda t: bool(re.search(r"\b(patient\s+)?[A-Z][a-z]+\s+[A-Z][a-z]+\b", t))
                   and not bool(re.search(r"\b(Chest|Lung|Right|Left|No|The|This|There)\b",
                                          t.split()[0] if t.split() else "", re.IGNORECASE))),
        ("R046", "contains explicit date (PHI leakage indicator)",
         lambda t: bool(re.search(
             r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{2}[/\-]\d{2})\b", t
         ))),
        ("R047", "empty report",
         lambda t: len(t.strip()) == 0),
    ]
    for rule_id, desc, check_fn in extra_structural:
        rules.append({
            "id":          rule_id,
            "description": desc,
            "type":        "structural",
            "check":       check_fn,
        })

    return rules


def build_default_rule_set() -> List[Dict]:
    """
    Assemble the complete 47-rule set (12 contradiction + 13 forbidden
    + 22 structural) used in the IDS-LLM validation module.
    """
    rules = (
        _build_contradiction_rules()   # R001–R012  (12 rules)
        + _build_forbidden_rules()     # R013–R025  (13 rules)
        + _build_structural_rules()    # R026–R047  (22 rules)
    )
    assert len(rules) == 47, f"Expected 47 rules, got {len(rules)}"
    return rules


class RuleBasedChecker:
    """
    Stage 1: Rule-based consistency and safety checker.

    Applies the 47-rule set to a generated report and returns a list of
    any rules that fired (violations). An empty violation list = pass.

    Args:
        rules: List of rule dicts (default: build_default_rule_set()).
        rule_file: Optional path to a JSON file with additional custom rules
                   (loaded and merged with the default set).
    """

    def __init__(
        self,
        rules: Optional[List[Dict]] = None,
        rule_file: Optional[str] = None,
    ):
        self.rules = rules or build_default_rule_set()

        if rule_file is not None:
            self._load_json_rules(rule_file)

        logger.info("RuleBasedChecker initialised with %d rules.", len(self.rules))

    def check(self, report: str) -> Tuple[bool, List[str]]:
        """
        Apply all rules to a report.

        Args:
            report: Generated report text to validate.

        Returns:
            Tuple of (passed: bool, violations: List[str]).
            passed=True means no rules fired.
        """
        violations = []
        for rule in self.rules:
            try:
                if rule["check"](report):
                    violations.append(f"{rule['id']}: {rule['description']}")
            except Exception as exc:
                logger.debug("Rule %s raised exception: %s", rule["id"], exc)

        passed = len(violations) == 0
        return passed, violations

    def _load_json_rules(self, rule_file: str) -> None:
        """
        Load additional regex-based rules from a JSON file.

        Expected JSON format:
        [
          {
            "id": "RX01",
            "description": "custom rule description",
            "type": "forbidden",
            "pattern": "regex pattern string"
          }, ...
        ]
        """
        path = Path(rule_file)
        if not path.exists():
            logger.warning("Rule file not found: %s; skipping.", rule_file)
            return
        with open(path) as f:
            json_rules = json.load(f)
        for r in json_rules:
            _p = re.compile(r["pattern"], re.IGNORECASE)
            self.rules.append({
                "id":          r["id"],
                "description": r["description"],
                "type":        r.get("type", "custom"),
                "check":       lambda t, p=_p: bool(p.search(t)),
            })
        logger.info("Loaded %d additional rules from %s.", len(json_rules), rule_file)


# ═════════════════════════════════════════════════════════════
#  Stage 2 — Clinical Entity Verifier
# ═════════════════════════════════════════════════════════════

# SNOMED-CT radiology subset used for entity extraction (paper Section 5.1k).
# Maps canonical clinical terms → entity categories.
# This is a curated subset; a full SNOMED-CT integration would require the
# official SNOMED-CT release (licensed separately).
_SNOMED_RADIOLOGY_TERMS: Dict[str, List[str]] = {
    "finding": [
        "pneumonia", "consolidation", "atelectasis", "effusion",
        "pneumothorax", "cardiomegaly", "edema", "infiltrate",
        "opacity", "mass", "nodule", "lesion", "calcification",
        "fibrosis", "emphysema", "bronchiectasis", "adenopathy",
        "hernia", "fracture", "dislocation",
    ],
    "anatomy": [
        "lung", "lobe", "pleura", "heart", "mediastinum",
        "diaphragm", "rib", "clavicle", "trachea", "hilum",
        "bronchus", "thorax", "spine", "vertebra", "costophrenic",
        "cardiophrenic", "aorta", "pulmonary",
    ],
    "laterality": ["right", "left", "bilateral", "unilateral"],
    "severity": [
        "mild", "moderate", "severe", "subtle", "extensive",
        "minimal", "small", "large", "trace",
    ],
    "qualifier": [
        "acute", "chronic", "stable", "new", "improved",
        "worsened", "unchanged", "developing", "resolving",
    ],
}

# Implausible co-occurrence pairs (finding, anatomy) — anatomically impossible
_IMPOSSIBLE_COMBINATIONS: List[Tuple[str, str]] = [
    ("pneumothorax", "heart"),       # pneumothorax is not in the heart
    ("cardiomegaly", "lung"),        # cardiomegaly is a cardiac finding
    ("consolidation", "spine"),      # consolidation does not occur in spine
    ("atelectasis", "rib"),          # atelectasis is not a rib finding
]


class ClinicalEntityVerifier:
    """
    Stage 2: Clinical entity extraction and plausibility verification.

    Extracts medical entities using a curated SNOMED-CT radiology subset
    and verifies:
      (a) Anatomical plausibility: each finding is associated with a
          compatible anatomical structure.
      (b) Finding–Impression consistency: findings mentioned in the
          Findings section are reflected in the Impression section.

    Args:
        snomed_terms: Optional custom SNOMED term mapping.
                      Defaults to _SNOMED_RADIOLOGY_TERMS.
    """

    def __init__(
        self,
        snomed_terms: Optional[Dict[str, List[str]]] = None,
    ):
        self.snomed_terms = snomed_terms or _SNOMED_RADIOLOGY_TERMS
        # Pre-compile patterns for efficiency
        self._finding_pattern  = self._compile_pattern("finding")
        self._anatomy_pattern  = self._compile_pattern("anatomy")
        self._severity_pattern = self._compile_pattern("severity")

    def check(self, report: str) -> Tuple[bool, List[str]]:
        """
        Verify clinical entities in a report.

        Args:
            report: Generated report text.

        Returns:
            Tuple of (passed: bool, issues: List[str]).
        """
        issues = []

        # (a) Impossible finding–anatomy combinations
        anatomy_issues = self._check_impossible_combinations(report)
        issues.extend(anatomy_issues)

        # (b) Finding–Impression consistency
        consistency_issues = self._check_impression_consistency(report)
        issues.extend(consistency_issues)

        # (c) Verify at least one finding or anatomy term is present
        if not self._has_clinical_content(report):
            issues.append("Report contains no recognisable clinical findings or anatomy.")

        passed = len(issues) == 0
        return passed, issues

    def extract_entities(self, report: str) -> Dict[str, List[str]]:
        """
        Extract all recognised clinical entities from a report.

        Returns:
            Dict mapping category → list of matched terms.
        """
        result = {}
        for category, terms in self.snomed_terms.items():
            matched = [
                t for t in terms
                if re.search(r"\b" + re.escape(t) + r"\b", report, re.IGNORECASE)
            ]
            result[category] = matched
        return result

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _compile_pattern(self, category: str) -> re.Pattern:
        terms = self.snomed_terms.get(category, [])
        escaped = [re.escape(t) for t in terms]
        return re.compile(
            r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE
        ) if escaped else re.compile(r"(?!)")

    def _check_impossible_combinations(self, report: str) -> List[str]:
        """Flag anatomically implausible finding+anatomy co-occurrences."""
        issues = []
        for finding, wrong_anatomy in _IMPOSSIBLE_COMBINATIONS:
            if (re.search(r"\b" + re.escape(finding) + r"\b", report, re.IGNORECASE)
                    and re.search(r"\b" + re.escape(wrong_anatomy) + r"\b", report, re.IGNORECASE)):
                # Heuristic: check if they appear within 15 words of each other
                tokens = report.lower().split()
                try:
                    fi = next(i for i, t in enumerate(tokens) if finding in t)
                    ai = next(i for i, t in enumerate(tokens) if wrong_anatomy in t)
                    if abs(fi - ai) < 15:
                        issues.append(
                            f"Implausible combination: '{finding}' near '{wrong_anatomy}'."
                        )
                except StopIteration:
                    pass
        return issues

    def _check_impression_consistency(self, report: str) -> List[str]:
        """
        Verify that major findings in the Findings section are echoed
        in the Impression section (or that no Impression section contradicts them).
        """
        issues = []

        # Split report into Findings and Impression sections (if present)
        findings_match   = re.search(
            r"\b(findings?)\s*:\s*(.*?)(?=\b(impression|conclusion|summary)\s*:|$)",
            report, re.IGNORECASE | re.DOTALL,
        )
        impression_match = re.search(
            r"\b(impression|conclusion|summary)\s*:\s*(.*?)$",
            report, re.IGNORECASE | re.DOTALL,
        )

        if findings_match and impression_match:
            findings_text   = findings_match.group(2)
            impression_text = impression_match.group(2)

            # Extract finding entities from each section
            findings_entities   = set(self._finding_pattern.findall(findings_text.lower()))
            impression_entities = set(self._finding_pattern.findall(impression_text.lower()))

            # Major findings in Findings but explicitly negated in Impression
            for entity in findings_entities:
                neg_pattern = re.compile(
                    r"\bno\s+" + re.escape(entity) + r"\b", re.IGNORECASE
                )
                if neg_pattern.search(impression_text):
                    issues.append(
                        f"Finding '{entity}' in Findings section but negated in Impression."
                    )

        return issues

    def _has_clinical_content(self, report: str) -> bool:
        """Return True if at least one finding or anatomy term is present."""
        return bool(
            self._finding_pattern.search(report)
            or self._anatomy_pattern.search(report)
        )


# ═════════════════════════════════════════════════════════════
#  Stage 3 — Semantic Anomaly Detector
# ═════════════════════════════════════════════════════════════

class SemanticAnomalyDetector:
    """
    Stage 3: Sentence-BERT + Isolation Forest anomaly detection.

    Reports are embedded with Sentence-BERT (all-MiniLM-L6-v2) and
    evaluated by an Isolation Forest trained on clean training-set reports.
    Reports with anomaly scores below the decision threshold are flagged.

    The Isolation Forest contamination parameter is set to 0.1 (calibrated
    on the validation set, per Section 5.1k).

    Args:
        sbert_model_name : Sentence-BERT model (default: all-MiniLM-L6-v2).
        contamination    : Isolation Forest contamination (paper: 0.1).
        threshold        : Anomaly score decision threshold.
                           Reports with score < threshold are flagged.
                           If None, the fitted Isolation Forest's internal
                           decision function is used directly.
    """

    def __init__(
        self,
        sbert_model_name: str = SBERT_MODEL_NAME,
        contamination:    float = ISOLATION_FOREST_CONTAMINATION,
        threshold:        Optional[float] = None,
    ):
        self.sbert_model_name = sbert_model_name
        self.contamination    = contamination
        self.threshold        = threshold

        self._sbert     = None   # lazy-loaded
        self._iso_forest = None  # fitted in fit()
        self._is_fitted  = False

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    def fit(self, clean_reports: List[str]) -> "SemanticAnomalyDetector":
        """
        Fit the Isolation Forest on embeddings of clean training reports.

        Args:
            clean_reports: List of clean (unmodified) report strings from
                           the training set.

        Returns:
            self (for chaining).
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError("scikit-learn is required. pip install scikit-learn")

        logger.info(
            "Fitting Isolation Forest on %d clean reports (contamination=%.2f) …",
            len(clean_reports), self.contamination,
        )
        embeddings = self._embed(clean_reports)

        self._iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
        )
        self._iso_forest.fit(embeddings)
        self._is_fitted = True
        logger.info("Isolation Forest fitted.")
        return self

    def check(self, report: str) -> Tuple[bool, float]:
        """
        Evaluate a single report for semantic anomalies.

        Args:
            report: Generated report text.

        Returns:
            Tuple of (passed: bool, anomaly_score: float).
            passed=True means the report is NOT anomalous.
            anomaly_score: higher (less negative) = more normal.
        """
        if not self._is_fitted:
            logger.warning(
                "Isolation Forest not fitted; defaulting to pass. "
                "Call fit() with clean training reports first."
            )
            return True, 0.0

        embedding      = self._embed([report])              # (1, D)
        score          = self._iso_forest.decision_function(embedding)[0]
        prediction     = self._iso_forest.predict(embedding)[0]  # 1=normal, -1=anomaly

        # prediction == 1 → normal (pass); -1 → anomaly (fail)
        passed = bool(prediction == 1)

        if self.threshold is not None:
            passed = score >= self.threshold

        return passed, float(score)

    def check_batch(
        self, reports: List[str]
    ) -> List[Tuple[bool, float]]:
        """
        Evaluate a batch of reports.

        Args:
            reports: List of report strings.

        Returns:
            List of (passed, anomaly_score) tuples.
        """
        if not self._is_fitted:
            return [(True, 0.0)] * len(reports)

        embeddings  = self._embed(reports)
        scores      = self._iso_forest.decision_function(embeddings)
        predictions = self._iso_forest.predict(embeddings)

        results = []
        for pred, score in zip(predictions, scores):
            if self.threshold is not None:
                passed = float(score) >= self.threshold
            else:
                passed = bool(pred == 1)
            results.append((passed, float(score)))
        return results

    def save(self, path: str) -> None:
        """Persist the fitted Isolation Forest to disk (joblib)."""
        import joblib
        if not self._is_fitted:
            raise RuntimeError("Cannot save: Isolation Forest is not fitted.")
        joblib.dump(self._iso_forest, path)
        logger.info("Isolation Forest saved to %s", path)

    def load(self, path: str) -> "SemanticAnomalyDetector":
        """Load a previously fitted Isolation Forest from disk."""
        import joblib
        self._iso_forest = joblib.load(path)
        self._is_fitted  = True
        logger.info("Isolation Forest loaded from %s", path)
        return self

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _get_sbert(self):
        """Lazy-load Sentence-BERT model."""
        if self._sbert is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "pip install sentence-transformers"
                )
            logger.info("Loading Sentence-BERT: %s …", self.sbert_model_name)
            self._sbert = SentenceTransformer(self.sbert_model_name)
        return self._sbert

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using Sentence-BERT. Returns (N, D) array."""
        sbert = self._get_sbert()
        return sbert.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


# ═════════════════════════════════════════════════════════════
#  Integrated IDS-LLM Pipeline
# ═════════════════════════════════════════════════════════════

class IDSLLMValidator:
    """
    Full three-stage IDS-LLM validation pipeline.

    Applies stages sequentially:
      1. Rule-based consistency check  (RuleBasedChecker)
      2. Clinical entity verification  (ClinicalEntityVerifier)
      3. Semantic anomaly detection    (SemanticAnomalyDetector)

    A report is VALID only if it passes all three stages.
    On failure, the pipeline records which stage(s) failed and why,
    enabling targeted regeneration feedback.

    Args:
        rule_file           : Optional path to a custom JSON rules file.
        sbert_model_name    : Sentence-BERT model name.
        contamination       : Isolation Forest contamination (paper: 0.1).
        anomaly_threshold   : Score threshold (None = use Isolation Forest
                              internal decision boundary).
    """

    def __init__(
        self,
        rule_file:        Optional[str] = None,
        sbert_model_name: str   = SBERT_MODEL_NAME,
        contamination:    float = ISOLATION_FOREST_CONTAMINATION,
        anomaly_threshold: Optional[float] = None,
    ):
        self.rule_checker   = RuleBasedChecker(rule_file=rule_file)
        self.clinical_verifier = ClinicalEntityVerifier()
        self.anomaly_detector  = SemanticAnomalyDetector(
            sbert_model_name=sbert_model_name,
            contamination=contamination,
            threshold=anomaly_threshold,
        )
        logger.info("IDSLLMValidator initialised (3-stage pipeline).")

    # ------------------------------------------------------------------
    #  Fitting (must be called before validate() for Stage 3)
    # ------------------------------------------------------------------

    def fit_anomaly_detector(self, clean_reports: List[str]) -> None:
        """
        Fit the Isolation Forest on clean training reports.
        Must be called before validate() for Stage 3 to be active.

        Args:
            clean_reports: List of clean report strings from the training set.
        """
        self.anomaly_detector.fit(clean_reports)

    def save_anomaly_model(self, path: str) -> None:
        """Persist the fitted Isolation Forest to disk."""
        self.anomaly_detector.save(path)

    def load_anomaly_model(self, path: str) -> None:
        """Load a pre-fitted Isolation Forest from disk."""
        self.anomaly_detector.load(path)

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def validate(self, report: str) -> ValidationResult:
        """
        Run all three validation stages on a single report.

        Args:
            report: Generated clinical report text.

        Returns:
            ValidationResult with per-stage outcomes and overall verdict.
        """
        # Stage 1 — Rule-based
        rule_passed, rule_violations = self.rule_checker.check(report)
        if not rule_passed:
            logger.debug(
                "Stage 1 FAILED: %d rule violation(s): %s",
                len(rule_violations), rule_violations[:3],
            )

        # Stage 2 — Clinical entity verification
        clinical_passed, clinical_issues = self.clinical_verifier.check(report)
        if not clinical_passed:
            logger.debug(
                "Stage 2 FAILED: %d clinical issue(s): %s",
                len(clinical_issues), clinical_issues[:3],
            )

        # Stage 3 — Semantic anomaly detection
        anomaly_passed, anomaly_score = self.anomaly_detector.check(report)
        if not anomaly_passed:
            logger.debug(
                "Stage 3 FAILED: anomaly score=%.4f (below threshold).",
                anomaly_score,
            )

        is_valid = rule_passed and clinical_passed and anomaly_passed

        result = ValidationResult(
            is_valid=is_valid,
            rule_passed=rule_passed,
            clinical_passed=clinical_passed,
            anomaly_passed=anomaly_passed,
            rule_violations=rule_violations,
            clinical_issues=clinical_issues,
            anomaly_score=anomaly_score,
            report=report,
        )

        log_fn = logger.debug if is_valid else logger.info
        log_fn("Validation result: %s", result.summary())
        return result

    def validate_batch(
        self,
        reports: List[str],
    ) -> List[ValidationResult]:
        """
        Validate a list of reports.

        For efficiency, Stage 3 (anomaly detection) is batched; Stages 1
        and 2 are applied per-report.

        Args:
            reports: List of report strings.

        Returns:
            List of ValidationResult objects (same order as input).
        """
        # Stages 1 & 2 (per-report)
        stage12_results = [
            (
                self.rule_checker.check(r),
                self.clinical_verifier.check(r),
            )
            for r in reports
        ]

        # Stage 3 (batched for efficiency)
        anomaly_results = self.anomaly_detector.check_batch(reports)

        results = []
        for i, report in enumerate(reports):
            (rule_passed, rule_violations), (clinical_passed, clinical_issues) = (
                stage12_results[i]
            )
            anomaly_passed, anomaly_score = anomaly_results[i]
            is_valid = rule_passed and clinical_passed and anomaly_passed
            results.append(
                ValidationResult(
                    is_valid=is_valid,
                    rule_passed=rule_passed,
                    clinical_passed=clinical_passed,
                    anomaly_passed=anomaly_passed,
                    rule_violations=rule_violations,
                    clinical_issues=clinical_issues,
                    anomaly_score=anomaly_score,
                    report=report,
                )
            )

        n_valid   = sum(1 for r in results if r.is_valid)
        n_invalid = len(results) - n_valid
        logger.info(
            "Batch validation: %d valid, %d rejected (of %d).",
            n_valid, n_invalid, len(results),
        )
        return results

    def compute_metrics(
        self,
        results: List[ValidationResult],
        ground_truth: List[bool],
    ) -> Dict[str, float]:
        """
        Compute precision, recall, F1, and accuracy against ground-truth labels.

        Args:
            results      : List of ValidationResult from validate_batch().
            ground_truth : List of bool (True = report is valid/clean).

        Returns:
            Dict with 'precision', 'recall', 'f1', 'accuracy'.
        """
        preds = [r.is_valid for r in results]
        tp = sum(p and g for p, g in zip(preds, ground_truth))
        fp = sum(p and not g for p, g in zip(preds, ground_truth))
        fn = sum(not p and g for p, g in zip(preds, ground_truth))
        tn = sum(not p and not g for p, g in zip(preds, ground_truth))

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy  = (tp + tn) / max(len(preds), 1)

        return {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "accuracy":  accuracy,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }


# ═════════════════════════════════════════════════════════════
#  CLI entry-point
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="IDS-LLM: Validate generated clinical reports."
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Directory containing generated .txt report files.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write validated reports and audit JSON.")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Anomaly score threshold for Stage 3 (default: auto).")
    parser.add_argument("--iso_forest_path", default=None,
                        help="Path to pre-fitted Isolation Forest (.joblib). "
                             "If not provided, Stage 3 defaults to pass.")
    parser.add_argument("--rule_file", default=None,
                        help="Optional JSON file with additional custom rules.")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "valid").mkdir(exist_ok=True)
    (output_dir / "rejected").mkdir(exist_ok=True)

    validator = IDSLLMValidator(
        rule_file=args.rule_file,
        anomaly_threshold=args.threshold,
    )
    if args.iso_forest_path:
        validator.load_anomaly_model(args.iso_forest_path)

    reports   = list(sorted(input_dir.glob("*.txt")))
    texts     = [p.read_text(encoding="utf-8").strip() for p in reports]
    results   = validator.validate_batch(texts)

    audit = []
    for report_path, result in zip(reports, results):
        dest_subdir = "valid" if result.is_valid else "rejected"
        dest_path   = output_dir / dest_subdir / report_path.name
        dest_path.write_text(result.report, encoding="utf-8")
        audit.append({
            "file":             report_path.name,
            "is_valid":         result.is_valid,
            "rule_passed":      result.rule_passed,
            "clinical_passed":  result.clinical_passed,
            "anomaly_passed":   result.anomaly_passed,
            "rule_violations":  result.rule_violations,
            "clinical_issues":  result.clinical_issues,
            "anomaly_score":    result.anomaly_score,
        })

    audit_path = output_dir / "validation_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    n_valid = sum(1 for r in results if r.is_valid)
    logger.info(
        "Validation complete: %d/%d valid | audit → %s",
        n_valid, len(results), audit_path,
    )

"""
SecureMed-LLM Report Decryption Module
========================================
Implements the local decryption step of the ECIES (Elliptic Curve Integrated
Encryption Scheme) with Curve25519, as described in Sections 3.1.1 (Level 6)
and 5.1l of the SecureMed-LLM paper.

Paper reference (Section 5.1l):
  "Validated report delivery uses ECIES with Curve25519. Each physician is
   assigned a public–private key pair; the validated report is encrypted with
   the physician's public key and decrypted locally."

Paper reference (Section 3.1.1 — Level 6):
  "The ciphertext is decrypted locally using the physician's private key,
   ensuring that only the authorized recipient can access the report."

Paper reference (Section 5.2.6 — Level 6: Secure Delivery and Local Decryption):
  "The encrypted report is transmitted over a secure internal hospital network.
   Upon reception, the report is decrypted locally using the physician's private
   key."

This module is the exact cryptographic inverse of src/encryption/encrypt_report.py.
The ECIES decryption procedure reverses the five-step encryption:
  1. Deserialize ephemeral X25519 public key from payload header.
  2. Perform X25519 ECDH using the recipient's static private key.
  3. Derive AES-256 key from the ECDH shared secret via HKDF-SHA256.
  4. Authenticate and decrypt the ciphertext with AES-256-GCM.
  5. Decode the plaintext bytes to a UTF-8 report string.

Encrypted payload format (binary) — mirrored from encrypt_report.py:
  [32 bytes]  ephemeral X25519 public key
  [12 bytes]  AES-GCM nonce
  [16 bytes]  AES-GCM authentication tag
  [N  bytes]  AES-256-GCM ciphertext of UTF-8 report bytes

Security properties:
  - Forward secrecy: each encryption uses a fresh ephemeral key pair;
    compromise of the physician's static private key does not expose past
    sessions (assuming ephemeral keys are discarded after encryption).
  - Authentication: AES-GCM provides authenticated encryption; tampering
    with the ciphertext or header causes a cryptographic exception, not
    silent data corruption.
  - Local-only decryption: the private key never leaves the clinician's
    device (paper Section 3.1.1).

Dependencies:
  pip install cryptography
"""

import logging
import os
from pathlib import Path
from typing import Tuple, Union

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants — must mirror encrypt_report.py exactly
# ---------------------------------------------------------------------------

# HKDF info tag — must be identical to the value used during encryption
HKDF_INFO: bytes = b"SecureMed-LLM-ECIES-v1"

# AES-256-GCM parameters
AES_KEY_BYTES: int = 32    # 256-bit key
GCM_NONCE_BYTES: int = 12  # 96-bit nonce
GCM_TAG_BYTES: int = 16    # 128-bit authentication tag

# Payload byte offsets — mirrored from encrypt_report.py
EPHEM_PUB_OFFSET: int = 0
EPHEM_PUB_END: int = 32                               # X25519 public key = 32 bytes
NONCE_OFFSET: int = EPHEM_PUB_END
NONCE_END: int = NONCE_OFFSET + GCM_NONCE_BYTES       # 44
TAG_OFFSET: int = NONCE_END
TAG_END: int = TAG_OFFSET + GCM_TAG_BYTES             # 60
CIPHERTEXT_OFFSET: int = TAG_END                      # 60

# Minimum payload length: 32 (ephem pub) + 12 (nonce) + 16 (tag) + 1 (ciphertext)
MIN_PAYLOAD_BYTES: int = CIPHERTEXT_OFFSET + 1


# ---------------------------------------------------------------------------
#  Key loading helpers (thin wrappers for ergonomic CLI / library use)
# ---------------------------------------------------------------------------

def load_private_key_pem(
    pem_data: bytes,
    password: bytes = None,
) -> X25519PrivateKey:
    """
    Load an X25519 private key from PEM-encoded bytes.

    Args:
        pem_data: PEM-encoded private key (PKCS#8 format, as saved by
                  encrypt_report.serialize_private_key_pem).
        password: Optional decryption password if the PEM was encrypted.

    Returns:
        X25519PrivateKey instance.

    Raises:
        ValueError: If the PEM data cannot be parsed.
        TypeError:  If the key type is not X25519.
    """
    key = serialization.load_pem_private_key(pem_data, password=password)
    if not isinstance(key, X25519PrivateKey):
        raise TypeError(
            f"Expected X25519PrivateKey from PEM, got {type(key).__name__}."
        )
    return key


def load_private_key_from_file(
    pem_path: Union[str, Path],
    password: bytes = None,
) -> X25519PrivateKey:
    """
    Load an X25519 private key from a PEM file on disk.

    Args:
        pem_path: Path to the physician's private key PEM file
                  (typically ``<physician_id>_private.pem``).
        password: Optional PEM decryption password.

    Returns:
        X25519PrivateKey instance.

    Raises:
        FileNotFoundError: If the PEM file does not exist.
    """
    pem_path = Path(pem_path)
    if not pem_path.exists():
        raise FileNotFoundError(f"Private key file not found: {pem_path}")
    pem_data = pem_path.read_bytes()
    logger.debug("Private key loaded from %s", pem_path)
    return load_private_key_pem(pem_data, password=password)


# ---------------------------------------------------------------------------
#  Core ECIES decryption
# ---------------------------------------------------------------------------

def _derive_aes_key(shared_secret: bytes, salt: bytes = None) -> bytes:
    """
    Derive a 256-bit AES key from the ECDH shared secret via HKDF-SHA256.

    This is the inverse KDF step used in encryption; parameters must match
    those in encrypt_report._derive_aes_key exactly to recover the same key.

    Args:
        shared_secret: Raw ECDH shared secret bytes (32 bytes for X25519).
        salt:          Optional HKDF salt (None = zero-length salt per RFC 5869).

    Returns:
        32-byte AES-256 key.
    """
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=AES_KEY_BYTES,
        salt=salt,
        info=HKDF_INFO,
    )
    return hkdf.derive(shared_secret)


def decrypt_report(
    payload: bytes,
    recipient_private_key: Union[X25519PrivateKey, bytes],
) -> str:
    """
    Decrypt an ECIES-encrypted clinical report payload.

    Implements the local decryption step of Level 6 of the SecureMed-LLM
    pipeline (paper Section 3.1.1 and Section 5.2.6). This is the exact
    cryptographic inverse of encrypt_report.encrypt_report().

    Decryption steps:
      1. Parse the binary payload: [ephem_pub | nonce | tag | ciphertext].
      2. Load the ephemeral X25519 public key from the payload header.
      3. Perform X25519 ECDH: recipient_private_key × ephemeral_pub → shared_secret.
      4. Derive AES-256 key via HKDF-SHA256 (identical parameters as encryption).
      5. Reconstruct the AES-GCM authenticated ciphertext (ciphertext + tag).
      6. Decrypt and authenticate with AES-256-GCM.
      7. Decode plaintext bytes as UTF-8.

    Args:
        payload:               Encrypted payload bytes produced by
                               encrypt_report.encrypt_report().
        recipient_private_key: The physician's X25519PrivateKey, or a PEM-encoded
                               private key as raw bytes (auto-detected).

    Returns:
        Decrypted clinical report as a UTF-8 string.

    Raises:
        ValueError:        If the payload is too short to be a valid ECIES payload.
        TypeError:         If recipient_private_key is not a recognised type.
        InvalidTag:        If AES-GCM authentication fails (payload is corrupt or
                           was tampered with). This is a hard cryptographic error.
        UnicodeDecodeError: If decrypted bytes are not valid UTF-8.
    """
    # ── Input validation ────────────────────────────────────────────────────
    if len(payload) < MIN_PAYLOAD_BYTES:
        raise ValueError(
            f"Payload too short: expected >= {MIN_PAYLOAD_BYTES} bytes, "
            f"got {len(payload)} bytes. Payload may be corrupt or truncated."
        )

    # ── Normalise private key type ──────────────────────────────────────────
    if isinstance(recipient_private_key, (bytes, bytearray)):
        # Assume PEM-encoded bytes if passed as bytes
        recipient_private_key = load_private_key_pem(bytes(recipient_private_key))
    elif not isinstance(recipient_private_key, X25519PrivateKey):
        raise TypeError(
            f"recipient_private_key must be X25519PrivateKey or PEM bytes, "
            f"got {type(recipient_private_key).__name__}."
        )

    # ── 1. Parse payload components ─────────────────────────────────────────
    ephem_pub_bytes = payload[EPHEM_PUB_OFFSET:EPHEM_PUB_END]    # 32 B
    nonce           = payload[NONCE_OFFSET:NONCE_END]              # 12 B
    tag             = payload[TAG_OFFSET:TAG_END]                  # 16 B
    ciphertext      = payload[CIPHERTEXT_OFFSET:]                  # N  B

    logger.debug(
        "Decrypting | payload=%d B | ciphertext=%d B",
        len(payload), len(ciphertext),
    )

    # ── 2. Deserialise ephemeral public key ─────────────────────────────────
    ephemeral_public = X25519PublicKey.from_public_bytes(ephem_pub_bytes)

    # ── 3. X25519 ECDH key agreement ────────────────────────────────────────
    shared_secret = recipient_private_key.exchange(ephemeral_public)

    # ── 4. Derive AES-256 key via HKDF-SHA256 ───────────────────────────────
    aes_key = _derive_aes_key(shared_secret)

    # ── 5. Reconstruct authenticated ciphertext (ciphertext ‖ tag) ──────────
    #  AESGCM.decrypt() expects the tag appended to the ciphertext, exactly
    #  as AESGCM.encrypt() returns it.
    ciphertext_with_tag = ciphertext + tag

    # ── 6. Decrypt and authenticate ─────────────────────────────────────────
    aesgcm = AESGCM(aes_key)
    try:
        plaintext_bytes = aesgcm.decrypt(nonce, ciphertext_with_tag, associated_data=None)
    except InvalidTag:
        # Re-raise with a more informative message; do NOT log the key material.
        raise InvalidTag(
            "AES-GCM authentication failed. The payload may have been tampered "
            "with, the wrong private key was used, or the ciphertext is corrupt."
        )

    # ── 7. Decode UTF-8 ─────────────────────────────────────────────────────
    report_text = plaintext_bytes.decode("utf-8")

    logger.info(
        "Report decrypted successfully | plaintext=%d B (%d chars)",
        len(plaintext_bytes), len(report_text),
    )
    return report_text


def decrypt_report_from_file(
    payload_path: Union[str, Path],
    recipient_private_key: Union[X25519PrivateKey, bytes],
) -> str:
    """
    Read an encrypted payload from disk and decrypt it.

    Args:
        payload_path:          Path to the encrypted payload file written by
                               encrypt_report.encrypt_report_to_file().
        recipient_private_key: Physician's X25519PrivateKey or PEM bytes.

    Returns:
        Decrypted clinical report string.

    Raises:
        FileNotFoundError: If the payload file does not exist.
        See decrypt_report() for further exceptions.
    """
    payload_path = Path(payload_path)
    if not payload_path.exists():
        raise FileNotFoundError(f"Encrypted payload file not found: {payload_path}")

    payload = payload_path.read_bytes()
    logger.info(
        "Encrypted payload read from %s (%d B)", payload_path, len(payload)
    )
    return decrypt_report(payload, recipient_private_key)


def decrypt_report_to_file(
    payload_path: Union[str, Path],
    recipient_private_key: Union[X25519PrivateKey, bytes],
    output_path: Union[str, Path],
    encoding: str = "utf-8",
) -> Path:
    """
    Decrypt an encrypted payload file and write the plaintext report to disk.

    Args:
        payload_path:          Path to the encrypted payload (.bin) file.
        recipient_private_key: Physician's X25519PrivateKey or PEM bytes.
        output_path:           Destination path for the decrypted report text.
        encoding:              Text encoding for the output file (default: utf-8).

    Returns:
        Path to the written plaintext file.

    Raises:
        FileNotFoundError: If payload_path does not exist.
        See decrypt_report() for further exceptions.
    """
    report_text = decrypt_report_from_file(payload_path, recipient_private_key)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding=encoding)

    logger.info("Decrypted report written → %s (%d chars)", output_path, len(report_text))
    return output_path


# ---------------------------------------------------------------------------
#  Convenience round-trip helper (useful in tests and CI)
# ---------------------------------------------------------------------------

def verify_round_trip(
    report_text: str,
    private_key: X25519PrivateKey,
    public_key: X25519PublicKey,
) -> bool:
    """
    Encrypt then immediately decrypt a report string; assert byte-level identity.

    Intended for unit-testing the encryption/decryption pair without touching
    the filesystem. Imports encrypt_report lazily to avoid circular imports at
    the module level.

    Args:
        report_text: Plaintext report to round-trip.
        private_key: Physician's X25519 private key.
        public_key:  Corresponding X25519 public key.

    Returns:
        True if decrypted output exactly matches the input; False otherwise.
        Logs a WARNING if the round-trip fails.
    """
    # Lazy import to avoid circular dependency at module level
    from src.encryption.encrypt_report import encrypt_report  # noqa: PLC0415

    payload = encrypt_report(report_text, public_key)
    recovered = decrypt_report(payload, private_key)
    success = recovered == report_text
    if success:
        logger.info("Round-trip verification PASSED (%d chars)", len(report_text))
    else:
        logger.warning(
            "Round-trip verification FAILED | original=%d chars | recovered=%d chars",
            len(report_text), len(recovered),
        )
    return success


# ---------------------------------------------------------------------------
#  CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "SecureMed-LLM: Decrypt a validated clinical report "
            "using ECIES/Curve25519 (paper Section 5.1l / 5.2.6)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── decrypt sub-command ────────────────────────────────────────────
    dec_parser = subparsers.add_parser(
        "decrypt",
        help="Decrypt an ECIES-encrypted clinical report with the physician's private key.",
    )
    dec_parser.add_argument(
        "--payload",
        required=True,
        help="Path to the encrypted payload file (.bin) produced by encrypt_report.",
    )
    dec_parser.add_argument(
        "--private_key",
        required=True,
        help="Path to the physician's private key PEM file (e.g. physician_private.pem).",
    )
    dec_parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path for the decrypted report text. "
            "If omitted, the report is printed to stdout."
        ),
    )
    dec_parser.add_argument(
        "--password",
        default=None,
        help="Optional PEM password if the private key file is passphrase-protected.",
    )

    # ── roundtrip sub-command (for testing) ───────────────────────────
    rt_parser = subparsers.add_parser(
        "roundtrip",
        help="Smoke-test: generate a key pair, encrypt and immediately decrypt a message.",
    )
    rt_parser.add_argument(
        "--message",
        default="This is a test SecureMed-LLM clinical report.",
        help="Test message string to encrypt and decrypt.",
    )

    args = parser.parse_args()

    # ── dispatch ───────────────────────────────────────────────────────
    if args.command == "decrypt":
        password_bytes = args.password.encode() if args.password else None
        try:
            private_key = load_private_key_from_file(args.private_key, password_bytes)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            print(f"ERROR loading private key: {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            if args.output:
                out_path = decrypt_report_to_file(
                    payload_path=args.payload,
                    recipient_private_key=private_key,
                    output_path=args.output,
                )
                print(f"Decrypted report written → {out_path}")
            else:
                report = decrypt_report_from_file(
                    payload_path=args.payload,
                    recipient_private_key=private_key,
                )
                print(report)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        except InvalidTag:
            print(
                "ERROR: Decryption failed — authentication tag mismatch. "
                "The payload may be corrupt or the wrong private key was supplied.",
                file=sys.stderr,
            )
            sys.exit(1)
        except UnicodeDecodeError as exc:
            print(f"ERROR: Decrypted bytes are not valid UTF-8: {exc}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "roundtrip":
        # Import keygen from sibling module
        from src.encryption.encrypt_report import generate_key_pair  # noqa: PLC0415

        priv_key, pub_key = generate_key_pair()
        print(f"Generated ephemeral key pair for round-trip test.")
        success = verify_round_trip(args.message, priv_key, pub_key)
        if success:
            print("Round-trip PASSED ✓")
            sys.exit(0)
        else:
            print("Round-trip FAILED ✗", file=sys.stderr)
            sys.exit(1)

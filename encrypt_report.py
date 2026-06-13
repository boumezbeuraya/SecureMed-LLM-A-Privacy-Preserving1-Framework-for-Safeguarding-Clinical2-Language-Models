"""
SecureMed-LLM Report Encryption Module
========================================
Implements ECIES (Elliptic Curve Integrated Encryption Scheme) with Curve25519
for end-to-end confidentiality of validated clinical reports, as described in
Sections 3.1.1 (Level 6) and 5.1l of the SecureMed-LLM paper.

Paper reference (Section 5.1l):
  "Validated report delivery uses ECIES with Curve25519. Each physician is
   assigned a public–private key pair; the validated report is encrypted with
   the physician's public key and decrypted locally."

Paper reference (Section 3.1.1 — Level 6):
  "The validated report is encrypted using the physician's public key via the
   Elliptic Curve Integrated Encryption Scheme (ECIES). The encrypted report is
   transmitted over a TLS-secured channel, providing layered security."

Scope of encryption (Section 5.1l — explicitly stated in paper):
  "Encryption is applied to the final validated report text only.
   Intermediate embeddings and model activations during inference are NOT
   encrypted; the inference server is assumed to be a trusted environment."

Implementation notes:
  - We use the `eciespy` library (pip install eciespy), which implements
    ECIES over secp256k1 by default. Curve25519 (X25519) key exchange is
    used for the underlying ECDH step via the `cryptography` library.
  - Because eciespy targets secp256k1, we implement ECIES manually using:
      • X25519 (Curve25519) ECDH key agreement          → shared secret
      • HKDF-SHA256                                      → derived symmetric key
      • AES-256-GCM                                      → authenticated encryption
    This is the standard ECIES construction described in the paper reference [38]:
      Gayoso Martínez et al., "Security and practical considerations when
      implementing ECIES", Cryptologia 39.3 (2015).
  - Key pairs are serialised to PEM (private) and raw bytes (public) for
    storage and transmission compatibility.
  - The ephemeral public key, AES-GCM nonce, and GCM authentication tag are
    prepended to the ciphertext to form a self-contained encrypted payload.

Encrypted payload format (binary):
  [32 bytes]  ephemeral X25519 public key
  [12 bytes]  AES-GCM nonce
  [16 bytes]  AES-GCM authentication tag
  [N  bytes]  AES-256-GCM ciphertext of UTF-8 report bytes

Dependencies:
  pip install cryptography
"""

import logging
import os
from pathlib import Path
from typing import Tuple, Union

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# HKDF info tag — domain-separates key derivation for this application
HKDF_INFO: bytes = b"SecureMed-LLM-ECIES-v1"

# AES-256-GCM parameters
AES_KEY_BYTES: int = 32   # 256-bit key
GCM_NONCE_BYTES: int = 12  # 96-bit nonce (GCM standard)
GCM_TAG_BYTES: int = 16   # 128-bit authentication tag

# Payload byte offsets
EPHEM_PUB_OFFSET: int = 0
EPHEM_PUB_END: int = 32                              # X25519 public key = 32 bytes
NONCE_OFFSET: int = EPHEM_PUB_END
NONCE_END: int = NONCE_OFFSET + GCM_NONCE_BYTES      # 44
TAG_OFFSET: int = NONCE_END
TAG_END: int = TAG_OFFSET + GCM_TAG_BYTES            # 60
CIPHERTEXT_OFFSET: int = TAG_END                     # 60


# ---------------------------------------------------------------------------
#  Key generation and serialisation
# ---------------------------------------------------------------------------

def generate_key_pair() -> Tuple[X25519PrivateKey, X25519PublicKey]:
    """
    Generate a new Curve25519 key pair for a physician.

    The private key is stored securely on the clinician's local device;
    the public key is registered with the trusted server (paper Section 3.1.1).

    Returns:
        Tuple of (private_key, public_key).
    """
    private_key = X25519PrivateKey.generate()
    public_key = private_key.public_key()
    logger.info("New Curve25519 key pair generated.")
    return private_key, public_key


def serialize_public_key(public_key: X25519PublicKey) -> bytes:
    """
    Serialise an X25519 public key to raw bytes (32 bytes).

    This raw format is used in the encrypted payload header and for
    ECDH computation during encryption/decryption.

    Args:
        public_key: An X25519PublicKey instance.

    Returns:
        Raw 32-byte public key material.
    """
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def serialize_private_key_pem(
    private_key: X25519PrivateKey,
    password: bytes = None,
) -> bytes:
    """
    Serialise an X25519 private key to PEM format for local storage.

    Args:
        private_key: An X25519PrivateKey instance.
        password:    Optional password for PEM encryption (recommended).

    Returns:
        PEM-encoded private key bytes.
    """
    encryption = (
        serialization.BestAvailableEncryption(password)
        if password
        else serialization.NoEncryption()
    )
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    )


def load_private_key_pem(
    pem_data: bytes,
    password: bytes = None,
) -> X25519PrivateKey:
    """
    Load an X25519 private key from PEM bytes.

    Args:
        pem_data: PEM-encoded private key bytes.
        password: Password if the PEM is encrypted.

    Returns:
        X25519PrivateKey instance.
    """
    return serialization.load_pem_private_key(pem_data, password=password)


def load_public_key_raw(raw_bytes: bytes) -> X25519PublicKey:
    """
    Load an X25519 public key from 32 raw bytes.

    Args:
        raw_bytes: 32-byte raw public key material.

    Returns:
        X25519PublicKey instance.
    """
    return X25519PublicKey.from_public_bytes(raw_bytes)


def save_key_pair(
    private_key: X25519PrivateKey,
    public_key: X25519PublicKey,
    output_dir: Union[str, Path],
    physician_id: str = "physician",
    password: bytes = None,
) -> Tuple[Path, Path]:
    """
    Persist a Curve25519 key pair to disk.

    The private key is saved as PEM; the public key is saved as raw bytes.
    In production, the private PEM should be stored only on the clinician's
    local device, never transmitted (paper Section 3.1.1).

    Args:
        private_key:  X25519PrivateKey to save.
        public_key:   X25519PublicKey to save.
        output_dir:   Directory for key files.
        physician_id: Identifier used in filenames.
        password:     Optional PEM encryption password.

    Returns:
        Tuple of (private_key_path, public_key_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    priv_path = output_dir / f"{physician_id}_private.pem"
    pub_path = output_dir / f"{physician_id}_public.bin"

    priv_path.write_bytes(serialize_private_key_pem(private_key, password))
    pub_path.write_bytes(serialize_public_key(public_key))

    # Restrict file permissions on the private key
    os.chmod(priv_path, 0o600)

    logger.info(
        "Key pair saved | private=%s | public=%s",
        priv_path, pub_path,
    )
    return priv_path, pub_path


# ---------------------------------------------------------------------------
#  Core ECIES encryption
# ---------------------------------------------------------------------------

def _derive_aes_key(shared_secret: bytes, salt: bytes = None) -> bytes:
    """
    Derive a 256-bit AES key from the ECDH shared secret via HKDF-SHA256.

    This implements the KDF step of ECIES (paper reference [38]).

    Args:
        shared_secret: Raw ECDH shared secret bytes.
        salt:          Optional HKDF salt (None = zero-length salt per RFC 5869).

    Returns:
        32-byte AES key.
    """
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=AES_KEY_BYTES,
        salt=salt,
        info=HKDF_INFO,
    )
    return hkdf.derive(shared_secret)


def encrypt_report(
    report_text: str,
    recipient_public_key: Union[X25519PublicKey, bytes],
) -> bytes:
    """
    Encrypt a validated clinical report using ECIES with Curve25519.

    Implements Level 6 of the SecureMed-LLM pipeline (paper Section 3.1.1):
      1. Generate ephemeral Curve25519 key pair.
      2. Perform X25519 ECDH with the recipient's public key → shared secret.
      3. Derive AES-256-GCM key from shared secret via HKDF-SHA256.
      4. Encrypt report bytes with AES-256-GCM (random nonce).
      5. Pack [ephemeral_pub | nonce | tag | ciphertext] into payload.

    Encryption is applied to the final validated report text only, as stated
    in the paper. Model inputs, embeddings, and activations are out of scope.

    Args:
        report_text:           Validated clinical report string (UTF-8).
        recipient_public_key:  Physician's X25519PublicKey or raw 32-byte key.

    Returns:
        Encrypted payload bytes:
          [32B ephemeral pub] + [12B nonce] + [16B GCM tag] + [NB ciphertext]

    Raises:
        ValueError: If report_text is empty.
        TypeError:  If recipient_public_key is not a recognised type.
    """
    if not report_text or not report_text.strip():
        raise ValueError("report_text must be a non-empty string.")

    # Normalise public key type
    if isinstance(recipient_public_key, bytes):
        recipient_public_key = load_public_key_raw(recipient_public_key)
    elif not isinstance(recipient_public_key, X25519PublicKey):
        raise TypeError(
            f"recipient_public_key must be X25519PublicKey or bytes, "
            f"got {type(recipient_public_key).__name__}."
        )

    # 1. Generate ephemeral key pair
    ephemeral_private = X25519PrivateKey.generate()
    ephemeral_public = ephemeral_private.public_key()

    # 2. ECDH key agreement
    shared_secret = ephemeral_private.exchange(recipient_public_key)

    # 3. Derive AES key
    aes_key = _derive_aes_key(shared_secret)

    # 4. Encrypt with AES-256-GCM
    nonce = os.urandom(GCM_NONCE_BYTES)
    aesgcm = AESGCM(aes_key)
    plaintext = report_text.encode("utf-8")
    # AESGCM.encrypt() returns ciphertext + 16-byte tag appended
    ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    # Split ciphertext and tag (tag is the last GCM_TAG_BYTES bytes)
    ciphertext = ciphertext_with_tag[:-GCM_TAG_BYTES]
    tag = ciphertext_with_tag[-GCM_TAG_BYTES:]

    # 5. Serialise ephemeral public key
    ephemeral_pub_bytes = serialize_public_key(ephemeral_public)

    # 6. Assemble payload: [ephem_pub | nonce | tag | ciphertext]
    payload = ephemeral_pub_bytes + nonce + tag + ciphertext

    logger.info(
        "Report encrypted | plaintext=%d B | payload=%d B",
        len(plaintext), len(payload),
    )
    return payload


def encrypt_report_to_file(
    report_text: str,
    recipient_public_key: Union[X25519PublicKey, bytes],
    output_path: Union[str, Path],
) -> Path:
    """
    Encrypt a clinical report and write the payload to a file.

    Args:
        report_text:          Validated report string.
        recipient_public_key: Physician's X25519 public key (object or raw bytes).
        output_path:          Destination path for the encrypted payload.

    Returns:
        Path to the written encrypted file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = encrypt_report(report_text, recipient_public_key)
    output_path.write_bytes(payload)
    logger.info("Encrypted report written → %s (%d B)", output_path, len(payload))
    return output_path


# ---------------------------------------------------------------------------
#  CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "SecureMed-LLM: Encrypt a validated clinical report "
            "using ECIES/Curve25519 (paper Section 5.1l)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── keygen sub-command ─────────────────────────────────────────────
    kg_parser = subparsers.add_parser(
        "keygen",
        help="Generate a new Curve25519 key pair for a physician.",
    )
    kg_parser.add_argument("--output_dir", required=True,
                           help="Directory to store key files.")
    kg_parser.add_argument("--physician_id", default="physician",
                           help="Identifier used in key filenames.")

    # ── encrypt sub-command ────────────────────────────────────────────
    enc_parser = subparsers.add_parser(
        "encrypt",
        help="Encrypt a validated clinical report with a physician's public key.",
    )
    enc_parser.add_argument("--report", required=True,
                            help="Path to the validated report text file.")
    enc_parser.add_argument("--public_key", required=True,
                            help="Path to the physician's public key (.bin).")
    enc_parser.add_argument("--output", required=True,
                            help="Output path for the encrypted payload.")

    args = parser.parse_args()

    if args.command == "keygen":
        priv_key, pub_key = generate_key_pair()
        save_key_pair(priv_key, pub_key, args.output_dir, args.physician_id)
        print(f"Key pair saved to {args.output_dir}/")

    elif args.command == "encrypt":
        report_path = Path(args.report)
        if not report_path.exists():
            print(f"ERROR: Report file not found: {report_path}", file=sys.stderr)
            sys.exit(1)
        pub_key_path = Path(args.public_key)
        if not pub_key_path.exists():
            print(f"ERROR: Public key file not found: {pub_key_path}", file=sys.stderr)
            sys.exit(1)

        report_text = report_path.read_text(encoding="utf-8")
        pub_key_bytes = pub_key_path.read_bytes()
        out_path = encrypt_report_to_file(report_text, pub_key_bytes, args.output)
        print(f"Encrypted payload written → {out_path}")

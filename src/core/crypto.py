"""Cryptographic primitives for AgentID.

Paper 1 scope: Ed25519 signatures + SHA-256 hashing + secure nonces.
SD-JWT and BBS+ primitives are deferred to Paper 3 (E5).

Design rule (Research Foundation v4 §2.6): security-critical operations are
deterministic and run OUTSIDE the LLM's inference path. Nothing in this module
talks to an LLM; all randomness comes from os.urandom.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature


NONCE_BITS = 128


# ---------- Hashing ----------

def sha256_hex(data: bytes) -> str:
    """Lowercase hex SHA-256. Used for output_hash, prev_turn_hash, key ids."""
    return hashlib.sha256(data).hexdigest()


# ---------- Nonces ----------

def new_nonce_hex(bits: int = NONCE_BITS) -> str:
    """Cryptographically random nonce as lowercase hex.

    128 bits is the default: single-use + timestamp bounds give birthday
    resistance well beyond any realistic session volume.
    """
    if bits % 8 != 0:
        raise ValueError("nonce bits must be a multiple of 8")
    return os.urandom(bits // 8).hex()


# ---------- Ed25519 key wrapper ----------

@dataclass(frozen=True)
class KeyPair:
    """An Ed25519 keypair plus a stable, public-key-derived key_id."""

    private: Ed25519PrivateKey
    public: Ed25519PublicKey
    key_id: str

    @property
    def public_bytes(self) -> bytes:
        return self.public.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )


def generate_keypair() -> KeyPair:
    """Generate a fresh Ed25519 keypair.

    key_id = first 16 hex chars of sha256(public_key_raw_bytes). Short enough
    to be readable in logs, long enough to be collision-resistant in realistic
    deployer key inventories.
    """
    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    return KeyPair(private=private, public=public, key_id=key_id_from_public(public))


def key_id_from_public(public: Ed25519PublicKey) -> str:
    """Derive the wire ``key_id`` from an Ed25519 public key."""
    pub_raw = public.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return sha256_hex(pub_raw)[:16]


def sign(private: Ed25519PrivateKey, data: bytes) -> str:
    """Sign `data` with Ed25519. Returns lowercase hex signature."""
    return private.sign(data).hex()


def verify(public: Ed25519PublicKey, data: bytes, signature_hex: str) -> bool:
    """Verify an Ed25519 signature. Returns True iff the signature is valid.

    Never raises on bad signatures — the caller needs a boolean to make a
    security decision; exceptions on the happy path are a footgun.
    """
    try:
        public.verify(bytes.fromhex(signature_hex), data)
        return True
    except (InvalidSignature, ValueError):
        return False

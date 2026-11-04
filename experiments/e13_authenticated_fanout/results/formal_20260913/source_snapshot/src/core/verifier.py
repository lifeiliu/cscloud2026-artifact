"""Verifier: checks a SignedBoundID against a presented output.

Checks performed, in order (fail-closed on the first failure):
  1. Key lookup  — resolve `key_id` to a public key + its revocation status.
  2. Signature   — Ed25519 over canonical JSON of the payload.
  3. Expiry      — created_at <= now <= expires_at.
  4. Nonce       — unseen before (single-use anti-replay).
  5. Binding     — sha256(presented_output_bytes) == payload.output_hash.
  6. Ancestry    — ancestry_depth matches the length of the presented chain.

Design notes:
  - The nonce store is abstract (`NonceStore`). A production deployment would
    use Redis or similar. For experiments we provide InMemoryNonceStore.
  - Revocation is abstract (`KeyDirectory.is_revoked`). The prototype treats
    all known keys as non-revoked; E3 and later may extend this.
  - This module is sync by design. The FastAPI wrapper (§12 file structure)
    can be a thin async shim around a Verifier instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Protocol

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .agent_id import SignedBoundID
from .crypto import sha256_hex, verify


# ---------- Reason codes for verification failures ----------

class VerifyReason:
    OK = "ok"
    UNKNOWN_KEY = "unknown_key"
    REVOKED_KEY = "revoked_key"
    BAD_SIGNATURE = "bad_signature"
    NOT_YET_VALID = "not_yet_valid"
    EXPIRED = "expired"
    REPLAYED_NONCE = "replayed_nonce"
    OUTPUT_MISMATCH = "output_mismatch"
    ANCESTRY_LENGTH_MISMATCH = "ancestry_length_mismatch"


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    reason: str

    @classmethod
    def success(cls) -> "VerifyResult":
        return cls(True, VerifyReason.OK)

    @classmethod
    def failure(cls, reason: str) -> "VerifyResult":
        return cls(False, reason)


# ---------- Pluggable stores ----------

class KeyDirectory(Protocol):
    def get(self, key_id: str) -> Optional[Ed25519PublicKey]: ...
    def is_revoked(self, key_id: str) -> bool: ...


class NonceStore(Protocol):
    def seen(self, nonce: str) -> bool: ...
    def mark(self, nonce: str) -> None: ...


class InMemoryKeyDirectory:
    """Trivial in-process key registry. Non-thread-safe by design (tests only)."""

    def __init__(self) -> None:
        self._keys: dict[str, Ed25519PublicKey] = {}
        self._revoked: set[str] = set()

    def add(self, key_id: str, public: Ed25519PublicKey) -> None:
        self._keys[key_id] = public

    def revoke(self, key_id: str) -> None:
        self._revoked.add(key_id)

    def get(self, key_id: str) -> Optional[Ed25519PublicKey]:
        return self._keys.get(key_id)

    def is_revoked(self, key_id: str) -> bool:
        return key_id in self._revoked


class InMemoryNonceStore:
    """Trivial in-process nonce tracker. Non-thread-safe by design (tests only)."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def seen(self, nonce: str) -> bool:
        return nonce in self._seen

    def mark(self, nonce: str) -> None:
        self._seen.add(nonce)


# ---------- Verifier ----------

@dataclass
class Verifier:
    keys: KeyDirectory
    nonces: NonceStore

    def verify(
        self,
        *,
        bound_id: SignedBoundID,
        presented_output: bytes,
        presented_chain_length: int = 0,
        now: Optional[datetime] = None,
    ) -> VerifyResult:
        """Verify `bound_id` binds `presented_output`.

        `presented_chain_length` is how many parent links the verifier actually
        received (0 for a root-level token). Cross-checked against
        `payload.ancestry_depth` — an attacker who drops hops is detected.
        """
        now = now or datetime.now(timezone.utc)

        # 1. Key lookup
        public = self.keys.get(bound_id.key_id)
        if public is None:
            return VerifyResult.failure(VerifyReason.UNKNOWN_KEY)
        if self.keys.is_revoked(bound_id.key_id):
            return VerifyResult.failure(VerifyReason.REVOKED_KEY)

        # 2. Signature
        canonical = bound_id.payload.to_canonical_json()
        if not verify(public, canonical, bound_id.signature_hex):
            return VerifyResult.failure(VerifyReason.BAD_SIGNATURE)

        # 3. Expiry
        created = parse_rfc3339(bound_id.payload.created_at)
        expires = parse_rfc3339(bound_id.payload.expires_at)
        if now < created:
            return VerifyResult.failure(VerifyReason.NOT_YET_VALID)
        if now > expires:
            return VerifyResult.failure(VerifyReason.EXPIRED)

        # 4. Nonce anti-replay
        if self.nonces.seen(bound_id.payload.nonce):
            return VerifyResult.failure(VerifyReason.REPLAYED_NONCE)

        # 5. Output binding
        if sha256_hex(presented_output) != bound_id.payload.output_hash:
            return VerifyResult.failure(VerifyReason.OUTPUT_MISMATCH)

        # 6. Ancestry length consistency
        if presented_chain_length != bound_id.payload.ancestry_depth:
            return VerifyResult.failure(VerifyReason.ANCESTRY_LENGTH_MISMATCH)

        # Only mark the nonce consumed on full success; if something later in
        # a compound flow rejects, callers may want to re-evaluate without
        # burning the nonce. For a successful verify this is one-shot.
        self.nonces.mark(bound_id.payload.nonce)
        return VerifyResult.success()


def parse_rfc3339(ts: str) -> datetime:
    """Parse the RFC 3339 'Z' format emitted by Deployer._rfc3339.

    Accepts both '...Z' and explicit '+00:00' offsets for robustness.
    """
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)

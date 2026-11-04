"""Defense layers L0-L4 per Research Foundation v4 Appendix A.

Additive composition:

  L0 = no defense (accept anything)
  L1 = L0 + channel MAC
  L2 = L1 + Ed25519 signature on the bound-id payload
  L3 = L2 + expiry + output binding + ancestry length consistency
  L4 = L3 + single-use nonce

The Verifier from core/ already implements L4-equivalent checks in one pass;
LayeredVerifier wraps it with a knob that short-circuits higher-layer checks
so we can exercise each row of the sanity-check matrix independently.

Failures short-circuit at the earliest-rejected layer, so the returned
`reason` string points at the FIRST check that caught the attack — useful for
the matrix write-up.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..core.agent_id import BoundIDPayload, SignedBoundID
from ..core.crypto import sha256_hex, verify as ed25519_verify
from ..core.verifier import (
    InMemoryKeyDirectory,
    InMemoryNonceStore,
    KeyDirectory,
    NonceStore,
    VerifyReason,
    VerifyResult,
    parse_rfc3339,
)
from .channel import WirePacket, channel_mac_valid


class DefenseLevel(enum.IntEnum):
    L0 = 0  # no defense
    L1 = 1  # + channel MAC
    L2 = 2  # + signature
    L3 = 3  # + expiry + output binding + ancestry
    L4 = 4  # + nonce anti-replay


@dataclass
class ReceivedMessage:
    """What the verifier pulls off the wire after channel + envelope parsing."""
    bound_id: SignedBoundID
    presented_output: bytes
    presented_chain_length: int


@dataclass
class LayeredVerifier:
    """Verifier that runs a prefix of the L0-L4 check list.

    Parameters:
      level: which defense layers are active. Checks above `level` are skipped.
      keys: public-key registry with revocation state.
      nonces: nonce store for anti-replay (only consulted at L4).
      channel_secrets: per-channel-id shared secret for the channel MAC.
    """

    level: DefenseLevel
    keys: KeyDirectory
    nonces: NonceStore
    channel_secrets: dict[str, bytes]

    def verify(
        self,
        *,
        packet: WirePacket,
        message: ReceivedMessage,
        now: Optional[datetime] = None,
    ) -> VerifyResult:
        now = now or datetime.now(timezone.utc)

        # ------ L1: channel MAC ------
        if self.level >= DefenseLevel.L1:
            secret = self.channel_secrets.get(packet.channel_id)
            if secret is None or not channel_mac_valid(
                secret, packet.payload_bytes, packet.mac_hex
            ):
                return VerifyResult.failure("channel_mac_invalid")

        # ------ L2: signature ------
        if self.level >= DefenseLevel.L2:
            public = self.keys.get(message.bound_id.key_id)
            if public is None:
                return VerifyResult.failure(VerifyReason.UNKNOWN_KEY)
            if self.keys.is_revoked(message.bound_id.key_id):
                return VerifyResult.failure(VerifyReason.REVOKED_KEY)
            canonical = message.bound_id.payload.to_canonical_json()
            if not ed25519_verify(public, canonical, message.bound_id.signature_hex):
                return VerifyResult.failure(VerifyReason.BAD_SIGNATURE)

        # ------ L3: expiry + output binding + ancestry consistency ------
        if self.level >= DefenseLevel.L3:
            payload = message.bound_id.payload
            created = parse_rfc3339(payload.created_at)
            expires = parse_rfc3339(payload.expires_at)
            if now < created:
                return VerifyResult.failure(VerifyReason.NOT_YET_VALID)
            if now > expires:
                return VerifyResult.failure(VerifyReason.EXPIRED)
            if sha256_hex(message.presented_output) != payload.output_hash:
                return VerifyResult.failure(VerifyReason.OUTPUT_MISMATCH)
            if message.presented_chain_length != payload.ancestry_depth:
                return VerifyResult.failure(VerifyReason.ANCESTRY_LENGTH_MISMATCH)

        # ------ L4: nonce anti-replay ------
        if self.level >= DefenseLevel.L4:
            nonce = message.bound_id.payload.nonce
            if self.nonces.seen(nonce):
                return VerifyResult.failure(VerifyReason.REPLAYED_NONCE)
            self.nonces.mark(nonce)

        return VerifyResult.success()


# ---------- Wire-format helpers shared by the attack harness ----------

def pack_wire_bytes(bound_id: SignedBoundID, presented_output: bytes, chain_length: int) -> bytes:
    """Canonical wire-bytes format the channel MAC is computed over.

    Order and separators are fixed so that an attacker cannot silently insert
    a field. Any mutation of bound_id, presented_output, or chain_length
    changes the byte string and therefore the MAC.
    """
    return json.dumps(
        {
            "bound_id": bound_id.to_wire(),
            "presented_output_b64_len": len(presented_output),
            "presented_output_sha256": sha256_hex(presented_output),
            "chain_length": chain_length,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def build_stack(
    level: DefenseLevel,
) -> tuple[LayeredVerifier, InMemoryKeyDirectory, InMemoryNonceStore, dict[str, bytes]]:
    """Convenience factory — fresh stores for a fresh verifier at `level`."""
    keys = InMemoryKeyDirectory()
    nonces = InMemoryNonceStore()
    channels: dict[str, bytes] = {}
    v = LayeredVerifier(level=level, keys=keys, nonces=nonces, channel_secrets=channels)
    return v, keys, nonces, channels

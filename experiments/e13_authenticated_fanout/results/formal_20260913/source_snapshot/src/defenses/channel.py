"""A minimal authenticated-channel model for the L0-L4 sanity check.

Real TLS provides confidentiality + integrity + server authentication on a
given connection. For the three classical attacks (Tampering, ID Spoofing,
Instance Spoofing) what matters is:

  - Tampering (in-flight mutation): if the attacker is on-path between agent
    and verifier, a keyed MAC over the wire bytes catches it.
  - ID Spoofing / Instance Spoofing: the attacker uses their OWN channel.
    Their MAC is valid for that channel; TLS alone cannot help. Signature
    (L2) and output binding (L3) are what actually defend these.

We model TLS as an HMAC-SHA256 MAC keyed by a per-connection shared secret.
This is deliberately a simplification — we are not implementing TLS; we are
implementing the one property (channel integrity) that matters for the
three-attack sanity check.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass


def new_channel_secret() -> bytes:
    """Fresh 32-byte per-connection secret, analogous to a TLS session key."""
    return os.urandom(32)


def channel_mac(secret: bytes, data: bytes) -> str:
    """HMAC-SHA256 of `data` under `secret`, hex-encoded."""
    return hmac.new(secret, data, hashlib.sha256).hexdigest()


def channel_mac_valid(secret: bytes, data: bytes, mac_hex: str) -> bool:
    """Constant-time MAC check."""
    try:
        expected = hmac.new(secret, data, hashlib.sha256).digest()
        actual = bytes.fromhex(mac_hex)
    except ValueError:
        return False
    return hmac.compare_digest(expected, actual)


@dataclass(frozen=True)
class WirePacket:
    """What travels from agent/deployer to verifier.

    Fields:
      payload_bytes: the signed-bound-id JSON + presented output, concatenated
        into the bytes the channel MAC is computed over. The format is
        opaque here; the attack harness constructs it.
      mac_hex: the channel MAC (HMAC-SHA256 hex) over payload_bytes.
      channel_id: identifier for the connection (for logging / tracing).
    """

    payload_bytes: bytes
    mac_hex: str
    channel_id: str

"""Deployer: issues BoundIDs by computing output_hash deterministically and signing.

This is the critical "out of the LLM inference path" component. The deployer
runtime:
  1. Observes the agent's actual bytes (response / tool-call / stream chunk).
  2. Computes output_hash itself (never accepts a self-reported hash).
  3. Binds output_hash + a fresh nonce into the signed payload.
  4. Returns the SignedBoundID to the verifier alongside the output.

This mechanism is what Research Foundation v4 §3 calls out as the required
defense against IPI-Triggered Instance Spoofing: the agent cannot substitute
a hash of attacker-directed output because it never controls the hash.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from .agent_id import BindingScope, BoundIDPayload, SignedBoundID
from .crypto import KeyPair, new_nonce_hex, sha256_hex, sign


class Deployer:
    """Stateless-ish issuer. Holds a keypair + a deployer identifier.

    For the experimental prototype, key storage is in memory. Real deployments
    would back this with an HSM or KMS (out of scope per §4.5).
    """

    def __init__(self, deployer_name: str, keypair: KeyPair) -> None:
        self.deployer_name = deployer_name
        self.keypair = keypair

    # ------ Public API ------

    def issue(
        self,
        *,
        instance_id: str,
        system_id: str,
        output_bytes: bytes,
        binding_scope: BindingScope,
        ttl: timedelta = timedelta(hours=1),
        parent_id: Optional[str] = None,
        ancestry_depth: int = 0,
        certifications: tuple[str, ...] = (),
        incident_count: int = 0,
        prev_turn_hash: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> SignedBoundID:
        """Issue a new SignedBoundID that binds `output_bytes`.

        Args:
            output_bytes: the raw bytes the deployer observed the agent emit,
                for the declared `binding_scope`. These are hashed verbatim.
            binding_scope: which slice of agent behaviour these bytes represent.
            ttl: token lifetime.
            parent_id / ancestry_depth: delegation chain metadata (§4.4).
                Caller is responsible for passing consistent values; verifier
                cross-checks ancestry_depth against the presented chain.
            prev_turn_hash: SHA-256 of the previous turn's canonical payload
                for multi-turn linking (§4.3). None on turn 0.
        """
        now = now or datetime.now(timezone.utc)
        payload = BoundIDPayload(
            instance_id=instance_id,
            system_id=system_id,
            deployer=self.deployer_name,
            created_at=_rfc3339(now),
            expires_at=_rfc3339(now + ttl),
            parent_id=parent_id,
            ancestry_depth=ancestry_depth,
            output_hash=sha256_hex(output_bytes),
            nonce=new_nonce_hex(),
            binding_scope=binding_scope,
            certifications=certifications,
            incident_count=incident_count,
            prev_turn_hash=prev_turn_hash,
        )
        signature_hex = sign(self.keypair.private, payload.to_canonical_json())
        return SignedBoundID(
            payload=payload,
            signature_hex=signature_hex,
            key_id=self.keypair.key_id,
        )


def _rfc3339(dt: datetime) -> str:
    """Format a datetime as RFC 3339 UTC with seconds precision, trailing Z.

    Normalising the format keeps canonical JSON stable across environments.
    """
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    dt_utc = dt.astimezone(timezone.utc).replace(microsecond=0)
    # strftime with %z produces "+0000"; we want "Z".
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

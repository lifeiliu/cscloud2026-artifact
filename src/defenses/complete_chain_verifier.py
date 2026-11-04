"""Composable end-to-end verifier for v2 delegation chains.

``ChainBackend.verify_chain`` intentionally validates only the ancestry proof
shape.  Consumers must additionally verify each deployer-issued payload and
its released bytes.  This module keeps that composition explicit: it checks
primary signatures, expiry, replay nonces, output hashes, and then dispatches
to the selected ancestry backend.

It is a local prototype component.  The nonce store is process-local and the
registry is in-memory; production deployments still need a durable registry,
key rotation, and a shared replay store.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Protocol, Sequence

from ..core.crypto import key_id_from_public, sha256_hex, verify
from ..core.verifier import parse_rfc3339
from .chain_designs.base import ChainBackend, ChainNode, ChainVerifyError, DeployerRegistry


class NonceStore(Protocol):
    """Minimal durable-replay interface used by the complete verifier."""

    def seen(self, nonce: str) -> bool: ...
    def mark(self, nonce: str) -> None: ...


def verify_complete_chain(
    *,
    chain: Sequence[ChainNode],
    outputs: Sequence[bytes],
    backend: ChainBackend,
    deployers: DeployerRegistry,
    now: Optional[datetime] = None,
    nonces: Optional[NonceStore] = None,
) -> None:
    """Fail closed unless every receipt and every ancestry predicate holds.

    ``outputs[i]`` must be the exact bytes released by the runtime that issued
    ``chain[i]``.  Nonces are checked before any state is mutated and consumed
    only after all receipt and ancestry predicates succeed.
    """
    if not chain:
        raise ChainVerifyError("empty chain")
    if len(chain) != len(outputs):
        raise ChainVerifyError(
            f"chain/output length mismatch: {len(chain)} records, {len(outputs)} outputs"
        )

    current_time = now or datetime.now(timezone.utc)
    nonces_to_consume: list[str] = []
    seen_in_chain: set[str] = set()

    for index, (node, output) in enumerate(zip(chain, outputs)):
        signed = node.signed
        payload = signed.payload
        public = deployers.get_public(payload.deployer)
        if public is None:
            raise ChainVerifyError(
                f"record {index} deployer {payload.deployer!r} is not in the trusted registry"
            )
        if signed.key_id != key_id_from_public(public):
            raise ChainVerifyError(f"record {index} key_id does not match trusted deployer key")
        if deployers.is_revoked(signed.key_id):
            raise ChainVerifyError(f"record {index} signer key is revoked")
        if not verify(public, payload.to_canonical_json(), signed.signature_hex):
            raise ChainVerifyError(f"record {index} primary signature invalid")

        created = parse_rfc3339(payload.created_at)
        expires = parse_rfc3339(payload.expires_at)
        if current_time < created:
            raise ChainVerifyError(f"record {index} is not yet valid")
        if current_time > expires:
            raise ChainVerifyError(f"record {index} has expired")
        if sha256_hex(output) != payload.output_hash:
            raise ChainVerifyError(f"record {index} output hash mismatch")
        if payload.nonce in seen_in_chain:
            raise ChainVerifyError(f"record {index} reuses a nonce within the presented chain")
        if nonces is not None and nonces.seen(payload.nonce):
            raise ChainVerifyError(f"record {index} nonce was already consumed")
        seen_in_chain.add(payload.nonce)
        nonces_to_consume.append(payload.nonce)

    backend.verify_chain(chain=chain, deployers=deployers)
    if nonces is not None:
        for nonce in nonces_to_consume:
            nonces.mark(nonce)

"""ChainBackend — common interface for the three Paper 2 chain designs.

Design intent: a backend owns the *shape* of `chain_proof` for one design.
The Deployer (caller) supplies identity / output / lifecycle fields; the
backend computes the proof (`extend`) and validates it (`verify`).

This module is deliberately pure-data + pure-crypto: no LLM calls, no I/O.
Determinism property required by Paper 1 §3 (cryptographic operations execute
outside the LLM inference path) carries over.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import json
from typing import Optional, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ...core.agent_id_v2 import (
    BoundIDPayloadV2,
    ChainProof,
    SignedBoundIDV2,
)
from ...core.crypto import KeyPair, sha256_hex


# ---------- Errors ----------

class ChainExtendError(Exception):
    """Raised when a backend cannot legally extend a chain (e.g. missing parent)."""


class ChainVerifyError(Exception):
    """Raised when a chain fails structural verification.

    Note: signature/key/expiry/output failures are reported by the v1
    verifier via `VerifyResult`; this exception is reserved for chain-level
    structural failures (broken hash continuity, missing co-signers, etc.).
    """


# ---------- Helper structs ----------

@dataclass(frozen=True)
class ChainNode:
    """One link in a presented delegation chain.

    A verifier consumes `Sequence[ChainNode]` ordered from root to leaf
    (root at index 0, leaf at index -1). The backend's `verify_chain`
    inspects each node's `chain_proof` against its predecessor.
    """

    signed: SignedBoundIDV2  # full signed payload + sig + key_id


class DeployerRegistry:
    """Multi-deployer key directory for signed_dag.

    Maps deployer_name -> KeyPair (for owners) or just the public key
    (for verifiers). Trivial in-memory implementation; production would
    delegate to a JWKS-style endpoint per Paper 1 §5.
    """

    def __init__(self) -> None:
        self._owners: dict[str, KeyPair] = {}
        self._publics: dict[str, Ed25519PublicKey] = {}
        self._revoked_key_ids: set[str] = set()

    def register_owner(self, deployer_name: str, keypair: KeyPair) -> None:
        """Register a deployer that this process can SIGN on behalf of."""
        self._owners[deployer_name] = keypair
        self._publics[deployer_name] = keypair.public

    def register_public(
        self, deployer_name: str, public: Ed25519PublicKey
    ) -> None:
        """Register a deployer for VERIFICATION only (no private key)."""
        self._publics[deployer_name] = public

    def get_owner(self, deployer_name: str) -> Optional[KeyPair]:
        return self._owners.get(deployer_name)

    def get_public(self, deployer_name: str) -> Optional[Ed25519PublicKey]:
        return self._publics.get(deployer_name)

    def revoke_key(self, key_id: str) -> None:
        """Mark a registered deployer key unacceptable to this verifier."""
        self._revoked_key_ids.add(key_id)

    def is_revoked(self, key_id: str) -> bool:
        return key_id in self._revoked_key_ids


class PinnedDeployerRegistry(DeployerRegistry):
    """Read-only registry for the E4b pinned-registry ablation.

    Pre-populated with legitimate deployers' public keys only. The attacker
    cannot register new deployers or access private keys. This forces the
    attacker to exploit chain-shape or lifecycle logic rather than
    side-stepping the threat model by becoming a deployer.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pinned = True

    def register_owner(self, deployer_name: str, keypair: KeyPair) -> None:
        if self._pinned:
            raise RuntimeError(
                f"Cannot register deployer {deployer_name!r}: registry is pinned (read-only)."
            )
        super().register_owner(deployer_name, keypair)

    def register_public(
        self, deployer_name: str, public: Ed25519PublicKey
    ) -> None:
        if self._pinned:
            raise RuntimeError(
                f"Cannot register deployer {deployer_name!r}: registry is pinned (read-only)."
            )
        super().register_public(deployer_name, public)

    @classmethod
    def from_keypairs(cls, deployers: dict[str, KeyPair]) -> "PinnedDeployerRegistry":
        """Create a pinned registry exposing only public keys."""
        reg = cls()
        reg._pinned = False
        for name, kp in deployers.items():
            reg.register_public(name, kp.public)
        reg._pinned = True
        return reg


# ---------- The interface ----------

class ChainBackend(ABC):
    """One chain design. Stateless across calls — all state is on the chain."""

    design_name: str  # subclass sets to "linked_list" / "merkle" / "signed_dag"

    @abstractmethod
    def build_proof_for_root(self) -> ChainProof:
        """Construct the chain_proof for a root node (no parent)."""

    @abstractmethod
    def build_proof_for_extend(
        self,
        *,
        parent: BoundIDPayloadV2,
        child_instance_id: str,
        ancestors: Sequence[BoundIDPayloadV2],
        deployers: DeployerRegistry,
        co_signer_names: Sequence[str] = (),
        child_payload: Optional[BoundIDPayloadV2] = None,
    ) -> ChainProof:
        """Construct the chain_proof for a non-root extend.

        Args:
            parent: the immediate parent's full payload (for hash / Merkle input).
            child_instance_id: bound into signed_dag's per-edge signature input
                so that an edge signature cannot be lifted to a different child.
            ancestors: full ordered ancestor list root..parent. Merkle uses all.
                Linked-list uses only `parent`. signed_dag uses only `parent`.
            deployers: registry to look up co-signer private keys.
            co_signer_names: which deployers should co-sign this edge
                (signed_dag only). Empty for linked_list / merkle.
            child_payload: a provisional child payload with the final child
                fields and co-signer policy, but without edge signatures.
                signed_dag uses it to bind the complete child transcript;
                linked-list and Merkle backends ignore it.
        """

    @abstractmethod
    def verify_chain(
        self,
        *,
        chain: Sequence[ChainNode],
        deployers: DeployerRegistry,
    ) -> None:
        """Verify the entire `chain` is structurally well-formed for this design.

        Raises ChainExtendError or ChainVerifyError on structural problems.
        Does NOT check Ed25519 signatures on the *primary* signed payload — that
        is the v1 Verifier's job (Paper 1's `core.verifier.Verifier.verify`).
        This method only checks the chain-shape proof (`chain_proof`).

        For signed_dag: this method DOES check the per-edge co-signer signatures
        because they live inside chain_proof itself, not on the primary signature.
        """


# ---------- Helper: canonical input bytes for a per-edge signature ----------

def edge_signing_input(parent: BoundIDPayloadV2, child: BoundIDPayloadV2) -> bytes:
    """Return a domain-separated, non-circular Signed-DAG edge transcript.

    An edge signature binds the SHA-256 digest of the *entire* parent receipt
    and all child fields, including output hash, expiry, nonce, lifecycle, and
    the agreed co-signer set.  Only ``chain_proof.signatures`` is excluded:
    including it would create a signature cycle because those are precisely
    the signatures being generated.  The child's primary signature remains
    outside this message and seals the finished payload, including the edge
    signatures.

    This is deliberately a canonical JSON manifest rather than concatenated
    bytes.  The explicit domain tag prevents a valid edge signature from being
    reinterpreted by another protocol.
    """
    unsigned_proof = replace(child.chain_proof, signatures=())
    child_without_edge_signatures = replace(child, chain_proof=unsigned_proof)
    manifest = {
        "domain": "com.cscloud2026.signed-dag.edge.v1",
        "parent_payload_sha256": sha256_hex(parent.to_canonical_json()),
        "child_payload_without_edge_signatures": json.loads(
            child_without_edge_signatures.to_canonical_json().decode("utf-8")
        ),
    }
    return json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

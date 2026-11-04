"""Multi-deployer test harness for Paper 2.

Wraps the Paper 1 `Deployer` and Paper 2's `ChainBackend` + `DeployerRegistry`
into one builder that:
  - holds N deployers, each with its own keypair and key_id
  - extends a chain of v2 payloads under a chosen ChainBackend
  - lets a caller specify which deployer hosts each child
  - returns the ordered chain of `ChainNode` for use with `Verifier`
    (Paper 1 verifier handles signature/expiry/output/nonce; the
    `ChainBackend.verify_chain` method handles chain-shape proofs)

This module is test/experiment-only — it is NOT a model of a real production
multi-deployer protocol. In production, edges would be co-signed via an out-
of-band protocol (HTTP callback, message bus). Here we centralise key access
in one process for reproducibility.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from ..core.agent_id_v2 import (
    BoundIDPayloadV2,
    ChainProof,
    LifecycleRecord,
    SignedBoundIDV2,
    make_lifecycle_new,
)
from ..core.crypto import KeyPair, generate_keypair, new_nonce_hex, sha256_hex, sign
from ..defenses.chain_designs.base import (
    ChainBackend,
    ChainNode,
    DeployerRegistry,
)


# ---------- Deployer wrapper that issues v2 payloads ----------

class DeployerV2:
    """Minimal deployer that issues v2 SignedBoundIDV2 payloads.

    Mirrors the Paper 1 `Deployer.issue` API but takes an explicit
    `chain_proof` and `lifecycle` because those are computed by the
    backend / caller, not the deployer.
    """

    def __init__(self, deployer_name: str, keypair: KeyPair) -> None:
        self.deployer_name = deployer_name
        self.keypair = keypair

    def issue(
        self,
        *,
        instance_id: str,
        system_id: str,
        output_bytes: bytes,
        binding_scope: str,
        chain_proof: ChainProof,
        lifecycle: LifecycleRecord,
        ttl: timedelta = timedelta(hours=1),
        parent_id: Optional[str] = None,
        ancestry_depth: int = 0,
        certifications: tuple[str, ...] = (),
        incident_count: int = 0,
        prev_turn_hash: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> SignedBoundIDV2:
        payload = self.build_payload(
            instance_id=instance_id,
            system_id=system_id,
            output_bytes=output_bytes,
            binding_scope=binding_scope,
            chain_proof=chain_proof,
            lifecycle=lifecycle,
            ttl=ttl,
            parent_id=parent_id,
            ancestry_depth=ancestry_depth,
            certifications=certifications,
            incident_count=incident_count,
            prev_turn_hash=prev_turn_hash,
            now=now,
        )
        return self.sign_payload(payload)

    def build_payload(
        self,
        *,
        instance_id: str,
        system_id: str,
        output_bytes: bytes,
        binding_scope: str,
        chain_proof: ChainProof,
        lifecycle: LifecycleRecord,
        ttl: timedelta = timedelta(hours=1),
        parent_id: Optional[str] = None,
        ancestry_depth: int = 0,
        certifications: tuple[str, ...] = (),
        incident_count: int = 0,
        prev_turn_hash: Optional[str] = None,
        now: Optional[datetime] = None,
        nonce: Optional[str] = None,
    ) -> BoundIDPayloadV2:
        """Build an unsigned payload for a co-signing protocol.

        Signed-DAG must construct this exact payload before co-signers can
        authorize its edge.  ``sign_payload`` is intentionally separate so
        the final primary signature includes the resulting edge signatures.
        """
        now = now or datetime.now(timezone.utc)
        return BoundIDPayloadV2(
            instance_id=instance_id, system_id=system_id,
            deployer=self.deployer_name, created_at=_rfc3339(now),
            expires_at=_rfc3339(now + ttl), parent_id=parent_id,
            ancestry_depth=ancestry_depth, chain_proof=chain_proof,
            output_hash=sha256_hex(output_bytes), nonce=nonce or new_nonce_hex(),
            binding_scope=binding_scope,  # type: ignore[arg-type]
            lifecycle=lifecycle, certifications=certifications,
            incident_count=incident_count, prev_turn_hash=prev_turn_hash,
        )

    def sign_payload(self, payload: BoundIDPayloadV2) -> SignedBoundIDV2:
        """Primary-sign a payload after all co-signer material is present."""
        sig = sign(self.keypair.private, payload.to_canonical_json())
        return SignedBoundIDV2(
            payload=payload,
            signature_hex=sig,
            key_id=self.keypair.key_id,
        )


def _rfc3339(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


# ---------- Chain builder ----------

class ChainBuilder:
    """Build benign chains across one or more deployers under a chosen backend.

    Usage:
        h = MultiDeployerHarness.with_deployers(["D_A", "D_B"])
        b = ChainBuilder(h, backend=LinkedListBackend())
        chain = b.start_root("D_A", "agent-0", "claude-3-sonnet", b"hello")
        chain = b.extend(chain, host="D_B", instance_id="agent-1",
                         system_id="claude-3-sonnet", output=b"step 1")
        # chain is a list[ChainNode]
    """

    def __init__(self, harness: "MultiDeployerHarness", backend: ChainBackend) -> None:
        self.harness = harness
        self.backend = backend

    def start_root(
        self,
        host: str,
        instance_id: str,
        system_id: str,
        output: bytes,
        *,
        binding_scope: str = "per_session",
        lifecycle: Optional[LifecycleRecord] = None,
    ) -> list[ChainNode]:
        """Issue the root node, returning a single-element chain."""
        d = self.harness.get_deployer(host)
        proof = self.backend.build_proof_for_root()
        signed = d.issue(
            instance_id=instance_id,
            system_id=system_id,
            output_bytes=output,
            binding_scope=binding_scope,
            chain_proof=proof,
            lifecycle=lifecycle or make_lifecycle_new(),
            parent_id=None,
            ancestry_depth=0,
        )
        return [ChainNode(signed=signed)]

    def extend(
        self,
        chain: Sequence[ChainNode],
        *,
        host: str,
        instance_id: str,
        system_id: str,
        output: bytes,
        binding_scope: str = "per_session",
        lifecycle: Optional[LifecycleRecord] = None,
        co_signer_names: Sequence[str] = (),
    ) -> list[ChainNode]:
        """Append one new node to `chain`. Caller supplies co_signer_names
        only for signed_dag; for linked_list / merkle pass an empty tuple
        (the default).
        """
        if not chain:
            raise ValueError("cannot extend an empty chain — use start_root")
        d = self.harness.get_deployer(host)
        parent_payload = chain[-1].signed.payload
        ancestors = [n.signed.payload for n in chain]

        if self.backend.design_name == "signed_dag":
            # Signed-DAG edge signatures bind all child fields but cannot
            # include themselves.  Build an exact provisional child whose
            # proof declares the final co-signer set and has an empty
            # signature list, then replace only that list before adding the
            # primary deployer signature.
            provisional = d.build_payload(
                instance_id=instance_id,
                system_id=system_id,
                output_bytes=output,
                binding_scope=binding_scope,
                chain_proof=ChainProof(
                    design="signed_dag", co_signers=tuple(co_signer_names)
                ),
                lifecycle=lifecycle or make_lifecycle_new(),
                parent_id=parent_payload.instance_id,
                ancestry_depth=parent_payload.ancestry_depth + 1,
            )
            proof = self.backend.build_proof_for_extend(
                parent=parent_payload,
                child_instance_id=instance_id,
                ancestors=ancestors,
                deployers=self.harness.registry,
                co_signer_names=co_signer_names,
                child_payload=provisional,
            )
            signed = d.sign_payload(replace(provisional, chain_proof=proof))
        else:
            proof = self.backend.build_proof_for_extend(
                parent=parent_payload,
                child_instance_id=instance_id,
                ancestors=ancestors,
                deployers=self.harness.registry,
                co_signer_names=co_signer_names,
            )
            signed = d.issue(
                instance_id=instance_id,
                system_id=system_id,
                output_bytes=output,
                binding_scope=binding_scope,
                chain_proof=proof,
                lifecycle=lifecycle or make_lifecycle_new(),
                parent_id=parent_payload.instance_id,
                ancestry_depth=parent_payload.ancestry_depth + 1,
            )
        return list(chain) + [ChainNode(signed=signed)]


# ---------- Harness ----------

class MultiDeployerHarness:
    """A bag of DeployerV2s + a DeployerRegistry, all sharing the same view.

    Construction options:
      - `MultiDeployerHarness.with_deployers([...])` — generates fresh keypairs
      - `MultiDeployerHarness({"D_A": kp_a, "D_B": kp_b})` — uses given keypairs
    """

    def __init__(self, deployers: dict[str, KeyPair]) -> None:
        self._deployers: dict[str, DeployerV2] = {}
        self.registry = DeployerRegistry()
        for name, kp in deployers.items():
            self._deployers[name] = DeployerV2(name, kp)
            self.registry.register_owner(name, kp)

    @classmethod
    def with_deployers(cls, names: Sequence[str]) -> "MultiDeployerHarness":
        return cls({name: generate_keypair() for name in names})

    def get_deployer(self, name: str) -> DeployerV2:
        d = self._deployers.get(name)
        if d is None:
            raise KeyError(f"unknown deployer {name!r}")
        return d

    def names(self) -> list[str]:
        return list(self._deployers.keys())

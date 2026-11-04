"""AgentID v2 schema — Paper 2 extension for multi-hop ancestry chains.

Extends Paper 1's `BoundIDPayload` (src/core/agent_id.py) with two new blocks:

  1. `chain_proof`  — per-hop or per-chain cryptographic proof bound to the
                      delegation lineage. Three variants:
                        - linked_list : prev_payload_hash + per-hop signature
                        - merkle      : root hash over the ordered ancestor list
                        - signed_dag  : deployer cross-signatures on each edge
  2. `lifecycle`    — invariant-bearing record of the lifecycle event that
                      created this instance (new / reload / branch / composite /
                      fine_tune), with event-specific fields.

Design rules (carried over from Paper 1):
  - Security-critical fields are signed as ONE unit (Ed25519 over canonical JSON).
  - Routing hints stay in the unsigned `RoutingHints` envelope (LDP rule).
  - Canonicalisation: sort_keys=True, no whitespace, ensure_ascii=True. Two
    semantically-equal payloads MUST produce byte-identical canonical JSON.
  - The agent NEVER computes its own chain_proof; the deployer (or co-deployers
    for signed_dag) computes and signs it outside the LLM inference path.

Anti-attack scope (paper §3 / kickoff §6):
  - B1 orphan-child injection      — defended by per-hop signature (parent must exist)
  - B2 chain-break splice          — defended by hash continuity (linked_list / merkle)
  - B3 cross-chain merge           — defended by deployer cross-signature (signed_dag)
  - B4 ancestor revision           — defended by Merkle root immutability
  - B5 sibling fanout obfuscation  — anomaly detection layer above the crypto

Out of scope for this module:
  - Selective disclosure on ancestry links (Paper 3 / E5)
  - Hardware-attested keys (Future Direction 1)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


# ---------- Type aliases ----------

ChainDesign = Literal[
    "linked_list", "merkle", "signed_dag",
    # Baselines (E2 — §6 ablation study)
    "no_chain", "hash_only", "naive_sig",
]
LifecycleEvent = Literal["new", "reload", "branch", "composite", "fine_tune"]
BindingScope = Literal["per_session", "per_turn", "per_tool_call", "per_n_tokens"]


# ---------- Sub-payloads ----------

@dataclass(frozen=True)
class ChainProof:
    """Cryptographic proof of position in the delegation chain.

    Only one of `prev_hash` / `merkle_root` / (`co_signers`+`signatures`) is
    populated, determined by `design`. All fields stay present (set to None or
    empty tuple) so the canonical JSON encoding has a stable key set across
    designs — this matters for cross-design comparison runs in Week 2.

    Field semantics by design:
      linked_list :
          prev_hash  = SHA-256 hex of the parent's canonical payload.
                       None iff this is a root node.
          (merkle_root, co_signers, signatures) MUST be None / empty.
      merkle :
          merkle_root = SHA-256 hex of the binary Merkle tree built over the
                        canonical payloads of all ancestors in order
                        (root first, immediate parent last). Empty for root.
          (prev_hash, co_signers, signatures) MUST be None / empty.
      signed_dag :
          co_signers = ordered tuple of deployer names that co-signed this edge,
                       e.g. ("deployer_A", "deployer_B"). Length >= 1; for
                       same-deployer hops, length is 1.
          signatures = ordered tuple of Ed25519 signature hex, one per co_signer,
                       each signing a domain-separated manifest containing the
                       parent-payload digest and the full child payload except
                       this signature list (excluded to avoid a signature cycle).
          (prev_hash, merkle_root) MUST be None.

    The verifier dispatches on `design` and ignores fields irrelevant to that
    design — but they must still be present and equal to None/() so the
    signed canonical JSON byte string is deterministic.
    """

    design: ChainDesign
    prev_hash: Optional[str] = None
    merkle_root: Optional[str] = None
    co_signers: tuple[str, ...] = ()
    signatures: tuple[str, ...] = ()


@dataclass(frozen=True)
class LifecycleRecord:
    """The lifecycle event that created this instance.

    Field semantics by event:
      new        : all extra fields default. branch_index and reload_count == 0.
      reload     : reload_count > 0; instance_id MUST equal the original.
      branch     : branch_index >= 1; parent_id (in BoundIDPayloadV2) MUST be
                   set to the branched-from instance.
      composite  : constituents non-empty (>=2); each MUST be co-signed, which
                   the deployer encodes by reusing chain_proof.signed_dag.
      fine_tune  : base_system_id != system_id; preserves provenance link to
                   the base model family.

    Note: this record describes the *event*, not the chain. Composite-merge
    cross-signatures live in `chain_proof` (signed_dag design); this struct
    just names the constituents.
    """

    event: LifecycleEvent
    branch_index: int = 0
    reload_count: int = 0
    constituents: tuple[str, ...] = ()  # composite only
    base_system_id: Optional[str] = None  # fine_tune only


# ---------- Top-level signed payload ----------

@dataclass(frozen=True)
class BoundIDPayloadV2:
    """Paper 2's signed payload. Adds `chain_proof` and `lifecycle` to v1.

    Naming: kept distinct from v1's `BoundIDPayload` so we can run regression
    tests against the unmodified v1 path during Week 1, and so a verifier can
    dispatch on schema version explicitly. Both versions share `to_canonical_json`
    semantics (RFC 8785 / JCS spirit: sort_keys, no whitespace, ensure_ascii).
    """

    # ----- Identity (unchanged from v1) -----
    instance_id: str          # e.g. "claude-3-sonnet:sess_2026050801"
    system_id: str            # e.g. "claude-3-sonnet"
    deployer: str             # e.g. "anthropic"

    # ----- Temporal bounds (unchanged) -----
    created_at: str           # RFC 3339 UTC, "2026-05-08T10:00:00Z"
    expires_at: str

    # ----- Ancestry (extended) -----
    parent_id: Optional[str]
    ancestry_depth: int       # 0 for root; verifier cross-checks against
                              # presented chain length (closes drop-hops)
    chain_proof: ChainProof   # NEW in v2 — cryptographic proof of position

    # ----- Behavioural binding (unchanged) -----
    output_hash: str          # SHA-256 hex of the bound content
    nonce: str                # 128-bit hex; single-use anti-replay
    binding_scope: BindingScope

    # ----- Lifecycle (NEW in v2) -----
    lifecycle: LifecycleRecord

    # ----- Policy constraints (NEW — B5 defense) -----
    max_children: Optional[int] = None  # if set, verifier rejects >k siblings from same parent

    # ----- Attributes (unchanged) -----
    certifications: tuple[str, ...] = ()
    incident_count: int = 0

    # ----- Multi-turn linkage (unchanged from v1) -----
    prev_turn_hash: Optional[str] = None

    # ----- Schema version tag — explicit so verifier can dispatch -----
    schema_version: Literal["v2"] = "v2"

    def to_canonical_json(self) -> bytes:
        """Deterministic JSON encoding suitable for signing.

        Contract: byte-identical output for semantically equal payloads.
        Tuples normalised to list to match v1 semantics.
        """
        d = asdict(self)
        # Normalise tuples (asdict preserves them; json renders them as arrays,
        # but be explicit to avoid surprises if a future Python version changes).
        d["certifications"] = list(self.certifications)
        d["chain_proof"]["co_signers"] = list(self.chain_proof.co_signers)
        d["chain_proof"]["signatures"] = list(self.chain_proof.signatures)
        d["lifecycle"]["constituents"] = list(self.lifecycle.constituents)
        return json.dumps(
            d,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")


# ---------- Wire envelope (parallel to v1's SignedBoundID + AgentIDEnvelope) ----------

@dataclass(frozen=True)
class SignedBoundIDV2:
    """A v2 payload + its primary signature + signer key id.

    Note: `signature_hex` here is the *primary* deployer's signature over the
    full payload (including chain_proof.signatures, which are themselves
    Ed25519 signatures over the parent edge). For signed_dag, the primary
    signature still anchors the payload as a whole; co-signer signatures
    inside chain_proof anchor each *edge*.
    """

    payload: BoundIDPayloadV2
    signature_hex: str
    key_id: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "payload": json.loads(self.payload.to_canonical_json().decode("utf-8")),
            "signature_hex": self.signature_hex,
            "key_id": self.key_id,
        }


# ---------- Constructor helpers ----------

def make_chain_proof_linked_list(prev_hash: Optional[str]) -> ChainProof:
    """Linked-list chain proof. `prev_hash` is None for a root node."""
    return ChainProof(design="linked_list", prev_hash=prev_hash)


def make_chain_proof_merkle(merkle_root: Optional[str]) -> ChainProof:
    """Merkle chain proof. `merkle_root` is None / empty for a root node."""
    return ChainProof(design="merkle", merkle_root=merkle_root)


def make_chain_proof_signed_dag(
    co_signers: tuple[str, ...], signatures: tuple[str, ...]
) -> ChainProof:
    """Signed-DAG chain proof. Both tuples must be non-empty and equal-length.

    For a root node, pass empty tuples (the verifier treats len()==0 as root).
    """
    if len(co_signers) != len(signatures):
        raise ValueError("co_signers and signatures must have equal length")
    return ChainProof(
        design="signed_dag",
        co_signers=co_signers,
        signatures=signatures,
    )


def make_lifecycle_new() -> LifecycleRecord:
    return LifecycleRecord(event="new")


def make_lifecycle_reload(reload_count: int) -> LifecycleRecord:
    if reload_count < 1:
        raise ValueError("reload_count must be >= 1 for a reload event")
    return LifecycleRecord(event="reload", reload_count=reload_count)


def make_lifecycle_branch(branch_index: int) -> LifecycleRecord:
    if branch_index < 1:
        raise ValueError("branch_index must be >= 1 for a branch event")
    return LifecycleRecord(event="branch", branch_index=branch_index)


def make_lifecycle_composite(constituents: tuple[str, ...]) -> LifecycleRecord:
    if len(constituents) < 2:
        raise ValueError("composite requires at least 2 constituents")
    return LifecycleRecord(event="composite", constituents=constituents)


def make_lifecycle_fine_tune(base_system_id: str) -> LifecycleRecord:
    if not base_system_id:
        raise ValueError("fine_tune requires non-empty base_system_id")
    return LifecycleRecord(event="fine_tune", base_system_id=base_system_id)

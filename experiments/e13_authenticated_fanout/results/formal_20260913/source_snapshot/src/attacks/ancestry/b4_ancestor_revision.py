"""B4 — Ancestor revision.

Threat: attacker mutates an already-extended ancestor's payload (e.g. flips
a permission bit, alters a certifications tuple) hoping the chain still
verifies if downstream proofs are recomputed against the *new* ancestor.

Defence surface:
  - linked_list : detected only if the attacker also has to re-issue every
                  downstream child (prev_hash recomputed). If the attacker
                  controls the deployer's key for the downstream children,
                  this defence collapses; if not, downstream signatures
                  fail. NOTE: this is the design's known weakness — a
                  same-deployer chain where the deployer is compromised
                  allows ancestor revision with re-signing.
  - merkle      : detected immediately. Mutating ancestor i invalidates
                  every merkle_root at hops > i, simultaneously, even if
                  the attacker re-signs each downstream payload primary
                  signature.
  - signed_dag  : detected. Mutating ancestor i changes the parent payload
                  bytes that downstream edges' co-signers signed; their
                  signatures on the now-changed edges fail to verify
                  (Paper 1 binds the canonical-JSON bytes of the parent).

This is the attack that earns Merkle its overhead.

Attack variants:
  output_hash    — flip the output_hash of an internal ancestor.
  certifications — add a fake certification to an ancestor.
  deployer       — change the ancestor's deployer field.
  parent_id      — change the ancestor's parent_id to a random value.

In all variants the attacker does NOT re-sign the modified ancestor (no access
to deployer key). The chain_proof fields on downstream nodes are left as-is.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Literal, Sequence

from ...core.crypto import sha256_hex
from ...defenses.chain_designs.base import ChainNode


def attack_b4_ancestor_revision(
    chain: Sequence[ChainNode],
    *,
    revise_index: int = 1,
    field: Literal[
        "output_hash", "certifications", "deployer", "parent_id"
    ] = "output_hash",
) -> list[ChainNode]:
    """Mutate an ancestor's payload in-place (without re-signing).

    Args:
        chain: a benign chain of length >= 3.
        revise_index: which ancestor to modify. Must satisfy
                      0 < revise_index < len(chain) - 1 (i.e., a true
                      internal ancestor, not root or leaf).
        field: which payload field to mutate.

    Returns:
        Chain with the ancestor's payload mutated. Merkle should always
        detect this; linked_list detects it at hop revise_index+1 via
        prev_hash mismatch; signed_dag detects at hop revise_index+1
        via co-signer signature mismatch.
    """
    if len(chain) < 3:
        raise ValueError("B4 requires chain length >= 3")
    if not (0 < revise_index < len(chain) - 1):
        raise ValueError(
            f"revise_index must be a true internal ancestor "
            f"(0 < idx < {len(chain) - 1}), got {revise_index}"
        )

    mutated = list(chain)
    old_payload = mutated[revise_index].signed.payload

    if field == "output_hash":
        new_payload = dataclasses.replace(
            old_payload,
            output_hash=sha256_hex(b"revised-output-" + uuid.uuid4().bytes),
        )
    elif field == "certifications":
        new_payload = dataclasses.replace(
            old_payload,
            certifications=old_payload.certifications + ("fabricated_cert",),
        )
    elif field == "deployer":
        new_payload = dataclasses.replace(
            old_payload,
            deployer=f"rogue-deployer-{uuid.uuid4().hex[:6]}",
        )
    elif field == "parent_id":
        new_payload = dataclasses.replace(
            old_payload,
            parent_id=f"nonexistent-{uuid.uuid4().hex[:8]}",
        )
    else:
        raise ValueError(f"unknown B4 field: {field!r}")

    mutated[revise_index] = ChainNode(
        signed=dataclasses.replace(
            mutated[revise_index].signed,
            payload=new_payload,
        )
    )

    return mutated

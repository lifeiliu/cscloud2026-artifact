"""Paper 2 chain-design backends.

Three primary implementations sharing one `ChainBackend` interface:
  - LinkedListBackend  : prev_payload_hash + per-hop Ed25519 signature
  - MerkleBackend      : root hash over the ordered ancestor list
  - SignedDAGBackend   : deployer cross-signatures on each edge

Three baseline backends for ablation study (E2):
  - NoChainBackend     : accept any chain unconditionally
  - HashOnlyBackend    : verify output_hash presence only
  - NaiveSigBackend    : verify per-node signatures only (no chain shape)

Selection is by `chain_proof.design` on a v2 payload; the verifier dispatches.
"""

from .base import (
    ChainBackend,
    ChainExtendError,
    ChainNode,
    ChainVerifyError,
    DeployerRegistry,
)
from .baselines import HashOnlyBackend, NaiveSigBackend, NoChainBackend
from .linked_list import LinkedListBackend
from .merkle import MerkleBackend
from .signed_dag import SignedDAGBackend


__all__ = [
    "ChainBackend",
    "ChainExtendError",
    "ChainNode",
    "ChainVerifyError",
    "DeployerRegistry",
    # Primary designs
    "LinkedListBackend",
    "MerkleBackend",
    "SignedDAGBackend",
    # Baselines (E2)
    "NoChainBackend",
    "HashOnlyBackend",
    "NaiveSigBackend",
]

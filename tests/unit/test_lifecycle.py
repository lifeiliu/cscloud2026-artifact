"""Tests for lifecycle invariant verifier (Paper 2, §4).

Covers L1–L5 invariants: positive paths, per-invariant violations,
cross-node constraints, composability with chain backends, and
error message quality (the red-team LLM reads these).
"""

from __future__ import annotations

import dataclasses

import pytest

from src.core.agent_id_v2 import (
    LifecycleRecord,
    make_lifecycle_branch,
    make_lifecycle_composite,
    make_lifecycle_fine_tune,
    make_lifecycle_new,
    make_lifecycle_reload,
)
from src.defenses.chain_designs import (
    ChainNode,
    LinkedListBackend,
    MerkleBackend,
    SignedDAGBackend,
)
from src.defenses.lifecycle import LifecycleVerifier, LifecycleVerifyError, verify_full
from src.runtime.multi_deployer import ChainBuilder, MultiDeployerHarness


# ---------- Fixtures ----------

@pytest.fixture
def harness():
    return MultiDeployerHarness.with_deployers(["D_A", "D_B"])


@pytest.fixture
def ll_backend():
    return LinkedListBackend()


def _build_root(harness, backend, lifecycle=None, host="D_A"):
    b = ChainBuilder(harness, backend=backend)
    return b.start_root(
        host, "agent-0", "claude-3-sonnet", b"root-output",
        lifecycle=lifecycle,
    )


def _extend(harness, backend, chain, instance_id, lifecycle=None,
            host="D_A", system_id="claude-3-sonnet", output=b"step"):
    b = ChainBuilder(harness, backend=backend)
    co = tuple(harness.names()) if backend.design_name == "signed_dag" else ()
    return b.extend(
        chain, host=host, instance_id=instance_id,
        system_id=system_id, output=output,
        lifecycle=lifecycle, co_signer_names=co,
    )


def _mutate_payload(chain, idx, **kwargs):
    """Return a new chain with the payload at `idx` mutated (unsigned)."""
    result = list(chain)
    old_node = result[idx]
    old_payload = old_node.signed.payload
    new_payload = dataclasses.replace(old_payload, **kwargs)
    new_signed = dataclasses.replace(old_node.signed, payload=new_payload)
    result[idx] = ChainNode(signed=new_signed)
    return result


def _mutate_lifecycle(chain, idx, **kwargs):
    """Return a new chain with the lifecycle record at `idx` mutated."""
    old_lc = chain[idx].signed.payload.lifecycle
    new_lc = dataclasses.replace(old_lc, **kwargs)
    return _mutate_payload(chain, idx, lifecycle=new_lc)


# ============================================================================
# L1: new event
# ============================================================================

class TestNewEvent:
    """L1 invariants for lifecycle.event == 'new'."""

    def test_valid_root(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        LifecycleVerifier().verify(chain=chain)  # no error

    def test_valid_non_root_new(self, harness, ll_backend):
        """A non-root 'new' node is allowed (new agent in delegation)."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "agent-1",
                        lifecycle=make_lifecycle_new())
        LifecycleVerifier().verify(chain=chain)

    def test_l1c_nonzero_reload_count(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _mutate_lifecycle(chain, 0, reload_count=1)
        with pytest.raises(LifecycleVerifyError, match="L1c"):
            LifecycleVerifier().verify(chain=chain)

    def test_l1d_nonzero_branch_index(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _mutate_lifecycle(chain, 0, branch_index=1)
        with pytest.raises(LifecycleVerifyError, match="L1d"):
            LifecycleVerifier().verify(chain=chain)

    def test_l1e_nonempty_constituents(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _mutate_lifecycle(chain, 0, constituents=("x", "y"))
        with pytest.raises(LifecycleVerifyError, match="L1e"):
            LifecycleVerifier().verify(chain=chain)

    def test_l1f_base_system_id_set(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _mutate_lifecycle(chain, 0, base_system_id="old-model")
        with pytest.raises(LifecycleVerifyError, match="L1f"):
            LifecycleVerifier().verify(chain=chain)

    def test_l1b_root_with_parent(self, harness, ll_backend):
        """Root 'new' node must have parent_id=None."""
        chain = _build_root(harness, ll_backend)
        chain = _mutate_payload(chain, 0, parent_id="phantom")
        with pytest.raises(LifecycleVerifyError, match="L1b"):
            LifecycleVerifier().verify(chain=chain)

    def test_l1a_root_nonzero_depth(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _mutate_payload(chain, 0, ancestry_depth=1)
        with pytest.raises(LifecycleVerifyError, match="L1a"):
            LifecycleVerifier().verify(chain=chain)


# ============================================================================
# L2: reload event
# ============================================================================

class TestReloadEvent:
    """L2 invariants for lifecycle.event == 'reload'."""

    def test_valid_reload(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _extend(
            harness, ll_backend, chain, "agent-0",  # same instance_id
            lifecycle=make_lifecycle_reload(1),
        )
        LifecycleVerifier().verify(chain=chain)

    def test_l2a_zero_reload_count(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "agent-0")
        chain = _mutate_lifecycle(chain, 1, event="reload", reload_count=0)
        with pytest.raises(LifecycleVerifyError, match="L2a"):
            LifecycleVerifier().verify(chain=chain)

    def test_l2b_reload_at_root(self, harness, ll_backend):
        """Reload cannot be the root node."""
        chain = _build_root(harness, ll_backend)
        chain = _mutate_lifecycle(chain, 0, event="reload", reload_count=1)
        with pytest.raises(LifecycleVerifyError, match="L2b"):
            LifecycleVerifier().verify(chain=chain)

    def test_l2c_different_instance_id(self, harness, ll_backend):
        """Reload must preserve instance_id."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(
            harness, ll_backend, chain, "agent-DIFFERENT",
            lifecycle=make_lifecycle_reload(1),
        )
        with pytest.raises(LifecycleVerifyError, match="L2c"):
            LifecycleVerifier().verify(chain=chain)

    def test_l2d_different_system_id(self, harness, ll_backend):
        """Reload must preserve system_id."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(
            harness, ll_backend, chain, "agent-0",
            system_id="different-model",
            lifecycle=make_lifecycle_reload(1),
        )
        with pytest.raises(LifecycleVerifyError, match="L2d"):
            LifecycleVerifier().verify(chain=chain)

    def test_l2e_different_deployer(self, harness, ll_backend):
        """Reload must preserve deployer."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(
            harness, ll_backend, chain, "agent-0",
            host="D_B",  # different deployer
            lifecycle=make_lifecycle_reload(1),
        )
        with pytest.raises(LifecycleVerifyError, match="L2e"):
            LifecycleVerifier().verify(chain=chain)

    def test_l2f_nonzero_branch_index(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _extend(
            harness, ll_backend, chain, "agent-0",
            lifecycle=make_lifecycle_reload(1),
        )
        chain = _mutate_lifecycle(chain, 1, branch_index=1)
        with pytest.raises(LifecycleVerifyError, match="L2f"):
            LifecycleVerifier().verify(chain=chain)


# ============================================================================
# L3: branch event
# ============================================================================

class TestBranchEvent:
    """L3 invariants for lifecycle.event == 'branch'."""

    def test_valid_branch(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _extend(
            harness, ll_backend, chain, "agent-branch-1",
            lifecycle=make_lifecycle_branch(1),
        )
        LifecycleVerifier().verify(chain=chain)

    def test_l3a_zero_branch_index(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "agent-branch-1")
        chain = _mutate_lifecycle(chain, 1, event="branch", branch_index=0)
        with pytest.raises(LifecycleVerifyError, match="L3a"):
            LifecycleVerifier().verify(chain=chain)

    def test_l3b_branch_at_root(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _mutate_lifecycle(chain, 0, event="branch", branch_index=1)
        with pytest.raises(LifecycleVerifyError, match="L3b"):
            LifecycleVerifier().verify(chain=chain)

    def test_l3c_same_instance_id(self, harness, ll_backend):
        """Branch must have a *new* instance_id (different from parent)."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(
            harness, ll_backend, chain, "agent-0",  # same as parent!
            lifecycle=make_lifecycle_branch(1),
        )
        with pytest.raises(LifecycleVerifyError, match="L3c"):
            LifecycleVerifier().verify(chain=chain)

    def test_l3d_different_system_id(self, harness, ll_backend):
        """Branch must preserve system_id."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(
            harness, ll_backend, chain, "agent-branch-1",
            system_id="different-model",
            lifecycle=make_lifecycle_branch(1),
        )
        with pytest.raises(LifecycleVerifyError, match="L3d"):
            LifecycleVerifier().verify(chain=chain)


# ============================================================================
# L4: composite event
# ============================================================================

class TestCompositeEvent:
    """L4 invariants for lifecycle.event == 'composite'."""

    def test_valid_composite(self, harness, ll_backend):
        """Composite with 2 constituents that exist in the chain."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "agent-1")
        # Composite merging agent-0 and agent-1
        chain = _extend(
            harness, ll_backend, chain, "agent-merged",
            lifecycle=make_lifecycle_composite(("agent-0", "agent-1")),
        )
        LifecycleVerifier().verify(chain=chain)

    def test_valid_composite_with_auxiliary(self, harness, ll_backend):
        """Constituents found in auxiliary chains."""
        chain_main = _build_root(harness, ll_backend)
        chain_aux = _build_root(
            harness, ll_backend,
            lifecycle=make_lifecycle_new(),
        )
        # Mutate aux root to have a distinct instance_id
        chain_aux = _mutate_payload(chain_aux, 0, instance_id="aux-agent")

        chain_main = _extend(
            harness, ll_backend, chain_main, "merged",
            lifecycle=make_lifecycle_composite(("agent-0", "aux-agent")),
        )
        LifecycleVerifier().verify(
            chain=chain_main, auxiliary_chains=[chain_aux]
        )

    def test_l4a_one_constituent(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "merged")
        chain = _mutate_lifecycle(
            chain, 1, event="composite", constituents=("agent-0",)
        )
        with pytest.raises(LifecycleVerifyError, match="L4a"):
            LifecycleVerifier().verify(chain=chain)

    def test_l4b_unknown_constituent(self, harness, ll_backend):
        """A constituent not found anywhere in the chain."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(
            harness, ll_backend, chain, "merged",
            lifecycle=make_lifecycle_composite(("agent-0", "phantom-agent")),
        )
        with pytest.raises(LifecycleVerifyError, match="L4b.*phantom-agent"):
            LifecycleVerifier().verify(chain=chain)

    def test_l4a_zero_constituents(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "merged")
        chain = _mutate_lifecycle(
            chain, 1, event="composite", constituents=()
        )
        with pytest.raises(LifecycleVerifyError, match="L4a"):
            LifecycleVerifier().verify(chain=chain)


# ============================================================================
# L5: fine_tune event
# ============================================================================

class TestFineTuneEvent:
    """L5 invariants for lifecycle.event == 'fine_tune'."""

    def test_valid_fine_tune(self, harness, ll_backend):
        """Fine-tuned from claude-3-sonnet (root) to my-fine-tuned-sonnet."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(
            harness, ll_backend, chain, "ft-agent",
            system_id="my-fine-tuned-sonnet",
            lifecycle=make_lifecycle_fine_tune("claude-3-sonnet"),
        )
        LifecycleVerifier().verify(chain=chain)

    def test_l5a_empty_base(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "ft-agent",
                        system_id="my-ft")
        chain = _mutate_lifecycle(
            chain, 1, event="fine_tune", base_system_id=""
        )
        with pytest.raises(LifecycleVerifyError, match="L5a"):
            LifecycleVerifier().verify(chain=chain)

    def test_l5a_none_base(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "ft-agent",
                        system_id="my-ft")
        chain = _mutate_lifecycle(
            chain, 1, event="fine_tune", base_system_id=None
        )
        with pytest.raises(LifecycleVerifyError, match="L5a"):
            LifecycleVerifier().verify(chain=chain)

    def test_l5b_same_system_id(self, harness, ll_backend):
        """base_system_id must differ from system_id."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(
            harness, ll_backend, chain, "ft-agent",
            system_id="claude-3-sonnet",  # same as base
        )
        chain = _mutate_lifecycle(
            chain, 1, event="fine_tune", base_system_id="claude-3-sonnet"
        )
        with pytest.raises(LifecycleVerifyError, match="L5b"):
            LifecycleVerifier().verify(chain=chain)

    def test_l5c_base_not_in_chain(self, harness, ll_backend):
        """base_system_id must appear somewhere in the chain."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(
            harness, ll_backend, chain, "ft-agent",
            system_id="my-ft",
        )
        chain = _mutate_lifecycle(
            chain, 1, event="fine_tune", base_system_id="unknown-base-model"
        )
        with pytest.raises(LifecycleVerifyError, match="L5c"):
            LifecycleVerifier().verify(chain=chain)


# ============================================================================
# L4 incremental: self-reference and forward-reference (red-team findings)
# ============================================================================

class TestCompositeIncrementalBuild:
    """L4 bugs found by Opus 4.6 red-team: self-ref and forward-ref."""

    def test_l4_selfref_blocked(self, harness, ll_backend):
        """Composite cannot list itself as a constituent."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "agent-1")
        # "merged" tries to name itself as a constituent
        chain = _extend(
            harness, ll_backend, chain, "merged",
            lifecycle=make_lifecycle_composite(("agent-0", "merged")),
        )
        with pytest.raises(LifecycleVerifyError, match="L4b.*merged"):
            LifecycleVerifier().verify(chain=chain)

    def test_l4_forward_ref_blocked(self, harness, ll_backend):
        """Composite cannot reference a node that appears later in the chain."""
        chain = _build_root(harness, ll_backend)
        # Node at index 1 references "future-agent" which will be at index 2
        chain = _extend(
            harness, ll_backend, chain, "merged",
            lifecycle=make_lifecycle_composite(("agent-0", "future-agent")),
        )
        # Even if we add future-agent afterwards, the composite at idx=1
        # should have failed already
        with pytest.raises(LifecycleVerifyError, match="L4b.*future-agent"):
            LifecycleVerifier().verify(chain=chain)

    def test_l4_backward_ref_allowed(self, harness, ll_backend):
        """Composite CAN reference nodes that appear earlier in the chain."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "agent-1")
        chain = _extend(
            harness, ll_backend, chain, "merged",
            lifecycle=make_lifecycle_composite(("agent-0", "agent-1")),
        )
        LifecycleVerifier().verify(chain=chain)  # should pass


# ============================================================================
# L5t: timestamp monotonicity (red-team finding)
# ============================================================================

class TestTimestampMonotonicity:
    """L5t bug found by red-team: child.created_at < parent.created_at."""

    def test_l5t_child_before_parent_blocked(self, harness, ll_backend):
        """A child cannot have created_at earlier than its parent."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "agent-1")
        # Mutate child's created_at to be before parent's
        parent_ts = chain[0].signed.payload.created_at
        early_ts = "2020-01-01T00:00:00Z"  # way before parent
        chain = _mutate_payload(chain, 1, created_at=early_ts)
        with pytest.raises(LifecycleVerifyError, match="L5t"):
            LifecycleVerifier().verify(chain=chain)

    def test_l5t_equal_timestamp_allowed(self, harness, ll_backend):
        """Same timestamp is fine (not strictly before)."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "agent-1")
        parent_ts = chain[0].signed.payload.created_at
        chain = _mutate_payload(chain, 1, created_at=parent_ts)
        LifecycleVerifier().verify(chain=chain)  # should pass

    def test_l5t_later_timestamp_allowed(self, harness, ll_backend):
        """Child after parent is the normal case."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "agent-1")
        LifecycleVerifier().verify(chain=chain)  # should pass


# ============================================================================
# L6a: nonce uniqueness (red-team finding)
# ============================================================================

class TestNonceUniqueness:
    """L6a bug found by red-team: duplicate nonces within a chain."""

    def test_l6a_duplicate_nonce_blocked(self, harness, ll_backend):
        """Two nodes cannot share the same nonce."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "agent-1")
        # Copy root's nonce to child
        root_nonce = chain[0].signed.payload.nonce
        chain = _mutate_payload(chain, 1, nonce=root_nonce)
        with pytest.raises(LifecycleVerifyError, match="L6a"):
            LifecycleVerifier().verify(chain=chain)

    def test_l6a_unique_nonces_pass(self, harness, ll_backend):
        """Normal chains have unique nonces (from crypto.new_nonce_hex)."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "agent-1")
        chain = _extend(harness, ll_backend, chain, "agent-2")
        LifecycleVerifier().verify(chain=chain)  # should pass


# ============================================================================
# Cross-node and composability
# ============================================================================

class TestCrossDesignComposability:
    """verify_full() should compose chain-shape + lifecycle checks."""

    @pytest.mark.parametrize("BackendCls", [LinkedListBackend, MerkleBackend])
    def test_verify_full_valid(self, harness, BackendCls):
        backend = BackendCls()
        chain = _build_root(harness, backend)
        chain = _extend(harness, backend, chain, "agent-1")
        verify_full(chain, backend, harness.registry)

    def test_verify_full_valid_signed_dag(self, harness):
        backend = SignedDAGBackend()
        chain = _build_root(harness, backend)
        chain = _extend(harness, backend, chain, "agent-1",
                        lifecycle=make_lifecycle_new())
        verify_full(chain, backend, harness.registry)

    def test_verify_full_lifecycle_error(self, harness, ll_backend):
        """Chain-shape is valid, but lifecycle is violated."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(
            harness, ll_backend, chain, "agent-0",
            lifecycle=make_lifecycle_reload(1),
        )
        # Mutate deployer in lifecycle record only (chain shape still valid)
        chain = _mutate_lifecycle(chain, 1, event="reload", reload_count=0)
        with pytest.raises(LifecycleVerifyError, match="L2a"):
            verify_full(chain, ll_backend, harness.registry)


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:

    def test_empty_chain(self):
        with pytest.raises(LifecycleVerifyError, match="L0"):
            LifecycleVerifier().verify(chain=[])

    def test_single_root(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        LifecycleVerifier().verify(chain=chain)  # no error

    def test_mixed_events_depth_3(self, harness, ll_backend):
        """root(new) → agent-1(new) → agent-1(reload): valid chain."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "agent-1",
                        lifecycle=make_lifecycle_new())
        chain = _extend(harness, ll_backend, chain, "agent-1",
                        lifecycle=make_lifecycle_reload(1))
        LifecycleVerifier().verify(chain=chain)

    def test_branch_then_reload(self, harness, ll_backend):
        """root → branch(agent-b) → reload(agent-b): valid chain."""
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "agent-b",
                        lifecycle=make_lifecycle_branch(1))
        chain = _extend(harness, ll_backend, chain, "agent-b",
                        lifecycle=make_lifecycle_reload(1))
        LifecycleVerifier().verify(chain=chain)


# ============================================================================
# Error message quality (red-team LLM reads these)
# ============================================================================

class TestErrorMessages:

    def test_error_contains_invariant_id(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _mutate_lifecycle(chain, 0, reload_count=5)
        try:
            LifecycleVerifier().verify(chain=chain)
            pytest.fail("expected LifecycleVerifyError")
        except LifecycleVerifyError as e:
            assert e.invariant_id == "L1c"
            assert e.node_index == 0
            assert "L1c" in str(e)

    def test_error_contains_field_values(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _extend(
            harness, ll_backend, chain, "agent-WRONG",
            lifecycle=make_lifecycle_reload(1),
        )
        try:
            LifecycleVerifier().verify(chain=chain)
            pytest.fail("expected LifecycleVerifyError")
        except LifecycleVerifyError as e:
            assert "agent-0" in str(e)      # expected value
            assert "agent-WRONG" in str(e)  # actual value

    def test_error_contains_hop_index(self, harness, ll_backend):
        chain = _build_root(harness, ll_backend)
        chain = _extend(harness, ll_backend, chain, "a1")
        chain = _extend(harness, ll_backend, chain, "a2")
        # Violate L1c at hop 2
        chain = _mutate_lifecycle(chain, 2, reload_count=99)
        try:
            LifecycleVerifier().verify(chain=chain)
            pytest.fail("expected LifecycleVerifyError")
        except LifecycleVerifyError as e:
            assert e.node_index == 2
            assert "hop 2" in str(e)

"""Week 2 exit criterion: 15-cell ASR matrix with Wilson CIs.

Runs a smaller trial count (100 per cell instead of 1,000) for CI speed,
but checks the same structural properties:
  - All 15 cells produce a CellResult
  - ASR is 0 for all cells (fixed attacks without deployer key are blocked)
  - Wilson CI upper bound < 0.05 (at n=100, Wilson upper for k=0 is ~0.036)
  - B5 anomaly detection produces the expected metrics

The full 1,000-trial matrix is run by `scripts/paper2_asr_matrix.py`.
"""

from __future__ import annotations

import pytest

from src.attacks.ancestry import (
    attack_b1_orphan_injection,
    attack_b2_chain_break_splice,
    attack_b3_cross_chain_merge,
    attack_b4_ancestor_revision,
    attack_b5_sibling_fanout,
    evaluate_fanout_detection,
    FanoutDetector,
)
from src.analysis.stats import wilson_ci, CellResult
from src.defenses.chain_designs import (
    ChainNode,
    ChainVerifyError,
    LinkedListBackend,
    MerkleBackend,
    SignedDAGBackend,
)
from src.runtime.multi_deployer import ChainBuilder, MultiDeployerHarness


# ---------- Fixtures ----------

TRIAL_COUNT = 100  # reduced for CI speed; full run uses 1,000

DESIGNS = {
    "linked_list": LinkedListBackend,
    "merkle": MerkleBackend,
    "signed_dag": SignedDAGBackend,
}


@pytest.fixture
def harness() -> MultiDeployerHarness:
    return MultiDeployerHarness.with_deployers(["D_A", "D_B"])


def _build_chain(harness, backend_cls, depth=5):
    backend = backend_cls()
    is_dag = backend.design_name == "signed_dag"
    b = ChainBuilder(harness, backend)
    chain = b.start_root("D_A", "agent-0", "claude-3-sonnet", b"root")
    hosts = ["D_A", "D_B"]
    for i in range(1, depth + 1):
        host = hosts[i % 2]
        parent_dep = chain[-1].signed.payload.deployer
        co = ()
        if is_dag:
            co = (parent_dep,) if parent_dep == host else (parent_dep, host)
        chain = b.extend(
            chain, host=host, instance_id=f"agent-{i}",
            system_id="claude-3-sonnet", output=f"out-{i}".encode(),
            co_signer_names=co,
        )
    return chain


# ---------- B1: Orphan injection ----------

class TestB1OrphanInjection:
    @pytest.mark.parametrize("design_name,cls", list(DESIGNS.items()))
    @pytest.mark.parametrize("variant", ["fabricated_parent", "missing_parent", "ghost_id"])
    def test_b1_detected(self, harness, design_name, cls, variant):
        chain = _build_chain(harness, cls)
        backend = cls()
        mutated = attack_b1_orphan_injection(chain, variant=variant)
        with pytest.raises(ChainVerifyError):
            backend.verify_chain(chain=mutated, deployers=harness.registry)


# ---------- B2: Chain-break splice ----------

class TestB2Splice:
    @pytest.mark.parametrize("design_name,cls", list(DESIGNS.items()))
    @pytest.mark.parametrize("variant", ["root_adjacent", "middle", "leaf_adjacent"])
    def test_b2_detected(self, harness, design_name, cls, variant):
        chain = _build_chain(harness, cls)
        backend = cls()
        mutated = attack_b2_chain_break_splice(chain, variant=variant)
        with pytest.raises(ChainVerifyError):
            backend.verify_chain(chain=mutated, deployers=harness.registry)


# ---------- B3: Cross-chain merge ----------

class TestB3CrossChain:
    @pytest.mark.parametrize("design_name,cls", list(DESIGNS.items()))
    @pytest.mark.parametrize("variant", ["graft_at_leaf", "graft_at_mid", "append_alien"])
    def test_b3_detected(self, harness, design_name, cls, variant):
        chain_y = _build_chain(harness, cls)
        chain_x = _build_chain(harness, cls)  # different keys from same harness
        backend = cls()
        mutated = attack_b3_cross_chain_merge(chain_y, chain_x, variant=variant)
        with pytest.raises(ChainVerifyError):
            backend.verify_chain(chain=mutated, deployers=harness.registry)


# ---------- B4: Ancestor revision ----------

class TestB4AncestorRevision:
    @pytest.mark.parametrize("design_name,cls", list(DESIGNS.items()))
    @pytest.mark.parametrize("field", ["output_hash", "certifications", "deployer", "parent_id"])
    def test_b4_detected(self, harness, design_name, cls, field):
        chain = _build_chain(harness, cls)
        backend = cls()
        mutated = attack_b4_ancestor_revision(chain, revise_index=2, field=field)
        with pytest.raises(ChainVerifyError):
            backend.verify_chain(chain=mutated, deployers=harness.registry)


# ---------- B5: Sibling fanout ----------

class TestB5SiblingFanout:
    def test_b5_fanout_detector(self, harness):
        det = FanoutDetector(threshold=10)
        for _ in range(11):
            det.record_child("parent-1")
        assert det.is_flagged("parent-1")
        assert not det.is_flagged("parent-2")

    def test_b5_below_threshold_not_detected(self, harness):
        chain = _build_chain(harness, LinkedListBackend)
        benign, malicious = attack_b5_sibling_fanout(
            chain, n_benign_siblings=5, n_malicious_siblings=1,
        )
        result = evaluate_fanout_detection(benign, malicious, threshold=10)
        # 6 children < threshold 10 → NOT flagged → attack "succeeds"
        assert not result["parent_flagged"]

    def test_b5_above_threshold_detected(self, harness):
        chain = _build_chain(harness, LinkedListBackend)
        benign, malicious = attack_b5_sibling_fanout(
            chain, n_benign_siblings=50, n_malicious_siblings=1,
        )
        result = evaluate_fanout_detection(benign, malicious, threshold=50)
        # 51 children > threshold 50 → flagged
        assert result["parent_flagged"]


# ---------- Wilson CI sanity ----------

class TestWilsonCI:
    def test_wilson_zero_successes(self):
        ci = wilson_ci(0, 1000)
        assert ci.point == 0.0
        assert ci.low == 0.0
        assert ci.high < 0.005  # well under 0.014

    def test_wilson_all_successes(self):
        ci = wilson_ci(1000, 1000)
        assert ci.point == 1.0
        assert ci.high == 1.0
        assert ci.low > 0.99

    def test_wilson_half(self):
        ci = wilson_ci(500, 1000)
        assert 0.45 < ci.low < 0.5
        assert 0.5 < ci.high < 0.55


# ---------- Full matrix integration (reduced trial count) ----------

class TestASRMatrixIntegration:
    """Run all 15 cells at reduced trial count and verify H1."""

    def test_all_15_cells_pass_h1(self, harness):
        from scripts.paper2_asr_matrix import run_cell, DESIGNS as D

        results = []
        for design_name in D:
            for attack in ["B1", "B2", "B3", "B4", "B5"]:
                r = run_cell(design_name, attack, TRIAL_COUNT, harness)
                results.append(r)

        assert len(results) == 15
        for r in results:
            assert r.asr == 0.0, f"{r.design}×{r.attack} had ASR={r.asr}"
            assert r.ci_high < 0.05, f"{r.design}×{r.attack} CI upper too high"

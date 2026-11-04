"""Integration coverage for E10's process-separated Signed-DAG path."""

from __future__ import annotations

import os
from pathlib import Path

from experiments.e10_multiprocess_cross_deployer.run import run_benchmark


def test_process_separated_cross_deployer_path_and_fail_closed_faults(
    tmp_path: Path,
) -> None:
    """Signer keys stay in spawned workers and every injected fault fails closed."""
    result = run_benchmark(n_trials=3, warmup=1, state_dir=tmp_path / "e10-state")

    assert result["valid_trials"] == 3
    assert result["aborted_trials"] == 0
    assert len(result["trial_records"]) == 3
    assert all(record["verified"] for record in result["trial_records"])

    initial_signers = result["signers_initial"]
    assert {signer["deployer"] for signer in initial_signers} == {
        "platform_A",
        "platform_B",
    }
    assert all(signer["pid"] != os.getpid() for signer in initial_signers)

    faults = result["faults"]
    assert faults["durable_replay"]["fresh_verifier_rejected_second_presentation"]
    assert faults["unregistered_parent"]["rejected"]
    assert faults["revoked_child_key"]["rejected"]
    assert faults["parent_refusal"]["issuance_aborted"]
    assert faults["child_key_rotation"]["old_receipt_rejected"]
    assert faults["child_key_rotation"]["new_receipt_accepted"]

    final_signers = result["signers_after_faults"]
    assert initial_signers[1]["key_id"] != final_signers[1]["key_id"]

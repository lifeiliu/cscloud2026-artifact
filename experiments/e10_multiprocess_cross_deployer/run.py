"""E10: process-separated cross-deployer Signed-DAG validation.

This experiment moves the signer boundary out of the in-process
``MultiDeployerHarness`` used by the unit tests.  It starts two independent
``spawn`` workers, one per deployer.  Each worker generates and retains its own
private key, and only returns its public key, signed receipts, or edge
signatures.  The verifier runs in the coordinator process with a public-only
registry and a file-backed SQLite nonce store.

The transport is deliberately local IPC (``multiprocessing.Pipe``), not a
network, container, KMS, or production authorization service.  The experiment
therefore measures local process-boundary protocol overhead and failure modes,
not cloud latency.

Run from the repository root:

    python -m experiments.e10_multiprocess_cross_deployer.run --trials 1000

The result JSON retains the raw per-trial timings, run configuration,
environment manifest, verifier predicates, and explicit fault outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from src.core.agent_id_v2 import (
    BoundIDPayloadV2,
    ChainProof,
    SignedBoundIDV2,
    make_chain_proof_signed_dag,
    make_lifecycle_new,
)
from src.core.crypto import generate_keypair, sign, verify
from src.defenses.chain_designs import (
    ChainNode,
    ChainVerifyError,
    DeployerRegistry,
    SignedDAGBackend,
)
from src.defenses.chain_designs.base import edge_signing_input
from src.defenses.complete_chain_verifier import verify_complete_chain
from src.runtime.multi_deployer import DeployerV2

ROOT_OUTPUT = b"planner: prepare a verifiable cross-deployer delegation"
CHILD_OUTPUT = b"writer: emit the attested report"
SYSTEM_ID = "e10-process-worker"
ROOT_DEPLOYER = "platform_A"
CHILD_DEPLOYER = "platform_B"
VERIFIER_PREDICATES = (
    "trusted registry lookup",
    "public-key-derived key_id match",
    "local key revocation",
    "Ed25519 primary signature",
    "validity interval",
    "released-output SHA-256",
    "within-chain nonce uniqueness",
    "durable SQLite replay lookup",
    "Signed-DAG parent reference and depth",
    "Signed-DAG parent and child edge co-signatures",
)


class SignerRPCError(RuntimeError):
    """A signer service rejected an issuance or co-signing request."""


class SQLiteNonceStore:
    """Minimal durable nonce store for sequential cross-verifier replay tests.

    The experiment creates a new instance for the second presentation to prove
    that replay state survives verifier-object recreation.  It is intentionally
    not represented as a distributed, concurrent production replay service.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS consumed_nonces ("
                "nonce TEXT PRIMARY KEY NOT NULL, consumed_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def seen(self, nonce: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM consumed_nonces WHERE nonce = ?", (nonce,)
            ).fetchone()
        return row is not None

    def mark(self, nonce: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO consumed_nonces (nonce, consumed_at) VALUES (?, ?)",
                (nonce, datetime.now(UTC).isoformat()),
            )


@dataclass(frozen=True)
class SignerInfo:
    """Public metadata returned by a signer process at bootstrap."""

    deployer: str
    pid: int
    key_id: str
    public_key_hex: str


def _error_response(error: Exception) -> dict[str, object]:
    return {"ok": False, "error_type": type(error).__name__, "error": str(error)}


def _signer_worker(deployer_name: str, connection: Connection) -> None:
    """Serve issue/co-sign requests while keeping the private key local."""

    keypair = generate_keypair()
    deployer = DeployerV2(deployer_name, keypair)
    backend = SignedDAGBackend()
    connection.send(
        {
            "ok": True,
            "op": "ready",
            "deployer": deployer_name,
            "pid": os.getpid(),
            "key_id": keypair.key_id,
            "public_key_hex": keypair.public_bytes.hex(),
        }
    )

    try:
        while True:
            request = connection.recv()
            op = request.get("op")
            if op == "shutdown":
                connection.send({"ok": True, "op": "shutdown"})
                return
            try:
                if op == "issue_root":
                    signed = deployer.issue(
                        instance_id=str(request["instance_id"]),
                        system_id=str(request["system_id"]),
                        output_bytes=bytes(request["output"]),
                        binding_scope="per_session",
                        chain_proof=backend.build_proof_for_root(),
                        lifecycle=make_lifecycle_new(),
                        ancestry_depth=0,
                    )
                    connection.send({"ok": True, "signed": signed})
                elif op == "prepare_cross_deployer_child":
                    parent = request["parent"]
                    if not isinstance(parent, SignedBoundIDV2):
                        raise TypeError("parent must be SignedBoundIDV2")
                    provisional = deployer.build_payload(
                        instance_id=str(request["instance_id"]),
                        system_id=str(request["system_id"]),
                        output_bytes=bytes(request["output"]),
                        binding_scope="per_session",
                        chain_proof=ChainProof(
                            design="signed_dag",
                            co_signers=(parent.payload.deployer, deployer_name),
                        ),
                        lifecycle=make_lifecycle_new(),
                        parent_id=parent.payload.instance_id,
                        ancestry_depth=parent.payload.ancestry_depth + 1,
                    )
                    connection.send({"ok": True, "provisional": provisional})
                elif op == "co_sign_edge":
                    parent = request["parent"]
                    child = request["child"]
                    if not isinstance(parent, SignedBoundIDV2):
                        raise TypeError("parent must be SignedBoundIDV2")
                    if not isinstance(child, BoundIDPayloadV2):
                        raise TypeError("child must be BoundIDPayloadV2")
                    if not bool(request.get("approve", True)):
                        raise SignerRPCError("parent policy denied the co-sign request")
                    if parent.payload.deployer != deployer_name:
                        raise SignerRPCError("parent receipt is not owned by this signer")
                    if parent.key_id != keypair.key_id or not verify(
                        keypair.public,
                        parent.payload.to_canonical_json(),
                        parent.signature_hex,
                    ):
                        raise SignerRPCError(
                            "parent receipt does not verify under local signer key"
                        )
                    if child.parent_id != parent.payload.instance_id:
                        raise SignerRPCError("child does not reference the supplied parent")
                    if child.ancestry_depth != parent.payload.ancestry_depth + 1:
                        raise SignerRPCError("child ancestry depth does not extend the parent")
                    expected_signers = (deployer_name, child.deployer)
                    if tuple(child.chain_proof.co_signers) != expected_signers:
                        raise SignerRPCError("child co-signer policy does not match this edge")
                    connection.send(
                        {
                            "ok": True,
                            "edge_signature": sign(
                                keypair.private, edge_signing_input(parent.payload, child)
                            ),
                        }
                    )
                elif op == "finalize_cross_deployer_child":
                    parent = request["parent"]
                    provisional = request["provisional"]
                    parent_signature = str(request["parent_signature"])
                    if not isinstance(parent, SignedBoundIDV2):
                        raise TypeError("parent must be SignedBoundIDV2")
                    if not isinstance(provisional, BoundIDPayloadV2):
                        raise TypeError("provisional must be BoundIDPayloadV2")
                    if provisional.deployer != deployer_name:
                        raise SignerRPCError("child payload is not owned by this signer")
                    expected_signers = (parent.payload.deployer, deployer_name)
                    if tuple(provisional.chain_proof.co_signers) != expected_signers:
                        raise SignerRPCError("unexpected co-signer order for cross-deployer edge")
                    message = edge_signing_input(parent.payload, provisional)
                    child_signature = sign(keypair.private, message)
                    proof = make_chain_proof_signed_dag(
                        co_signers=expected_signers,
                        signatures=(parent_signature, child_signature),
                    )
                    signed = deployer.sign_payload(replace(provisional, chain_proof=proof))
                    connection.send({"ok": True, "signed": signed})
                else:
                    raise ValueError(f"unknown signer operation: {op!r}")
            except Exception as error:  # Returned explicitly; never counted as a rejection.
                connection.send(_error_response(error))
    finally:
        connection.close()


class SignerService:
    """Request/response client for one spawned signer process."""

    def __init__(self, context, deployer: str) -> None:
        self.deployer = deployer
        parent, child = context.Pipe(duplex=True)
        self._connection = parent
        self._process = context.Process(
            target=_signer_worker,
            args=(deployer, child),
            name=f"e10-signer-{deployer}",
        )
        self._process.start()
        child.close()
        ready = self._receive()
        if ready.get("op") != "ready":
            raise SignerRPCError(f"{deployer} did not complete bootstrap: {ready}")
        self.info = SignerInfo(
            deployer=str(ready["deployer"]),
            pid=int(ready["pid"]),
            key_id=str(ready["key_id"]),
            public_key_hex=str(ready["public_key_hex"]),
        )

    def _receive(self) -> dict[str, object]:
        if not self._connection.poll(20):
            raise SignerRPCError(f"{self.deployer} signer RPC timed out")
        response = self._connection.recv()
        if not isinstance(response, dict):
            raise SignerRPCError(f"{self.deployer} signer returned malformed response")
        if not response.get("ok"):
            raise SignerRPCError(
                f"{self.deployer} signer rejected request: "
                f"{response.get('error_type')}: {response.get('error')}"
            )
        return response

    def call(self, op: str, **payload: object) -> dict[str, object]:
        self._connection.send({"op": op, **payload})
        return self._receive()

    def close(self) -> None:
        if self._process.is_alive():
            with suppress(EOFError, OSError, SignerRPCError):
                self.call("shutdown")
            self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=10)
        self._connection.close()


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile of no measurements")
    index = (len(sorted_values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (index - lower)


def _stats(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": round(sum(values) / len(values), 6),
        "p50": round(_percentile(ordered, 0.50), 6),
        "p95": round(_percentile(ordered, 0.95), 6),
        "p99": round(_percentile(ordered, 0.99), 6),
    }


def _rejection_reason(
    *,
    chain: Sequence[ChainNode],
    outputs: Sequence[bytes],
    registry: DeployerRegistry,
    nonces: SQLiteNonceStore | None = None,
) -> tuple[bool, str]:
    """Return whether a verifier predicate rejects a chain and its reason."""
    try:
        verify_complete_chain(
            chain=chain,
            outputs=outputs,
            backend=SignedDAGBackend(),
            deployers=registry,
            nonces=nonces,
        )
    except ChainVerifyError as error:
        return True, str(error)
    return False, "accepted unexpectedly"


class MultiprocessCrossDeployerExperiment:
    """Own two signer workers and a public-only verifier registry."""

    def __init__(self) -> None:
        self.context = get_context("spawn")
        self.parent = SignerService(self.context, ROOT_DEPLOYER)
        self.child = SignerService(self.context, CHILD_DEPLOYER)

    def close(self) -> None:
        self.child.close()
        self.parent.close()

    def registry(self) -> DeployerRegistry:
        registry = DeployerRegistry()
        for service in (self.parent, self.child):
            registry.register_public(
                service.info.deployer,
                Ed25519PublicKey.from_public_bytes(bytes.fromhex(service.info.public_key_hex)),
            )
        return registry

    def issue_chain(self, trial_id: str) -> tuple[list[ChainNode], list[bytes], float]:
        """Issue root and cross-deployer child through the explicit co-sign RPC."""
        root_response = self.parent.call(
            "issue_root",
            instance_id=f"planner-{trial_id}",
            system_id=SYSTEM_ID,
            output=ROOT_OUTPUT,
        )
        root = root_response["signed"]
        if not isinstance(root, SignedBoundIDV2):
            raise TypeError("root signer did not return SignedBoundIDV2")

        start = time.perf_counter_ns()
        provisional_response = self.child.call(
            "prepare_cross_deployer_child",
            parent=root,
            instance_id=f"writer-{trial_id}",
            system_id=SYSTEM_ID,
            output=CHILD_OUTPUT,
        )
        provisional = provisional_response["provisional"]
        if not isinstance(provisional, BoundIDPayloadV2):
            raise TypeError("child signer did not return a provisional payload")
        parent_signature_response = self.parent.call(
            "co_sign_edge", parent=root, child=provisional, approve=True
        )
        child_response = self.child.call(
            "finalize_cross_deployer_child",
            parent=root,
            provisional=provisional,
            parent_signature=parent_signature_response["edge_signature"],
        )
        issue_latency_ms = (time.perf_counter_ns() - start) / 1_000_000
        child = child_response["signed"]
        if not isinstance(child, SignedBoundIDV2):
            raise TypeError("child signer did not return SignedBoundIDV2")
        return [ChainNode(root), ChainNode(child)], [ROOT_OUTPUT, CHILD_OUTPUT], issue_latency_ms

    def parent_refusal(self, trial_id: str) -> str:
        """Request the parent co-signature with denial enabled; issuance must stop."""
        root_response = self.parent.call(
            "issue_root",
            instance_id=f"denied-planner-{trial_id}",
            system_id=SYSTEM_ID,
            output=ROOT_OUTPUT,
        )
        root = root_response["signed"]
        if not isinstance(root, SignedBoundIDV2):
            raise TypeError("root signer did not return SignedBoundIDV2")
        provisional_response = self.child.call(
            "prepare_cross_deployer_child",
            parent=root,
            instance_id=f"denied-writer-{trial_id}",
            system_id=SYSTEM_ID,
            output=CHILD_OUTPUT,
        )
        provisional = provisional_response["provisional"]
        if not isinstance(provisional, BoundIDPayloadV2):
            raise TypeError("child signer did not return a provisional payload")
        try:
            self.parent.call("co_sign_edge", parent=root, child=provisional, approve=False)
        except SignerRPCError as error:
            return str(error)
        raise AssertionError("parent refusal unexpectedly returned a co-signature")

    def rotate_child_signer(self) -> tuple[SignerInfo, SignerInfo]:
        """Replace only the child signer and retain no handle to its old key."""
        old_info = self.child.info
        replacement = SignerService(self.context, CHILD_DEPLOYER)
        old_child = self.child
        self.child = replacement
        old_child.close()
        return old_info, replacement.info


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _environment_manifest() -> dict[str, str]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cryptography": _package_version("cryptography"),
        "source_sha256": _source_sha256(),
        "git_revision": _git_revision(),
    }


def _run_faults(
    experiment: MultiprocessCrossDeployerExperiment, state_dir: Path
) -> dict[str, dict[str, object]]:
    """Exercise replay, registry, revocation, refusal, and rotation failures."""
    faults: dict[str, dict[str, object]] = {}
    registry = experiment.registry()

    replay_chain, replay_outputs, _ = experiment.issue_chain("fault-replay")
    replay_path = state_dir / "replay.sqlite"
    first_store = SQLiteNonceStore(replay_path)
    verify_complete_chain(
        chain=replay_chain,
        outputs=replay_outputs,
        backend=SignedDAGBackend(),
        deployers=registry,
        nonces=first_store,
    )
    second_store = SQLiteNonceStore(replay_path)
    rejected, reason = _rejection_reason(
        chain=replay_chain,
        outputs=replay_outputs,
        registry=registry,
        nonces=second_store,
    )
    faults["durable_replay"] = {
        "first_presentation_accepted": True,
        "fresh_verifier_rejected_second_presentation": rejected,
        "reason": reason,
        "nonce_store": "SQLite file reopened by a new verifier-store instance",
    }

    registry_chain, registry_outputs, _ = experiment.issue_chain("fault-registry")
    missing_registry = DeployerRegistry()
    missing_registry.register_public(
        CHILD_DEPLOYER,
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(experiment.child.info.public_key_hex)),
    )
    rejected, reason = _rejection_reason(
        chain=registry_chain,
        outputs=registry_outputs,
        registry=missing_registry,
    )
    faults["unregistered_parent"] = {"rejected": rejected, "reason": reason}

    revoked_chain, revoked_outputs, _ = experiment.issue_chain("fault-revocation")
    revoked_registry = experiment.registry()
    revoked_registry.revoke_key(revoked_chain[-1].signed.key_id)
    rejected, reason = _rejection_reason(
        chain=revoked_chain,
        outputs=revoked_outputs,
        registry=revoked_registry,
    )
    faults["revoked_child_key"] = {"rejected": rejected, "reason": reason}

    faults["parent_refusal"] = {
        "issuance_aborted": True,
        "reason": experiment.parent_refusal("fault-refusal"),
    }

    old_chain, old_outputs, _ = experiment.issue_chain("fault-rotation-old")
    old_info, new_info = experiment.rotate_child_signer()
    rotated_registry = experiment.registry()
    old_rejected, old_reason = _rejection_reason(
        chain=old_chain,
        outputs=old_outputs,
        registry=rotated_registry,
    )
    new_chain, new_outputs, _ = experiment.issue_chain("fault-rotation-new")
    rotation_store = SQLiteNonceStore(state_dir / "rotation.sqlite")
    try:
        verify_complete_chain(
            chain=new_chain,
            outputs=new_outputs,
            backend=SignedDAGBackend(),
            deployers=rotated_registry,
            nonces=rotation_store,
        )
        new_accepted = True
        new_reason = "accepted"
    except ChainVerifyError as error:
        new_accepted = False
        new_reason = str(error)
    faults["child_key_rotation"] = {
        "old_key_id": old_info.key_id,
        "new_key_id": new_info.key_id,
        "old_receipt_rejected": old_rejected,
        "old_receipt_reason": old_reason,
        "new_receipt_accepted": new_accepted,
        "new_receipt_reason": new_reason,
    }
    return faults


def run_benchmark(
    n_trials: int = 1000,
    warmup: int = 100,
    *,
    state_dir: Path | None = None,
) -> dict[str, object]:
    """Run valid cross-deployer paths and deterministic fault cases.

    Any unexpected signer or verifier exception aborts the run.  A failure is
    recorded as blocked only when the expected component returns an explicit
    fail-closed reason.
    """
    if n_trials <= 0 or warmup < 0:
        raise ValueError("n_trials must be positive and warmup must be non-negative")

    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if state_dir is None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="e10-multiprocess-")
        state_dir = Path(temporary_directory.name)
    state_dir.mkdir(parents=True, exist_ok=True)

    experiment = MultiprocessCrossDeployerExperiment()
    try:
        registry = experiment.registry()
        initial_signers = [asdict(experiment.parent.info), asdict(experiment.child.info)]
        nonce_store = SQLiteNonceStore(state_dir / "benchmark.sqlite")
        for warmup_index in range(warmup):
            chain, outputs, _ = experiment.issue_chain(f"warmup-{warmup_index}")
            verify_complete_chain(
                chain=chain,
                outputs=outputs,
                backend=SignedDAGBackend(),
                deployers=registry,
                nonces=nonce_store,
            )

        issuance_latencies: list[float] = []
        verification_latencies: list[float] = []
        trial_records: list[dict[str, object]] = []
        for trial_index in range(n_trials):
            chain, outputs, issue_latency_ms = experiment.issue_chain(f"trial-{trial_index}")
            verify_start = time.perf_counter_ns()
            verify_complete_chain(
                chain=chain,
                outputs=outputs,
                backend=SignedDAGBackend(),
                deployers=registry,
                nonces=nonce_store,
            )
            verify_latency_ms = (time.perf_counter_ns() - verify_start) / 1_000_000
            issuance_latencies.append(issue_latency_ms)
            verification_latencies.append(verify_latency_ms)
            trial_records.append(
                {
                    "trial": trial_index,
                    "chain_nodes": len(chain),
                    "cross_deployer_issue_ms": round(issue_latency_ms, 6),
                    "complete_verify_ms": round(verify_latency_ms, 6),
                    "verified": True,
                }
            )

        faults = _run_faults(experiment, state_dir)
        return {
            "schema_version": "e10-multiprocess-cross-deployer.v1",
            "environment": _environment_manifest(),
            "run_configuration": {
                "module": "experiments.e10_multiprocess_cross_deployer.run",
                "command": (
                    "python -m experiments.e10_multiprocess_cross_deployer.run "
                    f"--trials {n_trials} --warmup {warmup}"
                ),
                "state_dir": str(state_dir),
            },
            "execution_mode": {
                "signer_processes": 2,
                "verifier_process": "coordinator process",
                "start_method": "spawn",
                "transport": "multiprocessing.Pipe local IPC request/response",
                "private_key_boundary": "each signer creates and retains its own Ed25519 private key; only public keys, receipts, and signatures leave the worker",
                "not_measured": "network, container, KMS, model-client, and production authorization-service latency",
            },
            "signers_initial": initial_signers,
            "signers_after_faults": [
                asdict(experiment.parent.info),
                asdict(experiment.child.info),
            ],
            "design": "signed_dag",
            "scenario": "platform_A planner -> platform_B writer",
            "node_outputs": "deterministic; no LLM invocation",
            "n_trials": n_trials,
            "warmup": warmup,
            "valid_trials": n_trials,
            "aborted_trials": 0,
            "verifier_predicates": list(VERIFIER_PREDICATES),
            "cross_deployer_issue_ms": _stats(issuance_latencies),
            "complete_verify_ms": _stats(verification_latencies),
            "trial_records": trial_records,
            "faults": faults,
        }
    finally:
        experiment.close()
        if temporary_directory is not None:
            temporary_directory.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("results.json"),
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    state_dir = args.out.parent / f"{args.out.stem}_state"
    results = run_benchmark(args.trials, args.warmup, state_dir=state_dir)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    faults = results["faults"]
    assert isinstance(faults, dict)
    print(
        "E10 process-separated Signed-DAG: "
        f"{results['valid_trials']}/{results['n_trials']} valid chains verified; "
        f"cross-deployer issue mean={results['cross_deployer_issue_ms']['mean']:.4f} ms; "
        f"complete verification mean={results['complete_verify_ms']['mean']:.4f} ms."
    )
    print(f"Fault cases recorded: {', '.join(sorted(faults))}.")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

"""E11: Docker bridge-network cross-deployer Signed-DAG validation.

The coordinator runs separately from two signer containers.  Signer private
keys never leave their respective containers; the coordinator holds only the
public-key registry and a local SQLite replay store.  RPC uses an authenticated
``multiprocessing.connection`` TCP channel on a closed Docker bridge network.
It is intentionally not TLS/mTLS, a cloud workload-identity system, a KMS, or
a production authorization service.

Run from the repository root:

    docker compose -f experiments/e11_container_cross_deployer/compose.yaml \
      up --build --abort-on-container-exit --exit-code-from coordinator
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import suppress
from dataclasses import asdict
from multiprocessing.connection import Client
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from experiments.e10_multiprocess_cross_deployer.run import (
    CHILD_DEPLOYER,
    CHILD_OUTPUT,
    ROOT_DEPLOYER,
    ROOT_OUTPUT,
    SYSTEM_ID,
    SQLiteNonceStore,
    SignerInfo,
    SignerRPCError,
    VERIFIER_PREDICATES,
    _environment_manifest,
    _rejection_reason,
    _stats,
)
from src.core.agent_id_v2 import BoundIDPayloadV2, SignedBoundIDV2
from src.defenses.chain_designs import ChainNode, ChainVerifyError, DeployerRegistry, SignedDAGBackend
from src.defenses.complete_chain_verifier import verify_complete_chain


def _authkey() -> bytes:
    value = os.environ.get("E11_RPC_AUTHKEY")
    if not value:
        raise RuntimeError("E11_RPC_AUTHKEY must be set")
    return value.encode("utf-8")


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class RemoteSignerService:
    """Small request/response client for a signer in another container."""

    def __init__(self, deployer: str, host: str, port: int = 6000) -> None:
        self.deployer = deployer
        self.host = host
        self.port = port
        deadline = time.monotonic() + 30
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            try:
                self._connection = Client((host, port), family="AF_INET", authkey=_authkey())
                break
            except OSError as error:
                last_error = error
                time.sleep(0.2)
        else:
            raise SignerRPCError(f"could not connect to {deployer} at {host}:{port}: {last_error}")

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
        with suppress(EOFError, OSError, SignerRPCError):
            self.call("shutdown")
        self._connection.close()


class ContainerCrossDeployerExperiment:
    """Issue chains through signer services in two distinct containers."""

    def __init__(self, parent_host: str, child_host: str) -> None:
        self.parent = RemoteSignerService(ROOT_DEPLOYER, parent_host)
        self.child = RemoteSignerService(CHILD_DEPLOYER, child_host)

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
        root_response = self.parent.call(
            "issue_root",
            instance_id=f"planner-{trial_id}",
            system_id=SYSTEM_ID,
            output=ROOT_OUTPUT,
        )
        root = root_response["signed"]
        if not isinstance(root, SignedBoundIDV2):
            raise TypeError("parent signer did not return SignedBoundIDV2")

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
        root_response = self.parent.call(
            "issue_root",
            instance_id=f"denied-planner-{trial_id}",
            system_id=SYSTEM_ID,
            output=ROOT_OUTPUT,
        )
        root = root_response["signed"]
        if not isinstance(root, SignedBoundIDV2):
            raise TypeError("parent signer did not return SignedBoundIDV2")
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


def _run_faults(
    experiment: ContainerCrossDeployerExperiment, state_dir: Path
) -> dict[str, dict[str, object]]:
    registry = experiment.registry()
    faults: dict[str, dict[str, object]] = {}

    replay_chain, replay_outputs, _ = experiment.issue_chain("fault-replay")
    replay_path = state_dir / "replay.sqlite"
    verify_complete_chain(
        chain=replay_chain,
        outputs=replay_outputs,
        backend=SignedDAGBackend(),
        deployers=registry,
        nonces=SQLiteNonceStore(replay_path),
    )
    rejected, reason = _rejection_reason(
        chain=replay_chain,
        outputs=replay_outputs,
        registry=registry,
        nonces=SQLiteNonceStore(replay_path),
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
    return faults


def run_benchmark(
    n_trials: int,
    warmup: int,
    *,
    parent_host: str,
    child_host: str,
    state_dir: Path,
) -> dict[str, object]:
    if n_trials <= 0 or warmup < 0:
        raise ValueError("n_trials must be positive and warmup must be non-negative")
    state_dir.mkdir(parents=True, exist_ok=True)
    experiment = ContainerCrossDeployerExperiment(parent_host, child_host)
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
        environment = _environment_manifest()
        environment["e10_signer_source_sha256"] = environment.pop("source_sha256")
        environment["e11_coordinator_source_sha256"] = _source_sha256()
        environment["container_hostname"] = os.environ.get("HOSTNAME", "unavailable")
        return {
            "schema_version": "e11-container-cross-deployer.v1",
            "environment": environment,
            "run_configuration": {
                "module": "experiments.e11_container_cross_deployer.coordinator",
                "command": (
                    "docker compose -f experiments/e11_container_cross_deployer/compose.yaml "
                    f"up --build --abort-on-container-exit --exit-code-from coordinator "
                    f"(E11_TRIALS={n_trials}, E11_WARMUP={warmup})"
                ),
                "state_dir": str(state_dir),
            },
            "execution_mode": {
                "signer_containers": 2,
                "verifier_container": "coordinator",
                "network": "Docker bridge network",
                "transport": "authenticated multiprocessing.connection TCP RPC",
                "private_key_boundary": "each signer container generates and retains its own Ed25519 private key; only public keys, receipts, and signatures cross the container boundary",
                "not_measured": "TLS/mTLS, cloud workload identity, KMS, model-client, and production authorization-service latency",
            },
            "signers_initial": initial_signers,
            "design": "signed_dag",
            "scenario": "platform_A planner container -> platform_B writer container",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=int(os.environ.get("E11_TRIALS", "1000")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("E11_WARMUP", "100")))
    parser.add_argument("--parent-host", default=os.environ.get("E11_PARENT_HOST", "parent-signer"))
    parser.add_argument("--child-host", default=os.environ.get("E11_CHILD_HOST", "child-signer"))
    parser.add_argument("--out", type=Path, default=Path("/results/results.json"))
    args = parser.parse_args()
    state_dir = args.out.parent / f"{args.out.stem}_state"
    result = run_benchmark(
        args.trials,
        args.warmup,
        parent_host=args.parent_host,
        child_host=args.child_host,
        state_dir=state_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(
        "E11 container-network Signed-DAG: "
        f"{result['valid_trials']}/{result['n_trials']} valid chains verified; "
        f"cross-deployer issue mean={result['cross_deployer_issue_ms']['mean']:.4f} ms; "
        f"complete verification mean={result['complete_verify_ms']['mean']:.4f} ms."
    )
    print(f"Fault cases recorded: {', '.join(sorted(result['faults']))}.")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

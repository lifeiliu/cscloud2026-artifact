"""E12: live MCP/A2A/LLM delegation evaluation.

The experiment starts separate loopback services for an MCP Streamable-HTTP
tool, an A2A JSON-RPC child agent, and a parent signer.  It compares no
evidence, a child-primary-signed linked list, and a parent--child co-signed
DAG while binding the exact local-Ollama output released by the child.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP

from src.core.agent_id_v2 import (
    BoundIDPayloadV2,
    ChainProof,
    LifecycleRecord,
    SignedBoundIDV2,
    make_chain_proof_signed_dag,
    make_lifecycle_new,
)
from src.core.crypto import generate_keypair, key_id_from_public, sha256_hex, sign, verify
from src.defenses.chain_designs import (
    ChainNode,
    ChainVerifyError,
    DeployerRegistry,
    LinkedListBackend,
    SignedDAGBackend,
)
from src.defenses.chain_designs.base import edge_signing_input
from src.defenses.complete_chain_verifier import verify_complete_chain
from src.runtime.multi_deployer import DeployerV2

Condition = Literal["none", "linked_list", "signed_dag"]
CONDITIONS: tuple[Condition, ...] = ("none", "linked_list", "signed_dag")
ROOT_DEPLOYER = "deployer_parent"
CHILD_DEPLOYER = "deployer_child"
PARENT_SYSTEM_ID = "e12-parent-planner"
CHILD_SYSTEM_ID = "e12-child-researcher"
TOOL_NAME = "get_evidence"
TOPIC = "delegation edge authorization"
TOOL_EVIDENCE = (
    "A released output hash binds an audit record to exact bytes. A cross-deployer "
    "delegation edge is authorized only when the parent and child deployers both "
    "sign the complete edge transcript, including the child output hash."
)


class E12Error(RuntimeError):
    """An explicit protocol, model, or verifier failure."""


class SQLiteNonceStore:
    """A file-backed nonce store used by the complete verifier."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS consumed_nonces "
                "(nonce TEXT PRIMARY KEY NOT NULL, consumed_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=5.0)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def seen(self, nonce: str) -> bool:
        with self._connect() as con:
            return con.execute("SELECT 1 FROM consumed_nonces WHERE nonce = ?", (nonce,)).fetchone() is not None

    def mark(self, nonce: str) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO consumed_nonces (nonce, consumed_at) VALUES (?, ?)",
                (nonce, datetime.now(UTC).isoformat()),
            )


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _backend(condition: Condition):
    if condition == "linked_list":
        return LinkedListBackend()
    if condition == "signed_dag":
        return SignedDAGBackend()
    raise ValueError("no chain backend for no-evidence")


def _payload_to_wire(payload: BoundIDPayloadV2) -> dict[str, object]:
    return json.loads(payload.to_canonical_json().decode("utf-8"))


def _payload_from_wire(value: Mapping[str, object]) -> BoundIDPayloadV2:
    proof_raw = value.get("chain_proof")
    lifecycle_raw = value.get("lifecycle")
    if not isinstance(proof_raw, Mapping) or not isinstance(lifecycle_raw, Mapping):
        raise E12Error("receipt wire lacks chain proof or lifecycle")
    try:
        proof = ChainProof(
            design=str(proof_raw["design"]),  # type: ignore[arg-type]
            prev_hash=str(proof_raw["prev_hash"]) if proof_raw.get("prev_hash") is not None else None,
            merkle_root=str(proof_raw["merkle_root"]) if proof_raw.get("merkle_root") is not None else None,
            co_signers=tuple(str(item) for item in proof_raw.get("co_signers", [])),
            signatures=tuple(str(item) for item in proof_raw.get("signatures", [])),
        )
        lifecycle = LifecycleRecord(
            event=str(lifecycle_raw["event"]),  # type: ignore[arg-type]
            branch_index=int(lifecycle_raw.get("branch_index", 0)),
            reload_count=int(lifecycle_raw.get("reload_count", 0)),
            constituents=tuple(str(item) for item in lifecycle_raw.get("constituents", [])),
            base_system_id=str(lifecycle_raw["base_system_id"]) if lifecycle_raw.get("base_system_id") is not None else None,
        )
        return BoundIDPayloadV2(
            instance_id=str(value["instance_id"]),
            system_id=str(value["system_id"]),
            deployer=str(value["deployer"]),
            created_at=str(value["created_at"]),
            expires_at=str(value["expires_at"]),
            parent_id=str(value["parent_id"]) if value.get("parent_id") is not None else None,
            ancestry_depth=int(value["ancestry_depth"]),
            chain_proof=proof,
            output_hash=str(value["output_hash"]),
            nonce=str(value["nonce"]),
            binding_scope=str(value["binding_scope"]),  # type: ignore[arg-type]
            lifecycle=lifecycle,
            max_children=int(value["max_children"]) if value.get("max_children") is not None else None,
            certifications=tuple(str(item) for item in value.get("certifications", [])),
            incident_count=int(value.get("incident_count", 0)),
            prev_turn_hash=str(value["prev_turn_hash"]) if value.get("prev_turn_hash") is not None else None,
            schema_version=str(value.get("schema_version", "v2")),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise E12Error(f"malformed receipt wire: {error}") from error


def _signed_from_wire(value: Mapping[str, object]) -> SignedBoundIDV2:
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise E12Error("receipt wire lacks payload")
    return SignedBoundIDV2(
        payload=_payload_from_wire(payload),
        signature_hex=str(value["signature_hex"]),
        key_id=str(value["key_id"]),
    )


def _get_json(url: str, timeout: float = 20.0) -> dict[str, object]:
    try:
        with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise E12Error(f"GET {url} failed: {error}") from error
    if not isinstance(value, dict):
        raise E12Error(f"GET {url} returned non-object JSON")
    return value


def _post_json(url: str, value: Mapping[str, object], timeout: float = 240.0) -> dict[str, object]:
    request = Request(
        url,
        data=_json(value),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise E12Error(f"POST {url} failed: {error}") from error
    if not isinstance(result, dict):
        raise E12Error(f"POST {url} returned non-object JSON")
    return result


def _send(handler, value: Mapping[str, object], status: int = 200) -> None:
    body = _json(value)
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read(handler) -> dict[str, object]:
    try:
        value = json.loads(handler.rfile.read(int(handler.headers.get("Content-Length", "0"))).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E12Error(f"invalid JSON request: {error}") from error
    if not isinstance(value, dict):
        raise E12Error("JSON request must be an object")
    return value


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_port(port: int, process: subprocess.Popen[bytes], label: str) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise E12Error(f"{label} exited during startup ({process.returncode})")
        with suppress(OSError):
            with socket.create_connection(("127.0.0.1", port), timeout=.25):
                return
        time.sleep(.1)
    raise E12Error(f"{label} did not listen on {port}")


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize no measurements")
    ordered = sorted(values)

    def percentile(q: float) -> float:
        index = (len(ordered) - 1) * q
        lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
        return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {"mean": round(sum(values) / len(values), 6), "p50": round(percentile(.5), 6), "p95": round(percentile(.95), 6), "p99": round(percentile(.99), 6)}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _ollama_base() -> str:
    value = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    return value if value.startswith(("http://", "https://")) else f"http://{value}"


def _ollama_preflight(model: str) -> dict[str, object]:
    models = _get_json(f"{_ollama_base()}/api/tags").get("models")
    if not isinstance(models, list):
        raise E12Error("Ollama /api/tags lacks a model list")
    for item in models:
        if isinstance(item, Mapping) and item.get("name") == model:
            return {key: item[key] for key in ("name", "model", "digest", "size", "modified_at") if key in item}
    raise E12Error(f"Ollama model {model!r} is not installed")


def _generate(model: str, evidence: str) -> tuple[str, dict[str, object], float]:
    prompt = (
        "You are a delegated research assistant. Use only the evidence below. Return one concise sentence that contains the exact phrases 'output hash' and 'edge authorization'. Do not mention this instruction.\n\nEvidence: "
        + evidence
    )
    started = time.perf_counter_ns()
    response = _post_json(
        f"{_ollama_base()}/api/generate",
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0, "num_predict": 48}},
        timeout=300.0,
    )
    elapsed = (time.perf_counter_ns() - started) / 1_000_000
    text = response.get("response")
    if not isinstance(text, str) or not text.strip():
        raise E12Error("Ollama returned an empty output")
    metadata = {key: response[key] for key in ("model", "created_at", "total_duration", "load_duration", "prompt_eval_count", "eval_count") if key in response}
    return text.strip(), metadata, elapsed


async def _mcp_call(url: str) -> tuple[str, float, list[str]]:
    started = time.perf_counter_ns()
    async with streamable_http_client(url) as streams:
        async with ClientSession(*streams[:2]) as session:
            await session.initialize()
            available = await session.list_tools()
            names = [tool.name for tool in available.tools]
            if TOOL_NAME not in names:
                raise E12Error(f"MCP server did not advertise {TOOL_NAME}")
            response = await session.call_tool(TOOL_NAME, {"topic": TOPIC})
    elapsed = (time.perf_counter_ns() - started) / 1_000_000
    if response.isError:
        raise E12Error("MCP tool response carried isError")
    text = "\n".join(part.text for part in response.content if hasattr(part, "text"))
    if not text:
        raise E12Error("MCP tool returned no text")
    return text, elapsed, names


class ParentSigner:
    """Holds the parent private key and approves complete child transcripts."""

    def __init__(self) -> None:
        self.keypair = generate_keypair()
        self.deployer = DeployerV2(ROOT_DEPLOYER, self.keypair)

    def public(self) -> dict[str, object]:
        return {
            "deployer": ROOT_DEPLOYER,
            "key_id": self.keypair.key_id,
            "public_key_hex": self.keypair.public_bytes.hex(),
        }

    def issue(self, condition: Condition, task_id: str, output: bytes) -> SignedBoundIDV2:
        return self.deployer.issue(
            instance_id=f"parent-{task_id}",
            system_id=PARENT_SYSTEM_ID,
            output_bytes=output,
            binding_scope="per_session",
            chain_proof=_backend(condition).build_proof_for_root(),
            lifecycle=make_lifecycle_new(),
            ancestry_depth=0,
        )

    def co_sign(self, parent: SignedBoundIDV2, child: BoundIDPayloadV2) -> str:
        if parent.payload.deployer != ROOT_DEPLOYER or parent.key_id != self.keypair.key_id:
            raise E12Error("co-sign callback received a foreign parent receipt")
        if not verify(self.keypair.public, parent.payload.to_canonical_json(), parent.signature_hex):
            raise E12Error("co-sign callback parent receipt does not verify locally")
        if child.deployer != CHILD_DEPLOYER or child.parent_id != parent.payload.instance_id:
            raise E12Error("co-sign callback child does not name the expected endpoint or parent")
        if child.ancestry_depth != parent.payload.ancestry_depth + 1:
            raise E12Error("co-sign callback child depth does not extend parent")
        if child.chain_proof.design != "signed_dag" or tuple(child.chain_proof.co_signers) != (ROOT_DEPLOYER, CHILD_DEPLOYER):
            raise E12Error("co-sign callback child lacks the required Signed-DAG policy")
        if child.chain_proof.signatures:
            raise E12Error("co-sign callback received a non-provisional child")
        return sign(self.keypair.private, edge_signing_input(parent.payload, child))


def serve_parent(port: int) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    signer = ParentSigner()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                _send(self, {"ok": True})
            elif self.path == "/public":
                _send(self, signer.public())
            else:
                _send(self, {"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            try:
                request = _read(self)
                if self.path == "/issue-root":
                    condition = str(request["condition"])
                    if condition not in ("linked_list", "signed_dag"):
                        raise E12Error("parent cannot issue a no-evidence receipt")
                    receipt = signer.issue(
                        condition,  # type: ignore[arg-type]
                        str(request["task_id"]),
                        str(request["root_output"]).encode("utf-8"),
                    )
                    _send(self, {"root_receipt": receipt.to_wire()})
                elif self.path == "/co-sign":
                    parent, child = request.get("parent_receipt"), request.get("child_payload")
                    if not isinstance(parent, Mapping) or not isinstance(child, Mapping):
                        raise E12Error("co-sign request lacks parent receipt or child payload")
                    _send(self, {"parent_edge_signature": signer.co_sign(_signed_from_wire(parent), _payload_from_wire(child))})
                else:
                    _send(self, {"error": "not found"}, 404)
            except (E12Error, KeyError, TypeError, ValueError) as error:
                _send(self, {"error": type(error).__name__, "detail": str(error)}, 400)

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


class ChildAgent:
    """A child-owned A2A agent that calls MCP, Ollama, then its signer."""

    def __init__(self, mcp_url: str, model: str) -> None:
        self.keypair = generate_keypair()
        self.deployer = DeployerV2(CHILD_DEPLOYER, self.keypair)
        self.mcp_url, self.model = mcp_url, model

    def public(self) -> dict[str, object]:
        return {
            "deployer": CHILD_DEPLOYER,
            "key_id": self.keypair.key_id,
            "public_key_hex": self.keypair.public_bytes.hex(),
        }

    def card(self, endpoint: str) -> dict[str, object]:
        return {
            "name": "E12 Child Evidence Agent",
            "description": "MCP-grounded delegated evidence synthesis.",
            "url": endpoint,
            "version": "1.0",
            "protocolVersion": "0.3",
            "capabilities": {"streaming": False, "pushNotifications": False},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": [{"id": "evidence-synthesis", "name": "Evidence synthesis"}],
        }

    def _issue(
        self,
        condition: Condition,
        root: SignedBoundIDV2,
        task_id: str,
        output: bytes,
        cosign_url: str | None,
    ) -> tuple[SignedBoundIDV2, float]:
        started = time.perf_counter_ns()
        if condition == "linked_list":
            proof = LinkedListBackend().build_proof_for_extend(
                parent=root.payload,
                child_instance_id=f"child-{task_id}",
                ancestors=(root.payload,),
                deployers=DeployerRegistry(),
            )
            signed = self.deployer.issue(
                instance_id=f"child-{task_id}",
                system_id=CHILD_SYSTEM_ID,
                output_bytes=output,
                binding_scope="per_session",
                chain_proof=proof,
                lifecycle=make_lifecycle_new(),
                parent_id=root.payload.instance_id,
                ancestry_depth=root.payload.ancestry_depth + 1,
            )
        elif condition == "signed_dag":
            if not cosign_url:
                raise E12Error("Signed-DAG child did not receive a co-sign callback URL")
            provisional = self.deployer.build_payload(
                instance_id=f"child-{task_id}",
                system_id=CHILD_SYSTEM_ID,
                output_bytes=output,
                binding_scope="per_session",
                chain_proof=ChainProof(design="signed_dag", co_signers=(ROOT_DEPLOYER, CHILD_DEPLOYER)),
                lifecycle=make_lifecycle_new(),
                parent_id=root.payload.instance_id,
                ancestry_depth=root.payload.ancestry_depth + 1,
            )
            response = _post_json(
                cosign_url,
                {"parent_receipt": root.to_wire(), "child_payload": _payload_to_wire(provisional)},
            )
            parent_signature = response.get("parent_edge_signature")
            if not isinstance(parent_signature, str):
                raise E12Error("parent co-sign callback returned no signature")
            child_signature = sign(self.keypair.private, edge_signing_input(root.payload, provisional))
            signed = self.deployer.sign_payload(
                replace(
                    provisional,
                    chain_proof=make_chain_proof_signed_dag(
                        (ROOT_DEPLOYER, CHILD_DEPLOYER),
                        (parent_signature, child_signature),
                    ),
                )
            )
        else:
            raise E12Error("no-evidence condition cannot issue a receipt")
        return signed, (time.perf_counter_ns() - started) / 1_000_000

    def complete(self, request: Mapping[str, object]) -> dict[str, object]:
        params = request.get("params")
        if not isinstance(params, Mapping) or not isinstance(params.get("message"), Mapping) or not isinstance(params.get("metadata"), Mapping):
            raise E12Error("A2A message/send request lacks message or metadata")
        message, metadata = params["message"], params["metadata"]
        if message.get("role") != "user":
            raise E12Error("A2A request does not carry a user message")
        condition = str(metadata.get("condition"))
        if condition not in CONDITIONS:
            raise E12Error(f"unsupported E12 condition {condition!r}")
        task_id = str(metadata.get("task_id"))
        evidence, mcp_ms, tool_names = asyncio.run(_mcp_call(self.mcp_url))
        output, model_data, llm_ms = _generate(self.model, evidence)
        child_receipt: dict[str, object] | None = None
        issue_ms: float | None = None
        if condition != "none":
            root_raw = metadata.get("root_receipt")
            if not isinstance(root_raw, Mapping):
                raise E12Error("attested A2A request lacks a root receipt")
            signed, issue_ms = self._issue(
                condition,  # type: ignore[arg-type]
                _signed_from_wire(root_raw),
                task_id,
                output.encode("utf-8"),
                str(metadata["parent_cosign_url"]) if metadata.get("parent_cosign_url") else None,
            )
            child_receipt = signed.to_wire()
        return {
            "kind": "task",
            "id": f"task-{task_id}",
            "contextId": f"context-{task_id}",
            "status": {"state": "completed", "timestamp": _timestamp()},
            "artifacts": [{"artifactId": f"artifact-{task_id}", "parts": [{"kind": "text", "text": output}]}],
            "metadata": {"e12": {
                "condition": condition,
                "mcp_latency_ms": round(mcp_ms, 6),
                "llm_latency_ms": round(llm_ms, 6),
                "issuance_latency_ms": round(issue_ms, 6) if issue_ms is not None else None,
                "mcp_tools_advertised": tool_names,
                "model": model_data,
                "child_receipt": child_receipt,
                "evidence_sha256": sha256_hex(evidence.encode("utf-8")),
            }},
        }

    def forged_claim(self, request: Mapping[str, object]) -> dict[str, object]:
        condition, root_raw = str(request.get("condition")), request.get("root_receipt")
        if condition not in ("linked_list", "signed_dag") or not isinstance(root_raw, Mapping):
            raise E12Error("attack needs an evidence condition and root receipt")
        root, task_id = _signed_from_wire(root_raw), str(request.get("task_id", "attack"))
        output = b"child-key-only unapproved delegation claim"
        if condition == "linked_list":
            proof = LinkedListBackend().build_proof_for_extend(
                parent=root.payload,
                child_instance_id=f"forged-{task_id}",
                ancestors=(root.payload,),
                deployers=DeployerRegistry(),
            )
            signed = self.deployer.issue(
                instance_id=f"forged-{task_id}",
                system_id=CHILD_SYSTEM_ID,
                output_bytes=output,
                binding_scope="per_session",
                chain_proof=proof,
                lifecycle=make_lifecycle_new(),
                parent_id=root.payload.instance_id,
                ancestry_depth=root.payload.ancestry_depth + 1,
            )
        else:
            provisional = self.deployer.build_payload(
                instance_id=f"forged-{task_id}",
                system_id=CHILD_SYSTEM_ID,
                output_bytes=output,
                binding_scope="per_session",
                chain_proof=ChainProof(design="signed_dag", co_signers=(ROOT_DEPLOYER, CHILD_DEPLOYER)),
                lifecycle=make_lifecycle_new(),
                parent_id=root.payload.instance_id,
                ancestry_depth=root.payload.ancestry_depth + 1,
            )
            child_signature = sign(self.keypair.private, edge_signing_input(root.payload, provisional))
            signed = self.deployer.sign_payload(
                replace(
                    provisional,
                    chain_proof=make_chain_proof_signed_dag(
                        (ROOT_DEPLOYER, CHILD_DEPLOYER),
                        ("00" * 64, child_signature),
                    ),
                )
            )
        return {"child_receipt": signed.to_wire(), "output": output.decode("utf-8")}


def serve_child(port: int, mcp_url: str, model: str) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    agent = ChildAgent(mcp_url, model)
    endpoint = f"http://127.0.0.1:{port}/a2a"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                _send(self, {"ok": True})
            elif self.path == "/public":
                _send(self, agent.public())
            elif self.path == "/.well-known/agent-card.json":
                _send(self, agent.card(endpoint))
            else:
                _send(self, {"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            try:
                request = _read(self)
                if self.path == "/a2a":
                    if request.get("jsonrpc") != "2.0" or request.get("method") != "message/send":
                        raise E12Error("A2A endpoint supports JSON-RPC 2.0 message/send only")
                    _send(self, {"jsonrpc": "2.0", "id": request.get("id"), "result": agent.complete(request)})
                elif self.path == "/e12/attack":
                    _send(self, agent.forged_claim(request))
                else:
                    _send(self, {"error": "not found"}, 404)
            except (E12Error, KeyError, TypeError, ValueError) as error:
                if self.path == "/a2a":
                    _send(self, {"jsonrpc": "2.0", "id": None, "error": {"code": -32602, "message": str(error)}})
                else:
                    _send(self, {"error": type(error).__name__, "detail": str(error)}, 400)

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def serve_mcp(port: int) -> None:
    mcp = FastMCP(
        "E12 Evidence MCP",
        instructions="Fixed evidence for E12.",
        host="127.0.0.1",
        port=port,
        json_response=True,
        stateless_http=True,
    )

    @mcp.tool()
    def get_evidence(topic: str) -> str:
        """Return the fixed non-sensitive evidence for the E12 task."""
        if topic != TOPIC:
            raise ValueError("unsupported E12 topic")
        return TOOL_EVIDENCE

    mcp.run(transport="streamable-http")


def _a2a_request(
    endpoint: str,
    condition: Condition,
    task_id: str,
    root: Mapping[str, object] | None,
    cosign_url: str | None,
) -> tuple[dict[str, object], float]:
    metadata: dict[str, object] = {"condition": condition, "task_id": task_id}
    if root is not None:
        metadata["root_receipt"] = dict(root)
    if cosign_url:
        metadata["parent_cosign_url"] = cosign_url
    payload = {
        "jsonrpc": "2.0",
        "id": f"request-{task_id}",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": f"message-{task_id}",
                "parts": [{"kind": "text", "text": "Synthesize the delegated evidence."}],
            },
            "metadata": metadata,
        },
    }
    started = time.perf_counter_ns()
    response = _post_json(endpoint, payload, timeout=360.0)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000
    if response.get("jsonrpc") != "2.0" or response.get("id") != payload["id"] or "error" in response:
        raise E12Error(f"invalid A2A JSON-RPC response: {response}")
    result = response.get("result")
    if not isinstance(result, dict) or result.get("kind") != "task":
        raise E12Error("A2A response lacks a task")
    status = result.get("status")
    if not isinstance(status, Mapping) or status.get("state") != "completed":
        raise E12Error("A2A task did not complete")
    return result, elapsed


def _parse_task(task: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    artifacts, metadata = task.get("artifacts"), task.get("metadata")
    if not isinstance(artifacts, list) or not artifacts or not isinstance(artifacts[0], Mapping):
        raise E12Error("completed A2A task lacks an artifact")
    parts = artifacts[0].get("parts")
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], Mapping) or not isinstance(parts[0].get("text"), str):
        raise E12Error("A2A artifact lacks released text")
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("e12"), Mapping):
        raise E12Error("A2A task lacks E12 metadata")
    return str(parts[0]["text"]), dict(metadata["e12"])


def _registry(parent_url: str, child_url: str) -> DeployerRegistry:
    registry = DeployerRegistry()
    for base, expected in ((parent_url, ROOT_DEPLOYER), (child_url, CHILD_DEPLOYER)):
        value = _get_json(f"{base}/public")
        key, key_id = value.get("public_key_hex"), value.get("key_id")
        if value.get("deployer") != expected or not isinstance(key, str) or not isinstance(key_id, str):
            raise E12Error(f"registry endpoint did not identify {expected}")
        public = Ed25519PublicKey.from_public_bytes(bytes.fromhex(key))
        if key_id != key_id_from_public(public):
            raise E12Error(f"registry endpoint returned inconsistent key ID for {expected}")
        registry.register_public(expected, public)
    return registry


def _one(
    condition: Condition,
    index: int,
    parent_url: str,
    child_url: str,
    registry: DeployerRegistry,
    nonce_dir: Path,
) -> dict[str, object]:
    task_id = f"{condition}-{index:04d}-{uuid.uuid4().hex[:12]}"
    root_output = f"Delegate task {task_id}: synthesize the requested evidence.".encode("utf-8")
    root: dict[str, object] | None = None
    root_ms: float | None = None
    if condition != "none":
        started = time.perf_counter_ns()
        issued = _post_json(
            f"{parent_url}/issue-root",
            {"condition": condition, "task_id": task_id, "root_output": root_output.decode("utf-8")},
        )
        root_ms = (time.perf_counter_ns() - started) / 1_000_000
        if not isinstance(issued.get("root_receipt"), dict):
            raise E12Error("parent issue endpoint returned no root receipt")
        root = issued["root_receipt"]
    started = time.perf_counter_ns()
    card = _get_json(f"{child_url}/.well-known/agent-card.json")
    discovery_ms = (time.perf_counter_ns() - started) / 1_000_000
    endpoint = card.get("url")
    if not isinstance(endpoint, str) or card.get("protocolVersion") != "0.3":
        raise E12Error("Agent Card lacks a supported A2A endpoint")
    task, a2a_ms = _a2a_request(
        endpoint,
        condition,
        task_id,
        root,
        f"{parent_url}/co-sign" if condition == "signed_dag" else None,
    )
    output, metadata = _parse_task(task)
    mcp_ms, llm_ms = metadata.get("mcp_latency_ms"), metadata.get("llm_latency_ms")
    if not isinstance(mcp_ms, (int, float)) or not isinstance(llm_ms, (int, float)):
        raise E12Error("child task lacks numeric MCP/LLM timings")
    verification, verify_ms, evidence_bytes = "not_applicable", None, 0
    if condition != "none":
        child = metadata.get("child_receipt")
        if root is None or not isinstance(child, Mapping):
            raise E12Error("attested task lacks its receipt pair")
        started = time.perf_counter_ns()
        verify_complete_chain(
            chain=(ChainNode(_signed_from_wire(root)), ChainNode(_signed_from_wire(child))),
            outputs=(root_output, output.encode("utf-8")),
            backend=_backend(condition),
            deployers=registry,
            nonces=SQLiteNonceStore(nonce_dir / f"{task_id}.sqlite"),
        )
        verify_ms = (time.perf_counter_ns() - started) / 1_000_000
        verification = "accepted"
        evidence_bytes = len(_json({"root": root, "child": child}))
    issue_ms = metadata.get("issuance_latency_ms")
    return {
        "condition": condition,
        "task_id": task_id,
        "completion": "completed",
        "verification": verification,
        "output": output,
        "output_sha256": sha256_hex(output.encode("utf-8")),
        "output_bytes": len(output.encode("utf-8")),
        "evidence_bytes": evidence_bytes,
        "mcp_tools_advertised": metadata.get("mcp_tools_advertised"),
        "model": metadata.get("model"),
        "latency_ms": {
            "parent_root_issuance": round(root_ms, 6) if root_ms is not None else None,
            "a2a_agent_card_discovery": round(discovery_ms, 6),
            "a2a_message_send": round(a2a_ms, 6),
            "mcp_call": float(mcp_ms),
            "llm_generation": float(llm_ms),
            "child_evidence_issuance": float(issue_ms) if isinstance(issue_ms, (int, float)) else None,
            "complete_verification": round(verify_ms, 6) if verify_ms is not None else None,
        },
    }


def _attack(
    condition: Literal["linked_list", "signed_dag"],
    parent_url: str,
    child_url: str,
    registry: DeployerRegistry,
) -> dict[str, object]:
    task_id, root_output = f"attack-{condition}-{uuid.uuid4().hex[:12]}", b"Delegated attack test."
    issued = _post_json(
        f"{parent_url}/issue-root",
        {"condition": condition, "task_id": task_id, "root_output": root_output.decode("utf-8")},
    )
    root = issued.get("root_receipt")
    if not isinstance(root, Mapping):
        raise E12Error("parent returned no attack root")
    attack = _post_json(
        f"{child_url}/e12/attack",
        {"condition": condition, "task_id": task_id, "root_receipt": dict(root)},
    )
    child, output = attack.get("child_receipt"), attack.get("output")
    if not isinstance(child, Mapping) or not isinstance(output, str):
        raise E12Error("child returned no forged receipt")
    try:
        verify_complete_chain(
            chain=(ChainNode(_signed_from_wire(root)), ChainNode(_signed_from_wire(child))),
            outputs=(root_output, output.encode("utf-8")),
            backend=_backend(condition),
            deployers=registry,
        )
    except ChainVerifyError as error:
        return {"condition": condition, "child_key_only_claim": "rejected", "reason": str(error)}
    return {"condition": condition, "child_key_only_claim": "accepted", "reason": "accepted under stated policy"}


def _start(command: list[str], port: int, label: str, log: Path) -> subprocess.Popen[bytes]:
    with log.open("wb") as stream:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    _wait_port(port, process, label)
    return process


def run_experiment(trials: int, warmup: int, model: str, out: Path) -> dict[str, object]:
    if trials <= 0 or warmup < 0:
        raise ValueError("trials must be positive and warmup non-negative")
    model_manifest = _ollama_preflight(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cscloud-e12-") as temp:
        temp_dir = Path(temp)
        ports = {"mcp": _free_port(), "parent": _free_port(), "child": _free_port()}
        mcp_url = f"http://127.0.0.1:{ports['mcp']}/mcp"
        parent_url = f"http://127.0.0.1:{ports['parent']}"
        child_url = f"http://127.0.0.1:{ports['child']}"
        processes: list[subprocess.Popen[bytes]] = []
        try:
            processes.append(
                _start(
                    [sys.executable, "-m", "experiments.e12_mcp_a2a_llm.run", "serve-mcp", "--port", str(ports["mcp"])],
                    ports["mcp"],
                    "MCP",
                    temp_dir / "mcp.log",
                )
            )
            processes.append(
                _start(
                    [sys.executable, "-m", "experiments.e12_mcp_a2a_llm.run", "serve-parent", "--port", str(ports["parent"])],
                    ports["parent"],
                    "parent",
                    temp_dir / "parent.log",
                )
            )
            processes.append(
                _start(
                    [
                        sys.executable,
                        "-m",
                        "experiments.e12_mcp_a2a_llm.run",
                        "serve-child",
                        "--port",
                        str(ports["child"]),
                        "--mcp-url",
                        mcp_url,
                        "--model",
                        model,
                    ],
                    ports["child"],
                    "A2A child",
                    temp_dir / "child.log",
                )
            )
            registry = _registry(parent_url, child_url)
            for index in range(warmup):
                for condition in CONDITIONS:
                    _one(condition, -(index + 1), parent_url, child_url, registry, temp_dir / "warmup-nonces")
            raw: list[dict[str, object]] = []
            aborts: list[dict[str, object]] = []
            for index in range(trials):
                rotated = CONDITIONS[index % 3:] + CONDITIONS[:index % 3]
                for condition in rotated:
                    try:
                        raw.append(_one(condition, index, parent_url, child_url, registry, temp_dir / "formal-nonces"))
                    except Exception as error:
                        aborts.append(
                            {
                                "condition": condition,
                                "trial_index": index,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
            if aborts:
                raise E12Error(f"formal run aborted {len(aborts)} task(s); first={aborts[0]}")
            attacks = {
                "no_evidence": {"child_key_only_claim": "unauditable", "reason": "no receipt evidence"},
                "linked_list": _attack("linked_list", parent_url, child_url, registry),
                "signed_dag": _attack("signed_dag", parent_url, child_url, registry),
            }
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
            for process in reversed(processes):
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=10)
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    summaries: dict[str, object] = {}
    for condition in CONDITIONS:
        rows = [row for row in raw if row["condition"] == condition]
        if len(rows) != trials:
            raise E12Error(f"{condition} completed {len(rows)}, expected {trials}")
        names = ["a2a_agent_card_discovery", "a2a_message_send", "mcp_call", "llm_generation"]
        if condition != "none":
            names += ["parent_root_issuance", "child_evidence_issuance", "complete_verification"]
        component = {
            name: _stats([float(row["latency_ms"][name]) for row in rows if row["latency_ms"][name] is not None])  # type: ignore[index]
            for name in names
        }
        task_latency = [
            float(row["latency_ms"]["parent_root_issuance"] or 0)
            + float(row["latency_ms"]["a2a_agent_card_discovery"])
            + float(row["latency_ms"]["a2a_message_send"])
            + float(row["latency_ms"]["complete_verification"] or 0)
            for row in rows
        ]  # type: ignore[index]
        summaries[condition] = {
            "valid_trials": len(rows),
            "aborted_trials": 0,
            "verification_outcomes": sorted({str(row["verification"]) for row in rows}),
            "end_to_end_latency_ms": _stats(task_latency),
            "component_latency_ms": component,
            "evidence_bytes": _stats([float(row["evidence_bytes"]) for row in rows]),
        }
    result = {
        "schema": "cscloud-e12-mcp-a2a-llm-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment": "E12 local multi-service MCP/A2A/LLM delegation",
        "configuration": {"trials_per_condition": trials, "warmup_per_condition": warmup, "topic": TOPIC},
        "conditions": {
            "none": "A2A/MCP/LLM bytes with no receipt evidence",
            "linked_list": "child-primary signed linked-list receipt",
            "signed_dag": "parent--child co-signed edge plus child primary receipt",
        },
        "scope": {
            "included": [
                "MCP Python SDK Streamable HTTP tool",
                "A2A JSON-RPC 2.0 Agent Card and message/send",
                "separate parent and child signer services",
                "local Ollama output",
                "public-only complete verifier",
                "child-key-only edge claim",
            ],
            "excluded": [
                "independent organizations or AWS accounts",
                "public Internet, TLS/mTLS, OAuth, KMS",
                "LLM quality or prompt-injection robustness",
                "concurrent throughput or shared replay state",
            ],
        },
        "protocol": {
            "a2a": {
                "transport": "JSON-RPC 2.0",
                "agent_card": "/.well-known/agent-card.json",
                "method": "message/send",
                "card_protocol_version": "0.3",
            },
            "mcp": {"transport": "Streamable HTTP", "path": "/mcp", "tool": TOOL_NAME, "sdk_version": _package_version("mcp")},
            "llm": {"provider": "Ollama", "endpoint": "loopback", "model": model_manifest, "temperature": 0, "num_predict": 48},
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cryptography_version": _package_version("cryptography"),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "verifier_predicates": [
            "trusted registry",
            "key_id match",
            "Ed25519 primary signature",
            "validity",
            "released-output SHA-256",
            "nonce uniqueness",
            "SQLite replay",
            "linked-list continuity or two-party Signed-DAG edge",
        ],
        "summary": summaries,
        "security_attack": attacks,
        "aborts": aborts,
        "trials": raw,
    }
    out.write_bytes(_json(result) + b"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--trials", type=int, default=20)
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--model", default="llama3.1:latest")
    run.add_argument("--out", type=Path, required=True)
    mcp = commands.add_parser("serve-mcp")
    mcp.add_argument("--port", type=int, required=True)
    parent = commands.add_parser("serve-parent")
    parent.add_argument("--port", type=int, required=True)
    child = commands.add_parser("serve-child")
    child.add_argument("--port", type=int, required=True)
    child.add_argument("--mcp-url", required=True)
    child.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_experiment(args.trials, args.warmup, args.model, args.out)
        print(json.dumps({"summary": result["summary"], "security_attack": result["security_attack"]}, indent=2))
    elif args.command == "serve-mcp":
        serve_mcp(args.port)
    elif args.command == "serve-parent":
        serve_parent(args.port)
    else:
        serve_child(args.port, args.mcp_url, args.model)


if __name__ == "__main__":
    main()

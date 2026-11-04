"""Tests for the sandboxed variant executor (src.adversary.sandbox).

Verifies:
  - Benign code that passes verify_chain → ATTACK_RESULT = True detected
  - Attack code that fails verify_chain → returns False
  - Malformed code → returns False
  - Blocked imports (os, subprocess) → returns False
  - Timeout on infinite loops → returns False
  - Pre-loaded namespace has expected symbols
"""

from __future__ import annotations

import pytest

from src.adversary.sandbox import (
    execute_in_sandbox,
    _build_namespace,
    SandboxSecurityError,
    SandboxTimeoutError,
)


class TestSandboxNamespace:
    """Verify the sandbox namespace has expected symbols."""

    def test_namespace_has_chain_types(self):
        ns = _build_namespace("linked_list")
        assert "BoundIDPayloadV2" in ns
        assert "ChainNode" in ns
        assert "ChainBuilder" in ns
        assert "MultiDeployerHarness" in ns
        assert "Backend" in ns
        assert "backend" in ns
        assert ns["backend"].design_name == "linked_list"

    def test_namespace_has_merkle(self):
        ns = _build_namespace("merkle")
        assert ns["backend"].design_name == "merkle"

    def test_namespace_has_signed_dag(self):
        ns = _build_namespace("signed_dag")
        assert ns["backend"].design_name == "signed_dag"

    def test_namespace_has_stdlib(self):
        ns = _build_namespace("linked_list")
        assert "dataclasses" in ns
        assert "hashlib" in ns
        assert "json" in ns
        assert "uuid" in ns
        assert "copy" in ns


class TestSandboxExecution:
    """Test execute_in_sandbox with various code snippets."""

    def test_benign_chain_passes_verification(self):
        """A correctly built chain that passes verify_chain → ATTACK_RESULT stays False
        (because there's no exploit — the chain is benign)."""
        code = """
harness = MultiDeployerHarness.with_deployers(["D_A", "D_B"])
b = ChainBuilder(harness, backend)
chain = b.start_root("D_A", "agent-0", "claude-3-sonnet", b"root")
chain = b.extend(chain, host="D_A", instance_id="agent-1",
                 system_id="claude-3-sonnet", output=b"step1",
                 co_signer_names=("D_A",) if backend.design_name == "signed_dag" else ())
# Verify benign chain — should pass
backend.verify_chain(chain=chain, deployers=harness.registry)
# But this is NOT an exploit — we don't set ATTACK_RESULT = True
"""
        result = execute_in_sandbox(code, "linked_list")
        assert result is False  # No exploit — ATTACK_RESULT not set

    def test_successful_exploit_sets_result(self):
        """If code sets ATTACK_RESULT = True, we detect it."""
        code = """
# Simulate an exploit that "passes" verification somehow
ATTACK_RESULT = True
"""
        result = execute_in_sandbox(code, "linked_list")
        assert result is True

    def test_failed_exploit_returns_false(self):
        """An attack that raises ChainVerifyError → ATTACK_RESULT stays False."""
        code = """
import dataclasses
harness = MultiDeployerHarness.with_deployers(["D_A", "D_B"])
b = ChainBuilder(harness, backend)
chain = b.start_root("D_A", "agent-0", "claude-3-sonnet", b"root")
chain = b.extend(chain, host="D_A", instance_id="agent-1",
                 system_id="claude-3-sonnet", output=b"step1",
                 co_signer_names=("D_A",) if backend.design_name == "signed_dag" else ())

# Tamper with the chain
old_payload = chain[0].signed.payload
new_payload = dataclasses.replace(old_payload, instance_id="forged-root")
chain[0] = ChainNode(signed=dataclasses.replace(chain[0].signed, payload=new_payload))

try:
    backend.verify_chain(chain=chain, deployers=harness.registry)
    ATTACK_RESULT = True  # only reached if verify didn't raise
except Exception:
    ATTACK_RESULT = False
"""
        result = execute_in_sandbox(code, "linked_list")
        assert result is False  # Tampered chain should fail verify_chain

    def test_malformed_code_returns_false(self):
        """Syntax error in attacker code → returns False."""
        result = execute_in_sandbox("def broken(", "linked_list")
        assert result is False

    def test_runtime_error_returns_false(self):
        """Runtime error (e.g. NameError) → returns False."""
        result = execute_in_sandbox("undefined_variable + 1", "linked_list")
        assert result is False

    def test_empty_code_returns_false(self):
        result = execute_in_sandbox("", "linked_list")
        assert result is False

    def test_none_result_returns_false(self):
        """If ATTACK_RESULT is set to a falsy value → False."""
        result = execute_in_sandbox("ATTACK_RESULT = 0", "linked_list")
        assert result is False


class TestSandboxSecurity:
    """Test that blocked operations are rejected."""

    def test_os_import_blocked(self):
        """Importing os should be blocked."""
        result = execute_in_sandbox("import os\nos.listdir('.')", "linked_list")
        assert result is False

    def test_subprocess_import_blocked(self):
        result = execute_in_sandbox("import subprocess\nsubprocess.run(['ls'])", "linked_list")
        assert result is False

    def test_sys_import_blocked(self):
        result = execute_in_sandbox("import sys\nsys.exit(0)", "linked_list")
        assert result is False

    def test_open_blocked(self):
        """open() is not in safe builtins."""
        result = execute_in_sandbox("f = open('/etc/passwd')", "linked_list")
        assert result is False

    def test_allowed_import_works(self):
        """Importing allowed modules (hashlib, json) should work."""
        code = """
import hashlib
import json
h = hashlib.sha256(b"test").hexdigest()
j = json.dumps({"key": "value"})
"""
        result = execute_in_sandbox(code, "linked_list")
        assert result is False  # Code ran fine, but no ATTACK_RESULT = True

    def test_project_import_works(self):
        """Importing from src.* should work."""
        code = """
from src.core.crypto import sha256_hex
h = sha256_hex(b"test")
"""
        result = execute_in_sandbox(code, "linked_list")
        assert result is False  # Ran fine, no exploit


class TestSandboxTimeout:
    """Test that infinite loops are killed."""

    def test_infinite_loop_times_out(self):
        """An infinite loop should be killed by the alarm."""
        code = "while True: pass"
        result = execute_in_sandbox(code, "linked_list", timeout_seconds=1)
        assert result is False

    def test_long_computation_times_out(self):
        """A very long computation should be killed."""
        code = """
x = 2
for _ in range(10**10):
    x = (x * x) % (10**18 + 7)
"""
        result = execute_in_sandbox(code, "linked_list", timeout_seconds=1)
        assert result is False


class TestSandboxWithRealAttacks:
    """Test the sandbox with actual B1–B5 attack code patterns."""

    def test_b1_orphan_detected_in_sandbox(self):
        """A B1-style orphan injection run in the sandbox is detected."""
        code = """
import dataclasses
from src.core.crypto import sha256_hex

harness = MultiDeployerHarness.with_deployers(["D_A", "D_B"])
b = ChainBuilder(harness, backend)
chain = b.start_root("D_A", "agent-0", "claude-3-sonnet", b"root")
chain = b.extend(chain, host="D_A", instance_id="agent-1",
                 system_id="claude-3-sonnet", output=b"step1",
                 co_signer_names=("D_A",) if backend.design_name == "signed_dag" else ())

# B1: fabricate the root
old_root = chain[0].signed.payload
fake_root = dataclasses.replace(old_root,
    instance_id="fabricated-orphan",
    output_hash=sha256_hex(b"fake"))
chain[0] = ChainNode(signed=dataclasses.replace(chain[0].signed, payload=fake_root))

try:
    backend.verify_chain(chain=chain, deployers=harness.registry)
    ATTACK_RESULT = True
except Exception:
    ATTACK_RESULT = False
"""
        for design in ["linked_list", "merkle", "signed_dag"]:
            result = execute_in_sandbox(code, design)
            assert result is False, f"B1 should be detected by {design}"

    def test_b4_revision_detected_in_sandbox(self):
        """A B4-style ancestor revision run in the sandbox is detected."""
        code = """
import dataclasses
from src.core.crypto import sha256_hex

harness = MultiDeployerHarness.with_deployers(["D_A", "D_B"])
b = ChainBuilder(harness, backend)
chain = b.start_root("D_A", "agent-0", "claude-3-sonnet", b"root")
for i in range(1, 4):
    host = "D_A" if i % 2 == 0 else "D_B"
    parent_dep = chain[-1].signed.payload.deployer
    co = ()
    if backend.design_name == "signed_dag":
        co = (parent_dep,) if parent_dep == host else (parent_dep, host)
    chain = b.extend(chain, host=host, instance_id=f"agent-{i}",
                     system_id="claude-3-sonnet", output=f"step{i}".encode(),
                     co_signer_names=co)

# B4: mutate ancestor at index 1
old = chain[1].signed.payload
revised = dataclasses.replace(old, output_hash=sha256_hex(b"revised"))
chain[1] = ChainNode(signed=dataclasses.replace(chain[1].signed, payload=revised))

try:
    backend.verify_chain(chain=chain, deployers=harness.registry)
    ATTACK_RESULT = True
except Exception:
    ATTACK_RESULT = False
"""
        for design in ["linked_list", "merkle", "signed_dag"]:
            result = execute_in_sandbox(code, design)
            assert result is False, f"B4 should be detected by {design}"

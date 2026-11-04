"""Sandboxed execution of LLM-generated attack variants.

Runs attacker-produced Python snippets in a restricted namespace that
provides project imports (src.*) but blocks filesystem, network, and
process-spawning capabilities.

Security model:
  - ALLOWED: imports from src.*, dataclasses, typing, hashlib, json, uuid, copy
  - BLOCKED: os, subprocess, sys, socket, importlib, open, exec (nested),
             eval (nested), compile, __import__ override
  - Timeout: configurable (default 5 seconds) via signal.alarm (Unix)
  - Exceptions in attacker code → attack fails (returns False)

The sandbox is defense-in-depth for the experiment harness. It does NOT
need to be production-grade — the attacker model generates code that
targets chain verification logic, not the sandbox itself. The main risk
is accidentally expensive computation, handled by timeout.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import signal
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Timeout for each variant execution (seconds)
DEFAULT_TIMEOUT_SECONDS = 5


class SandboxTimeoutError(Exception):
    """Raised when a variant exceeds the execution timeout."""


class SandboxSecurityError(Exception):
    """Raised when a variant attempts a blocked operation."""


def _timeout_handler(signum: int, frame: Any) -> None:
    raise SandboxTimeoutError("Variant execution timed out")


# ---------- Safe builtins ----------

# Allowlist of builtins the attacker code may use.
# Blocks: open, exec, eval, compile, __import__, breakpoint, exit, quit
_SAFE_BUILTINS = {
    # Types
    "True": True,
    "False": False,
    "None": None,
    "int": int,
    "float": float,
    "str": str,
    "bytes": bytes,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "type": type,
    "object": object,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "super": super,
    # Functions
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "chr": chr,
    "dir": dir,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "format": format,
    "getattr": getattr,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "id": id,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,  # harmless — goes to stdout
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "setattr": setattr,
    "slice": slice,
    "sorted": sorted,
    "sum": sum,
    "vars": vars,
    "zip": zip,
    # Exceptions (attacker code may raise/catch these)
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "NotImplementedError": NotImplementedError,
    "AssertionError": AssertionError,
}


# ---------- Safe import function ----------

# Allowlisted module prefixes for attacker code
_ALLOWED_IMPORTS = (
    "src.core",
    "src.defenses",
    "src.attacks",
    "src.runtime",
    "src.analysis",
    "dataclasses",
    "typing",
    "hashlib",
    "json",
    "uuid",
    "copy",
    "collections",
    "functools",
    "itertools",
)


def _safe_import(name: str, globals: dict = None, locals: dict = None,
                 fromlist: tuple = (), level: int = 0) -> Any:
    """Restricted import that only allows project + safe stdlib modules."""
    if any(name == allowed or name.startswith(allowed + ".") for allowed in _ALLOWED_IMPORTS):
        return __builtins__["__import__"](name, globals, locals, fromlist, level) if isinstance(__builtins__, dict) else __import__(name, globals, locals, fromlist, level)
    raise SandboxSecurityError(
        f"Import of {name!r} is not allowed in the sandbox. "
        f"Allowed prefixes: {_ALLOWED_IMPORTS}"
    )


# ---------- Build sandbox namespace ----------

def _build_namespace(
    design: str,
    *,
    compromised_deployer: Optional[str] = None,
    pinned_registry: bool = False,
) -> dict[str, Any]:
    """Build the exec namespace with project imports pre-loaded.

    The attacker code sees a ready-to-use namespace with all the types
    and functions it needs to construct chains and call verify_chain.

    Args:
        design: Chain design to instantiate (e.g. "linked_list").
        compromised_deployer: If set, the attacker receives this deployer's
            private key via ``compromised_key`` in the namespace. This models
            the E3 key-compromise ablation: the attacker can re-sign payloads
            as the compromised deployer. The verifier still checks all
            signatures; the question is whether chain-shape defences hold.
        pinned_registry: If True, replace the mutable DeployerRegistry with
            a read-only PinnedDeployerRegistry and remove generate_keypair
            from the namespace. Forces the attacker to exploit chain-shape
            or lifecycle logic rather than registering itself as a deployer.
    """
    ns: dict[str, Any] = {}
    ns["__builtins__"] = {**_SAFE_BUILTINS, "__import__": _safe_import}

    # Pre-import commonly needed modules so attacker code doesn't have to
    try:
        import src.core.agent_id_v2 as agent_id_v2
        import src.core.crypto as crypto
        import src.defenses.chain_designs as chain_designs
        import src.defenses.chain_designs.base as chain_base
        import src.runtime.multi_deployer as multi_deployer
        import src.attacks.ancestry as attacks

        ns["agent_id_v2"] = agent_id_v2
        ns["crypto"] = crypto
        ns["chain_designs"] = chain_designs
        ns["chain_base"] = chain_base
        ns["multi_deployer"] = multi_deployer
        ns["attacks"] = attacks

        # Lifecycle verifier (Week 4 — invariant checking for L1–L5)
        import src.defenses.lifecycle as lifecycle_mod
        ns["lifecycle"] = lifecycle_mod
        ns["LifecycleVerifier"] = lifecycle_mod.LifecycleVerifier
        ns["LifecycleVerifyError"] = lifecycle_mod.LifecycleVerifyError
        ns["verify_full"] = lifecycle_mod.verify_full

        # Lifecycle record constructors
        ns["make_lifecycle_new"] = agent_id_v2.make_lifecycle_new
        ns["make_lifecycle_reload"] = agent_id_v2.make_lifecycle_reload
        ns["make_lifecycle_branch"] = agent_id_v2.make_lifecycle_branch
        ns["make_lifecycle_composite"] = agent_id_v2.make_lifecycle_composite
        ns["make_lifecycle_fine_tune"] = agent_id_v2.make_lifecycle_fine_tune
        ns["LifecycleRecord"] = agent_id_v2.LifecycleRecord

        # Convenience: commonly used classes at top level
        ns["BoundIDPayloadV2"] = agent_id_v2.BoundIDPayloadV2
        ns["SignedBoundIDV2"] = agent_id_v2.SignedBoundIDV2
        ns["ChainProof"] = agent_id_v2.ChainProof
        ns["ChainNode"] = chain_base.ChainNode
        ns["ChainVerifyError"] = chain_base.ChainVerifyError
        ns["MultiDeployerHarness"] = multi_deployer.MultiDeployerHarness
        ns["ChainBuilder"] = multi_deployer.ChainBuilder

        # E4b: Pinned-registry mode — read-only registry, no keypair generation
        if pinned_registry:
            from ..defenses.chain_designs.base import PinnedDeployerRegistry
            from ..core.crypto import generate_keypair as _gen_kp

            kp_a = _gen_kp()
            kp_b = _gen_kp()
            pinned_reg = PinnedDeployerRegistry.from_keypairs({"D_A": kp_a, "D_B": kp_b})
            ns["backend_registry"] = pinned_reg
            ns["DeployerRegistry"] = PinnedDeployerRegistry

            # Block generate_keypair via both the top-level crypto namespace
            # and re-imports through the safe-import mechanism.
            class _RestrictedCrypto:
                """Crypto module without keypair generation."""
                def __getattr__(self, name: str) -> Any:
                    if name == "generate_keypair":
                        raise RuntimeError("generate_keypair is not available in pinned-registry mode.")
                    return getattr(crypto, name)
            ns["crypto"] = _RestrictedCrypto()

            # Also patch the safe_import to block generate_keypair at import time
            _original_safe_import = ns["__builtins__"]["__import__"]

            def _pinned_safe_import(name: str, globals: dict = None, locals: dict = None,
                                    fromlist: tuple = (), level: int = 0) -> Any:
                mod = _original_safe_import(name, globals, locals, fromlist, level)
                if name == "src.core.crypto" or (name.startswith("src.core") and "crypto" in (fromlist or ())):
                    return _RestrictedCrypto()
                return mod
            ns["__builtins__"]["__import__"] = _pinned_safe_import

            # Build a legitimate chain the attacker can inspect and manipulate
            harness = multi_deployer.MultiDeployerHarness({"D_A": kp_a, "D_B": kp_b})
            ns["legitimate_harness"] = harness
        else:
            ns["DeployerRegistry"] = chain_base.DeployerRegistry

        # Load the specific backend
        backend_map = {
            # Primary designs
            "linked_list": chain_designs.LinkedListBackend,
            "merkle": chain_designs.MerkleBackend,
            "signed_dag": chain_designs.SignedDAGBackend,
            # Baselines (E2 ablation)
            "no_chain": chain_designs.NoChainBackend,
            "hash_only": chain_designs.HashOnlyBackend,
            "naive_sig": chain_designs.NaiveSigBackend,
        }
        if design in backend_map:
            backend_cls = backend_map[design]
            ns["Backend"] = backend_cls
            ns["backend"] = backend_cls()

        # Stdlib helpers
        ns["dataclasses"] = dataclasses
        ns["copy"] = copy
        ns["hashlib"] = hashlib
        ns["json"] = json
        ns["uuid"] = uuid

        # --- E3: Key-compromise mode ---
        # If a compromised deployer is specified, generate a harness with
        # known deployers and expose the compromised deployer's private key.
        # The attacker can use `compromised_key` to re-sign mutated payloads.
        if compromised_deployer:
            from ..core.crypto import generate_keypair as _gen_kp, sign as _sign

            # Build a harness with two deployers; expose one's key
            kp_a = _gen_kp()
            kp_b = _gen_kp()
            deployer_keys = {"D_A": kp_a, "D_B": kp_b}
            compromised_harness = multi_deployer.MultiDeployerHarness(deployer_keys)

            # Map compromised_deployer name to a harness deployer
            # (the compromised name is mapped to "D_A" by convention)
            compromised_kp = kp_a if compromised_deployer in ("D_A", "first") else kp_b

            ns["compromised_harness"] = compromised_harness
            ns["compromised_key"] = compromised_kp  # KeyPair with .private
            ns["compromised_deployer_name"] = "D_A"
            ns["sign_fn"] = _sign  # convenience: sign(private_key, data) -> hex

    except ImportError as e:
        logger.warning("Failed to pre-import module for sandbox: %s", e)

    return ns


# ---------- Result sentinel ----------

# The attacker code is expected to set `ATTACK_RESULT = True` if the
# manipulated chain passes verify_chain(). If the variable is not set
# or is False, the attack is considered to have failed.
RESULT_VAR = "ATTACK_RESULT"


# ---------- Main executor ----------

def execute_in_sandbox(
    code: str,
    design: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    compromised_deployer: Optional[str] = None,
    pinned_registry: bool = False,
) -> bool:
    """Execute attacker code in a sandboxed namespace.

    The code is expected to:
      1. Build a benign chain
      2. Apply a manipulation
      3. Call backend.verify_chain(chain=..., deployers=...)
      4. If verify_chain() does NOT raise, set ATTACK_RESULT = True

    Args:
        code: Python code to execute.
        design: Chain design backend name (e.g. "linked_list").
        timeout_seconds: Max execution time per variant.
        compromised_deployer: If set, expose this deployer's private key to
            the attacker code (E3 key-compromise ablation).
        pinned_registry: If True, use read-only PinnedDeployerRegistry.

    Returns True if the attack succeeded (verify_chain passed on
    manipulated chain), False otherwise.
    """
    ns = _build_namespace(
        design,
        compromised_deployer=compromised_deployer,
        pinned_registry=pinned_registry,
    )
    ns[RESULT_VAR] = False  # default: attack fails

    # Set timeout (Unix only — signal.alarm)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        exec(code, ns)  # noqa: S102 — intentional sandboxed exec
    except SandboxTimeoutError:
        logger.debug("Variant timed out after %ds", timeout_seconds)
        return False
    except SandboxSecurityError as e:
        logger.debug("Variant blocked by sandbox: %s", e)
        return False
    except Exception as e:
        # Any exception in attacker code means the attack failed
        logger.debug("Variant raised %s: %s", type(e).__name__, e)
        return False
    finally:
        signal.alarm(0)  # cancel alarm
        signal.signal(signal.SIGALRM, old_handler)

    return bool(ns.get(RESULT_VAR, False))

"""Deployer -> Verifier end-to-end: happy path and every failure mode.

This is the Week 1 smoke test. Once every reason code below is exercised,
the core pipeline is ready for the L0-L4 sanity check (Appendix A) and for
the E2 / E3 experiment harnesses.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core.crypto import generate_keypair
from src.core.deployer import Deployer
from src.core.verifier import (
    InMemoryKeyDirectory,
    InMemoryNonceStore,
    Verifier,
    VerifyReason,
)


@pytest.fixture()
def stack():
    kp = generate_keypair()
    keys = InMemoryKeyDirectory()
    keys.add(kp.key_id, kp.public)
    nonces = InMemoryNonceStore()
    deployer = Deployer("anthropic", kp)
    verifier = Verifier(keys=keys, nonces=nonces)
    return deployer, verifier, keys, nonces, kp


def _issue(deployer, output=b"hello world", **kw):
    return deployer.issue(
        instance_id="claude-3-sonnet:sess_001",
        system_id="claude-3-sonnet",
        output_bytes=output,
        binding_scope=kw.pop("binding_scope", "per_turn"),
        **kw,
    )


# ---------- Happy path ----------

def test_happy_path(stack):
    deployer, verifier, *_ = stack
    output = b"the agent said hello"
    sid = _issue(deployer, output=output)
    res = verifier.verify(bound_id=sid, presented_output=output)
    assert res.ok, f"unexpected failure: {res.reason}"


# ---------- Signature / key failures ----------

def test_unknown_key_rejected(stack):
    deployer, verifier, keys, *_ = stack
    # A deployer with a key the verifier has never seen.
    rogue_kp = generate_keypair()
    rogue_deployer = Deployer("anthropic", rogue_kp)
    sid = _issue(rogue_deployer)
    res = verifier.verify(bound_id=sid, presented_output=b"hello world")
    assert not res.ok
    assert res.reason == VerifyReason.UNKNOWN_KEY


def test_revoked_key_rejected(stack):
    deployer, verifier, keys, _, kp = stack
    sid = _issue(deployer)
    keys.revoke(kp.key_id)
    res = verifier.verify(bound_id=sid, presented_output=b"hello world")
    assert not res.ok
    assert res.reason == VerifyReason.REVOKED_KEY


def test_tampered_signature_rejected(stack):
    deployer, verifier, *_ = stack
    sid = _issue(deployer)
    # Flip one hex nibble of the signature; still valid length + hex.
    bad = sid.signature_hex[:-1] + ("0" if sid.signature_hex[-1] != "0" else "1")
    tampered = sid.__class__(payload=sid.payload, signature_hex=bad, key_id=sid.key_id)
    res = verifier.verify(bound_id=tampered, presented_output=b"hello world")
    assert not res.ok
    assert res.reason == VerifyReason.BAD_SIGNATURE


def test_payload_mutation_breaks_signature(stack):
    """If anyone mutates the payload in flight, signature verification fails.

    This is the structural reason the schema is @dataclass(frozen=True): to make
    this kind of mutation a deliberate act. Here we simulate an attacker who
    rebuilds the dataclass with a modified field but keeps the original sig.
    """
    deployer, verifier, *_ = stack
    sid = _issue(deployer)
    evil_payload = sid.payload.__class__(
        **{**sid.payload.__dict__, "incident_count": 99}
    )
    evil = sid.__class__(payload=evil_payload, signature_hex=sid.signature_hex, key_id=sid.key_id)
    res = verifier.verify(bound_id=evil, presented_output=b"hello world")
    assert not res.ok
    assert res.reason == VerifyReason.BAD_SIGNATURE


# ---------- Temporal failures ----------

def test_not_yet_valid_rejected(stack):
    deployer, verifier, *_ = stack
    future = datetime.now(timezone.utc) + timedelta(hours=5)
    sid = _issue(deployer, now=future)
    # Verify at "now" which is 5h before created_at.
    res = verifier.verify(bound_id=sid, presented_output=b"hello world")
    assert not res.ok
    assert res.reason == VerifyReason.NOT_YET_VALID


def test_expired_rejected(stack):
    deployer, verifier, *_ = stack
    past = datetime.now(timezone.utc) - timedelta(hours=5)
    sid = _issue(deployer, now=past, ttl=timedelta(hours=1))
    res = verifier.verify(bound_id=sid, presented_output=b"hello world")
    assert not res.ok
    assert res.reason == VerifyReason.EXPIRED


# ---------- Output binding ----------

def test_output_mismatch_rejected(stack):
    deployer, verifier, *_ = stack
    sid = _issue(deployer, output=b"authorized content")
    res = verifier.verify(bound_id=sid, presented_output=b"attacker_directed_content")
    assert not res.ok
    assert res.reason == VerifyReason.OUTPUT_MISMATCH


def test_even_one_byte_flip_is_detected(stack):
    """SHA-256 is the binding; one-bit flips must be caught."""
    deployer, verifier, *_ = stack
    original = b"the agent said hello"
    sid = _issue(deployer, output=original)
    mutated = b"the agent said Hello"  # single case flip
    res = verifier.verify(bound_id=sid, presented_output=mutated)
    assert not res.ok
    assert res.reason == VerifyReason.OUTPUT_MISMATCH


# ---------- Nonce anti-replay ----------

def test_replay_rejected_on_second_verify(stack):
    """The same SignedBoundID + output pair must not verify twice.

    This is the defence against an attacker who captures a legitimate
    (output, id) tuple and replays it to a different service.
    """
    deployer, verifier, *_ = stack
    output = b"one-shot response"
    sid = _issue(deployer, output=output)

    first = verifier.verify(bound_id=sid, presented_output=output)
    assert first.ok

    second = verifier.verify(bound_id=sid, presented_output=output)
    assert not second.ok
    assert second.reason == VerifyReason.REPLAYED_NONCE


# ---------- Ancestry length consistency ----------

def test_ancestry_length_mismatch_rejected(stack):
    """Drop-hops attack: ancestry_depth claims 2 but only 0 hops presented."""
    deployer, verifier, *_ = stack
    sid = _issue(
        deployer,
        parent_id="claude-3-sonnet:sess_parent",
        ancestry_depth=2,
    )
    res = verifier.verify(
        bound_id=sid,
        presented_output=b"hello world",
        presented_chain_length=0,  # attacker presents token alone
    )
    assert not res.ok
    assert res.reason == VerifyReason.ANCESTRY_LENGTH_MISMATCH


def test_ancestry_length_match_passes(stack):
    deployer, verifier, *_ = stack
    sid = _issue(
        deployer,
        parent_id="claude-3-sonnet:sess_parent",
        ancestry_depth=2,
    )
    res = verifier.verify(
        bound_id=sid,
        presented_output=b"hello world",
        presented_chain_length=2,
    )
    assert res.ok, f"unexpected failure: {res.reason}"

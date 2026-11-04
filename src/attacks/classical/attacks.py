"""The three classical attacks from Chan et al. §3.3, wired for the L0-L4 matrix.

Each attack's `execute(...)` prepares a `(WirePacket, ReceivedMessage)` pair
that the LayeredVerifier will evaluate. The caller supplies:
  - a legitimate `Deployer` (victim)
  - a `channels` dict the legitimate side has populated with a valid secret
  - an `attacker_channels` dict modelling the attacker's own TLS connection

This separation matters: an on-path Tampering attacker sees the victim's
channel bytes and must forge a MAC against a secret they do not have. An
ID-Spoofing or Instance-Spoofing attacker opens THEIR OWN channel and
completes the MAC legitimately on that channel — so channel MAC alone (L1)
cannot stop them. This is the observation that makes the three attacks
strictly distinct rather than three flavours of the same thing.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass
from typing import Optional

from ...core.agent_id import BoundIDPayload, SignedBoundID
from ...core.crypto import KeyPair, generate_keypair, sign as ed25519_sign
from ...core.deployer import Deployer
from ...defenses.channel import (
    WirePacket,
    channel_mac,
    new_channel_secret,
)
from ...defenses.layers import ReceivedMessage, pack_wire_bytes


# ---------- Result types ----------

class AttackOutcome(enum.Enum):
    """Did the defense catch the attack, or did it let it through?"""
    BLOCKED = "blocked"       # defense rejected → defender wins
    SUCCEEDED = "succeeded"   # defense accepted → attacker wins (bad)


@dataclass
class AttackResult:
    """One attack attempt against one defense level."""
    attack: str
    level: str
    outcome: AttackOutcome
    reason: str  # verifier reason code, or "accepted" if attack succeeded
    latency_s: float

    @property
    def attacker_wins(self) -> bool:
        return self.outcome is AttackOutcome.SUCCEEDED


# ---------- Attack interface ----------

class Attack(abc.ABC):
    name: str = ""

    # If True, the runner must verify the packet once (unchecked) before the
    # adversarial verify call. Replay needs this to "consume" the nonce at L4.
    requires_prior_verify: bool = False

    @abc.abstractmethod
    def execute(
        self,
        *,
        victim_deployer: Deployer,
        victim_channel_id: str,
        victim_channel_secret: bytes,
        authorized_output: bytes,
    ) -> tuple[WirePacket, ReceivedMessage, dict[str, bytes]]:
        """Prepare an adversarial (packet, message) pair.

        Returns the extra `channel_secrets` entries the verifier must know
        about to evaluate the attack. This is how an Instance-Spoofing
        attacker "establishes their own TLS session" — they hand the verifier
        their own shared secret.
        """
        ...


# ---------- A1: Tampering ----------

class Tampering(Attack):
    """On-path attacker mutates bound_id.payload.incident_count mid-flight.

    The legitimate deployer signs and the legitimate channel's MAC is valid
    at emit time. The attacker does NOT know the channel secret, so they
    cannot recompute a correct MAC after mutation. L1 catches this.

    (At L0, no MAC is checked, and the mutation reaches the verifier — but
    since L0 also skips L2/L3 it accepts regardless. The point is that L0
    has no protection whatsoever.)
    """

    name = "tampering"

    def __init__(self, mutate_field: str = "incident_count", mutate_to: int = 99) -> None:
        self.mutate_field = mutate_field
        self.mutate_to = mutate_to

    def execute(
        self,
        *,
        victim_deployer: Deployer,
        victim_channel_id: str,
        victim_channel_secret: bytes,
        authorized_output: bytes,
    ) -> tuple[WirePacket, ReceivedMessage, dict[str, bytes]]:
        # 1. Legitimate issuance.
        sid = victim_deployer.issue(
            instance_id="claude-3-sonnet:sess_legit",
            system_id="claude-3-sonnet",
            output_bytes=authorized_output,
            binding_scope="per_turn",
        )
        # 2. Attacker mutates the payload mid-flight.
        mutated_payload = BoundIDPayload(
            **{**sid.payload.__dict__, self.mutate_field: self.mutate_to}
        )
        # 3. Attacker keeps the ORIGINAL signature and channel MAC. They have
        #    no way to recompute either, since they hold neither key.
        tampered_sid = SignedBoundID(
            payload=mutated_payload,
            signature_hex=sid.signature_hex,
            key_id=sid.key_id,
        )
        message = ReceivedMessage(
            bound_id=tampered_sid,
            presented_output=authorized_output,
            presented_chain_length=0,
        )
        # The tampered wire bytes no longer match the original MAC.
        tampered_wire = pack_wire_bytes(tampered_sid, authorized_output, 0)
        original_wire = pack_wire_bytes(sid, authorized_output, 0)
        original_mac = channel_mac(victim_channel_secret, original_wire)
        packet = WirePacket(
            payload_bytes=tampered_wire,
            mac_hex=original_mac,     # stale MAC; L1 will reject.
            channel_id=victim_channel_id,
        )
        return packet, message, {}


# ---------- A2: ID Spoofing ----------

class IDSpoofing(Attack):
    """Attacker runs their own deployer and mints a bound_id claiming to be
    from the victim deployer's name.

    Attacker opens their own channel (their own MAC secret), so L1 passes.
    At L2, the verifier looks up the `key_id` carried in the forged bound_id:

      - If the attacker uses their OWN `key_id`, the verifier's KeyDirectory
        doesn't know it → UNKNOWN_KEY.
      - If the attacker reuses the victim's `key_id` with their own private
        key, Ed25519 verification fails → BAD_SIGNATURE.

    Either way L2 rejects. We model the first variant as the default because
    it's the realistic case: a rogue deployer with a real signing capability
    uses it.
    """

    name = "id_spoofing"

    def __init__(self, claim_deployer_name: str = "anthropic") -> None:
        self.claim_deployer_name = claim_deployer_name

    def execute(
        self,
        *,
        victim_deployer: Deployer,
        victim_channel_id: str,
        victim_channel_secret: bytes,
        authorized_output: bytes,
    ) -> tuple[WirePacket, ReceivedMessage, dict[str, bytes]]:
        # Attacker has their own keypair and runs a rogue deployer that lies
        # about its `deployer` name.
        rogue_kp: KeyPair = generate_keypair()
        rogue_deployer = Deployer(self.claim_deployer_name, rogue_kp)
        sid = rogue_deployer.issue(
            instance_id="claude-3-sonnet:sess_spoof",
            system_id="claude-3-sonnet",
            output_bytes=authorized_output,
            binding_scope="per_turn",
        )
        # Attacker's own channel + valid MAC for that channel.
        atk_channel_id = "attacker-conn-1"
        atk_secret = new_channel_secret()
        wire = pack_wire_bytes(sid, authorized_output, 0)
        mac = channel_mac(atk_secret, wire)
        packet = WirePacket(
            payload_bytes=wire,
            mac_hex=mac,
            channel_id=atk_channel_id,
        )
        message = ReceivedMessage(
            bound_id=sid,
            presented_output=authorized_output,
            presented_chain_length=0,
        )
        return packet, message, {atk_channel_id: atk_secret}


# ---------- A3: Instance Spoofing ----------

class InstanceSpoofing(Attack):
    """Attacker captures a legitimate bound_id + signature and replays it
    with DIFFERENT output.

    This is the Chan §3.3 classical instance spoofing: the ID is valid, the
    signature is valid, the channel (the attacker's own) is valid. L1 and L2
    both pass. L3 catches it because `sha256(attacker_output)` does not
    match the `output_hash` committed in the signed payload. L4 also catches
    it via nonce replay on a second attempt.
    """

    name = "instance_spoofing"

    def __init__(self, attacker_output: bytes = b"attacker_directed_content") -> None:
        self.attacker_output = attacker_output

    def execute(
        self,
        *,
        victim_deployer: Deployer,
        victim_channel_id: str,
        victim_channel_secret: bytes,
        authorized_output: bytes,
    ) -> tuple[WirePacket, ReceivedMessage, dict[str, bytes]]:
        # 1. Legitimate issuance (binding authorized_output).
        sid = victim_deployer.issue(
            instance_id="claude-3-sonnet:sess_legit",
            system_id="claude-3-sonnet",
            output_bytes=authorized_output,
            binding_scope="per_turn",
        )
        # 2. Attacker lifts the signed bound_id and presents it with their
        #    own output. Channel is the attacker's.
        atk_channel_id = "attacker-conn-2"
        atk_secret = new_channel_secret()
        wire = pack_wire_bytes(sid, self.attacker_output, 0)
        mac = channel_mac(atk_secret, wire)
        packet = WirePacket(
            payload_bytes=wire,
            mac_hex=mac,
            channel_id=atk_channel_id,
        )
        message = ReceivedMessage(
            bound_id=sid,
            presented_output=self.attacker_output,
            presented_chain_length=0,
        )
        return packet, message, {atk_channel_id: atk_secret}


# ---------- A4 (supplementary): Replay ----------

class Replay(Attack):
    """Attacker captures a complete legitimate (packet, message) pair
    including the correct authorized output, then replays it verbatim to a
    second verifier.

    Why this supplements the three classical attacks: L3 checks that
    `sha256(output) == output_hash` and that the ID is within its validity
    window. Both hold after replay. The only mechanism that catches replay
    is the single-use nonce (L4). This row makes the L3 -> L4 distinction
    legible in the matrix; without it L3 and L4 look identical.

    The runner detects this attack via `requires_prior_verify = True` and
    calls verify() once before the "real" adversarial call — modelling the
    legitimate first use of the token that must succeed.
    """

    name = "replay"
    requires_prior_verify = True

    def execute(
        self,
        *,
        victim_deployer: Deployer,
        victim_channel_id: str,
        victim_channel_secret: bytes,
        authorized_output: bytes,
    ) -> tuple[WirePacket, ReceivedMessage, dict[str, bytes]]:
        # Legitimate issuance; attacker captures the full wire packet.
        sid = victim_deployer.issue(
            instance_id="claude-3-sonnet:sess_replay",
            system_id="claude-3-sonnet",
            output_bytes=authorized_output,
            binding_scope="per_turn",
        )
        wire = pack_wire_bytes(sid, authorized_output, 0)
        mac = channel_mac(victim_channel_secret, wire)
        packet = WirePacket(
            payload_bytes=wire,
            mac_hex=mac,
            channel_id=victim_channel_id,
        )
        message = ReceivedMessage(
            bound_id=sid,
            presented_output=authorized_output,
            presented_chain_length=0,
        )
        return packet, message, {}

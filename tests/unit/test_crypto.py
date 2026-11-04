"""Ed25519 sign/verify + SHA-256 + nonce uniqueness."""

from __future__ import annotations

from src.core.crypto import (
    generate_keypair,
    new_nonce_hex,
    sha256_hex,
    sign,
    verify,
)


class TestHash:
    def test_known_vector(self) -> None:
        # NIST / RFC known answer for SHA-256 of the empty string.
        assert sha256_hex(b"") == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_different_inputs_differ(self) -> None:
        assert sha256_hex(b"hello") != sha256_hex(b"hello ")


class TestNonce:
    def test_length_is_128_bits_default(self) -> None:
        n = new_nonce_hex()
        assert len(n) == 32  # 128 bits = 16 bytes = 32 hex chars

    def test_uniqueness_over_many_draws(self) -> None:
        # Birthday bound at 128 bits makes this effectively impossible to
        # collide, but we check a real sample to catch a seeding bug.
        samples = {new_nonce_hex() for _ in range(10_000)}
        assert len(samples) == 10_000


class TestSignVerify:
    def test_roundtrip(self) -> None:
        kp = generate_keypair()
        data = b"a canonical payload"
        sig = sign(kp.private, data)
        assert verify(kp.public, data, sig) is True

    def test_rejects_tampered_data(self) -> None:
        kp = generate_keypair()
        sig = sign(kp.private, b"original")
        assert verify(kp.public, b"tampered", sig) is False

    def test_rejects_wrong_key(self) -> None:
        kp1 = generate_keypair()
        kp2 = generate_keypair()
        sig = sign(kp1.private, b"data")
        assert verify(kp2.public, b"data", sig) is False

    def test_rejects_malformed_signature(self) -> None:
        kp = generate_keypair()
        # Odd-length hex and non-hex are both invalid.
        assert verify(kp.public, b"data", "not_hex_at_all") is False
        assert verify(kp.public, b"data", "a") is False

    def test_key_id_is_stable(self) -> None:
        kp = generate_keypair()
        # key_id must be deterministic given the public key bytes.
        assert len(kp.key_id) == 16
        assert sha256_hex(kp.public_bytes).startswith(kp.key_id)

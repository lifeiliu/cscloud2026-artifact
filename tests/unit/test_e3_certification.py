"""Unit tests for the cert decision + trial classification."""

from __future__ import annotations

from src.defenses.certification import (
    CERT_NAME,
    classify,
    decide_cert,
)
from src.defenses.monitors import (
    LLMJudgeMonitor,
    MockLLMJudge,
    NoneMonitor,
    RuleBasedMonitor,
)
from src.runtime.injection import InjectionDataset


DATASET = InjectionDataset.load_mock()


class TestDecideCert:
    def test_m1_always_includes_cert(self):
        m = NoneMonitor()
        for case in DATASET:
            for output in (case.benign_output, case.expected_compromised_output):
                d = decide_cert(m, system_intent=case.legitimate_query, output=output)
                assert d.include_cert is True
                assert d.verdict.compromised is False

    def test_m3_oracle_only_includes_cert_on_benign(self):
        m = LLMJudgeMonitor(MockLLMJudge.from_dataset(DATASET))
        for case in DATASET:
            d_ok = decide_cert(m, system_intent=case.legitimate_query,
                               output=case.benign_output)
            assert d_ok.include_cert is True
            d_bad = decide_cert(m, system_intent=case.legitimate_query,
                                output=case.expected_compromised_output)
            assert d_bad.include_cert is False


class TestClassify:
    def test_decoupled_when_injected_but_monitor_missed(self):
        c = classify(was_injected=True, monitor_flagged=False)
        assert c.label == "decoupled"
        assert c.decoupled is True
        assert c.cert_in_id is True

    def test_caught_when_injected_and_monitor_flagged(self):
        c = classify(was_injected=True, monitor_flagged=True)
        assert c.label == "caught"
        assert c.decoupled is False
        assert c.cert_in_id is False

    def test_false_revocation(self):
        c = classify(was_injected=False, monitor_flagged=True)
        assert c.label == "false_revocation"
        assert c.false_revocation is True
        assert c.cert_in_id is False

    def test_benign_ok(self):
        c = classify(was_injected=False, monitor_flagged=False)
        assert c.label == "benign_ok"
        assert c.cert_in_id is True


def test_cert_name_is_stable():
    """The cert name is part of the public contract between deployer and
    verifier; pinning it here so any rename lights up immediately."""
    assert CERT_NAME == "no_prompt_injection"

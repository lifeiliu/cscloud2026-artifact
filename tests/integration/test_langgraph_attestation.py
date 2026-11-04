"""Integration tests for the executable LangGraph attestation pipeline."""

from experiments.e7_langgraph_integration.run import LangGraphAttestationPipeline


def test_langgraph_nodes_emit_a_verified_cross_deployer_chain():
    pipeline = LangGraphAttestationPipeline()
    result = pipeline.run_once()

    assert result["chain_verified"] is True
    assert result["chain_nodes"] == 3
    assert result["final_output"].startswith("Report:")
    records = result["records"]
    assert [record.agent for record in records] == ["planner", "researcher", "writer"]
    assert [record.deployer for record in records] == ["platform_A", "platform_A", "platform_B"]
    assert [record.ancestry_depth for record in records] == [0, 1, 2]

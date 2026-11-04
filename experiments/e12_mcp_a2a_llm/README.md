# E12: MCP/A2A/LLM end-to-end evaluation

E12 launches separate local services for a Python MCP SDK Streamable-HTTP tool,
an A2A JSON-RPC child agent, and a parent signer. The child discovers the A2A
endpoint through its Agent Card, calls the MCP tool, asks local Ollama to emit
the released output, and then uses the selected evidence condition.

```bash
python3 -m experiments.e12_mcp_a2a_llm.run run \
  --trials 30 --warmup 10 --model llama3.1:latest \
  --out paper/acsac2026/cscloud2026/review-stage/experiments/e12_mcp_a2a_llm/results.json
```

The raw result separates completed tasks, aborts, exact output hashes,
component latency, and the child-key-only edge-claim verdict. It is not a
public-Internet, cloud, mTLS/OAuth, KMS, or independent-organization evaluation.

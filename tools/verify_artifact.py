#!/usr/bin/env python3
"""Check distribution hashes and independently account for recorded results."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

def check(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    for name, info in manifest['files'].items():
        p = ROOT / name
        check(p.is_file() and sha(p) == info['sha256'] and p.stat().st_size == info['bytes'], name)
    formal = ROOT / 'experiments/e13_authenticated_fanout/results/formal_20260913'
    public = json.loads((formal / 'public_snapshot_hashes.json').read_text())
    original = json.loads((formal / 'source_hashes.json').read_text())
    for name, expected in public.items():
        check(sha(formal / 'source_snapshot' / name) == expected, name)
        if name.endswith('.py'):
            check(expected == original[name], 'Executable snapshot modified: ' + name)
    check(sha(formal / 'events.jsonl') == '6792c70c7b1ccb8d5b7df84b8ed98eff43bc2cc8346aee6f5878f4c9c6206a64', 'Signed corpus changed')
    kinds, boundary = Counter(), Counter()
    inputs = {True: set(), False: set()}
    sequences = set()
    for line in (formal / 'events.jsonl').open():
        event = json.loads(line)
        kinds[event['kind']] += 1
        if event['kind'] == 'serial':
            check(event['actual'] == event['expected'], 'Verdict mismatch')
            req_hash = hashlib.sha256(json.dumps(event['request'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            check(req_hash == event['request_hash'], 'Request hash mismatch')
            if event['actual'] != 'accepted':
                check(event['state_before'] == event['state_after'], 'Rejected request changed state')
            if event['block'] == 'boundary':
                boundary[(event['control'], event['actual'])] += 1
                inputs[event['control']].add(event['request_hash'])
                sequences.add(event['store'])
        elif event['kind'] == 'race':
            check(len(event['results']) == 4, 'Race process count')
            check(sum(r['code'] == 'accepted' for r in event['results']) == 1, 'Race winner count')
        elif event['kind'] == 'crash':
            check(event['state_before'] == event['state_after'], 'Crash changed state')
    check(kinds == {'serial': 7137, 'race': 18, 'restart': 9, 'crash': 9}, 'Event accounting')
    check(inputs[True] == inputs[False] and len(inputs[True]) == 3420, 'Paired candidate sets')
    check(len(sequences) == 360, 'Sequence count')
    check(boundary[(True, 'accepted')] == 3420 and boundary[(False, 'accepted')] == 2340, 'Boundary accepts')
    check(sum(v for (control, verdict), v in boundary.items() if not control and verdict != 'accepted') == 1080, 'Overflow rejects')
    structural = 0
    for folder in ('asr_20260830_063346', 'complete_receipt_asr_20260830_064451'):
        with (ROOT / 'artifacts/current' / folder / 'asr_matrix.csv').open() as f:
            rows = [r for r in csv.DictReader(f) if r['attack'] in {'B1', 'B2', 'B3', 'B4'}]
        check(len(rows) == 12, 'Structural cells')
        check(all(int(r['n_successes']) == 0 for r in rows), 'Structural successes')
        count = sum(int(r['n_trials']) for r in rows)
        check(count == 12000, 'Structural trials')
        structural += count
    current = ROOT / 'artifacts/current'
    live = json.loads((current / 'e12_mcp_a2a_llm/results.json').read_text())
    check(sha(ROOT / 'experiments/e12_mcp_a2a_llm/run.py') == live['environment']['source_sha256'], 'E12 runner provenance')
    live_means = {}
    for condition, expected in {'none': 770.8, 'linked_list': 826.2, 'signed_dag': 813.1}.items():
        rows = [r for r in live['trials'] if r['condition'] == condition]
        check(len(rows) == 30 and all(r['completion'] == 'completed' for r in rows), 'Live task completion')
        if condition != 'none':
            check(all(r['verification'] == 'accepted' for r in rows), 'Live verification')
        ms = mean(sum(float(r['latency_ms'][k] or 0) for k in ('parent_root_issuance', 'a2a_agent_card_discovery', 'a2a_message_send', 'complete_verification')) for r in rows)
        check(abs(ms - live['summary'][condition]['end_to_end_latency_ms']['mean']) < 0.00001, 'Live mean from raw trials')
        check(round(ms, 1) == expected, 'Live paper mean')
        live_means[condition] = round(ms, 1)
    check(live['security_attack']['linked_list']['child_key_only_claim'] == 'accepted', 'Linked compromise witness')
    check(live['security_attack']['signed_dag']['child_key_only_claim'] == 'rejected', 'Co-signed compromise witness')
    cloud = json.loads((current / 'e11_aws_20260831/rerun_01/results.json').read_text())
    for filename, field in [('experiments/e10_multiprocess_cross_deployer/run.py', 'e10_signer_source_sha256'), ('experiments/e11_container_cross_deployer/coordinator.py', 'e11_coordinator_source_sha256')]:
        check(sha(ROOT / filename) == cloud['environment'][field], 'Cloud runner provenance')
    check(len(cloud['trial_records']) == cloud['n_trials'] == cloud['valid_trials'] == 1000, 'Cloud trial count')
    check(all(r['verified'] for r in cloud['trial_records']), 'Cloud verification')
    for field, expected in [('cross_deployer_issue_ms', 3.651), ('complete_verify_ms', 5.015)]:
        ms = mean(r[field] for r in cloud['trial_records'])
        check(abs(ms - cloud[field]['mean']) < 0.00001 and round(ms, 3) == expected, 'Cloud raw trial mean')
    graph = json.loads((current / 'langgraph_20260830_064130.json').read_text())
    check(graph['n_trials'] == 1000 and graph['all_chains_verified'], 'LangGraph trial count')
    check(round(graph['total_sign_overhead_per_task_ms']['mean'], 3) == 0.512 and round(graph['full_chain_verify_ms']['mean'], 3) == 0.832, 'LangGraph means')
    with (current / 'perf_20260830_064113.csv').open() as f:
        perf = {r['design']: r for r in csv.DictReader(f)}
    for design, expected in [('linked_list', 24.3), ('merkle', 72.7), ('signed_dag', 499.2)]:
        check(int(perf[design]['trials']) == 1000 and int(perf[design]['depth']) == 5, 'Local performance configuration')
        check(round(float(perf[design]['per_hop_mean_us']), 1) == expected, 'Local performance mean')
    print(json.dumps({'status': 'passed', 'manifest_files': len(manifest['files']),
                      'snapshot_files': len(public), 'events': sum(kinds.values()),
                      'distinct_candidates': len(inputs[True]), 'sequences': len(sequences),
                      'within_limit_accepted': 2340, 'overflow_rejected': 1080,
                      'structural_checks_excluding_historical_B5': structural,
                      'live_mean_ms_from_raw_trials': live_means,
                      'cloud_trials': 1000, 'cloud_means_match_raw_trials': True,
                      'langgraph_and_local_perf_aggregates_match_paper': True}, indent=2))

if __name__ == '__main__':
    main()

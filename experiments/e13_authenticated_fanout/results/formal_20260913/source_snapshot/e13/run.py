"""Run and replay E13. Only explicit validation/policy rejections are outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import shutil
import sqlite3
import sys
import time
import traceback

import cryptography
from bootstrap import HERE, ROOT
from fixtures import Factory, NOW, request, unpack
from scenarios import composition_cases
from verifier import AdmissionRejected, BACKENDS, FanoutVerifier
from src.defenses.chain_designs.base import ChainVerifyError
from src.defenses.complete_chain_verifier import verify_complete_chain


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def evaluate(req, path=None):
    chain, outputs, registry = unpack(req)
    if path is None:
        try:
            verify_complete_chain(chain=chain, outputs=outputs, deployers=registry,
                                  backend=BACKENDS[req['design']](), now=NOW)
            return 'accepted'
        except ChainVerifyError:
            return 'cryptographic_validation'
    try:
        FanoutVerifier(path, initialize=False).verify(
            chain=chain, outputs=outputs, registry=registry, design=req['design'], now=NOW)
        return 'accepted'
    except AdmissionRejected as exc:
        return exc.code


def worker(req, database, ready, start, results):
    ready.put(os.getpid())
    start.wait(30)
    try:
        code = evaluate(req, database)
        results.put({'pid': os.getpid(), 'request_hash': digest(req), 'code': code})
    except BaseException:
        results.put({'pid': os.getpid(), 'error': traceback.format_exc()})


def racing(reqs, database):
    ctx = mp.get_context('spawn')
    ready, results, start = ctx.Queue(), ctx.Queue(), ctx.Event()
    jobs = [ctx.Process(target=worker, args=(req, str(database), ready, start, results))
            for req in reqs]
    try:
        for job in jobs:
            job.start()
        for _ in jobs:
            ready.get(timeout=30)
        start.set()
        rows = [results.get(timeout=30) for _ in jobs]
        for job in jobs:
            job.join(timeout=10)
            if job.exitcode != 0:
                raise RuntimeError(f'worker failed: {job.pid}, {job.exitcode}')
        if any('error' in row for row in rows):
            raise RuntimeError(rows)
        return rows
    finally:
        for job in jobs:
            if job.is_alive():
                job.terminate()
                job.join(timeout=5)
        ready.close()
        results.close()


def dirty_writer(database, ready):
    db = sqlite3.connect(database, isolation_level=None)
    db.execute('BEGIN IMMEDIATE')
    db.execute("INSERT INTO receipts VALUES ('uncommitted', 'crash-signer', 'crash-nonce')")
    db.execute("INSERT INTO edges VALUES ('crash-parent','uncommitted')")
    db.execute("INSERT INTO submissions VALUES ('uncommitted')")
    ready.set()
    time.sleep(60)


def killed_writer(database):
    ctx = mp.get_context('spawn')
    ready = ctx.Event()
    job = ctx.Process(target=dirty_writer, args=(str(database), ready))
    job.start()
    try:
        if not ready.wait(20):
            raise RuntimeError('writer did not stage transaction')
        job.terminate()
        job.join(timeout=10)
        if job.exitcode is None or job.exitcode == 0:
            raise RuntimeError('writer did not terminate as expected')
        return {'pid': job.pid, 'exitcode': job.exitcode, 'staged_tables': 3}
    finally:
        if job.is_alive():
            job.kill()
            job.join(timeout=5)


class Recorder:
    def __init__(self, output):
        self.output = output
        self.stores = {}
        self.file = (output / 'events.jsonl').open('x')
        self.counts = Counter()
        self.total = 0

    def store(self, name):
        if name not in self.stores:
            self.stores[name] = FanoutVerifier(self.output / 'state' / (name + '.sqlite'))
        return self.stores[name]

    def log(self, row):
        self.total += 1
        row['event_id'] = self.total
        self.file.write(json.dumps(row, sort_keys=True) + '\n')
        self.file.flush()
        arm = 'crypto_only' if row.get('control') else 'authenticated_fanout'
        self.counts[(row['block'], row['design'], row['kind'], arm,
                     row.get('actual', 'checked'))] += 1

    def call(self, store, scenario, factory, chain, expected, bad_output=False,
             *, block='composition', control=False):
        req = request(chain, factory)
        if bad_output:
            req['outputs_hex'][-1] = b'substituted'.hex()
        return self.call_wire(store, scenario, req, expected, block=block, control=control)

    def call_wire(self, store, scenario, req, expected, *, block, control=False):
        v = None if control else self.store(store)
        before = digest(v.snapshot()) if v else None
        actual = evaluate(req, v.database if v else None)
        after = digest(v.snapshot()) if v else None
        row = {'kind': 'serial', 'block': block, 'design': req['design'],
               'store': store, 'scenario': scenario, 'control': control,
               'request': req, 'request_hash': digest(req),
               'expected': expected, 'actual': actual,
               'state_before': before, 'state_after': after}
        self.log(row)
        if actual != expected:
            raise AssertionError(f'{scenario}: expected {expected}, got {actual}')
        if v and actual != 'accepted' and before != after:
            raise AssertionError(f'{scenario}: rejected request changed state')
        return actual

    def race(self, name, reqs, rejection, scenario):
        v = self.store(name)
        before = digest(v.snapshot())
        rows = racing(reqs, v.database)
        counts = Counter(row['code'] for row in rows)
        after = digest(v.snapshot())
        self.log({'kind': 'race', 'block': 'persistence_concurrency', 'design': reqs[0]['design'],
                  'store': name, 'scenario': scenario, 'requests': reqs, 'results': rows,
                  'expected_counts': {'accepted': 1, rejection: len(reqs)-1},
                  'actual_counts': dict(counts), 'state_before': before, 'state_after': after})
        assert counts == {'accepted': 1, rejection: len(reqs)-1}, rows
        return rows


def boundaries(rec, design, seed, repeats):
    rng = random.Random(seed)
    for limit in (0, 1, 5, 20):
        for repeat in range(repeats):
            f = Factory(design)
            root = f.node(instance='boundary-root', limit=limit)
            children = [f.node(root, instance=f'child-{i}') for i in range(limit+3)]
            order = list(range(len(children)))
            rng.shuffle(order)
            name = f'{design}-{seed}-limit{limit}-repeat{repeat}'
            for position, index in enumerate(order):
                chain = children[index]
                rec.call(name, 'boundary_control', f, chain, 'accepted', block='boundary', control=True)
                rec.call(name, 'boundary_policy', f, chain,
                         'accepted' if position < limit else 'fanout_limit', block='boundary')
            assert len(rec.store(name).snapshot()['edges']) == limit


def persistent_cases(rec, design, seed):
    f = Factory(design)
    root = f.node(instance='race-root', limit=2)
    prefix = f'{design}-{seed}-persistent'
    first = f.node(root, instance='first')
    rec.call(prefix, 'before_restart', f, first, 'accepted', block='persistence_concurrency')
    # evaluate opens a fresh database connection for every call. These calls
    # additionally run in fresh processes with only serialized public data.
    reqs = [request(f.node(root, instance=f'contender-{i}'), f) for i in range(4)]
    rows = rec.race(prefix, reqs, 'fanout_limit', 'last_slot_four_processes')
    assert len(rec.store(prefix).snapshot()['edges']) == 2
    replay_req = request(first, f)
    outcome = racing([replay_req], rec.store(prefix).database)
    assert outcome[0]['code'] == 'terminal_replay'
    rec.log({'kind': 'restart', 'block': 'persistence_concurrency', 'design': design,
             'store': prefix, 'scenario': 'fresh_process_replay', 'request': replay_req,
             'result': outcome[0], 'expected': 'terminal_replay'})

    duplicate_store = prefix + '-duplicate'
    duplicate = request(f.node(f.node(instance='duplicate-root', limit=5), instance='duplicate'), f)
    rec.race(duplicate_store, [duplicate]*4, 'terminal_replay', 'same_terminal_four_processes')
    assert len(rec.store(duplicate_store).snapshot()['edges']) == 1

    v = rec.store(prefix)
    before = digest(v.snapshot())
    crash = killed_writer(v.database)
    after = digest(FanoutVerifier(v.database).snapshot())
    rec.log({'kind': 'crash', 'block': 'persistence_concurrency', 'design': design,
             'store': prefix, 'scenario': 'killed_uncommitted_writer',
             'state_before': before, 'state_after': after, 'worker': crash})
    assert before == after
    rec.call(prefix, 'quota_after_crash', f, f.node(root, instance='post-crash-overflow'),
             'fanout_limit', block='persistence_concurrency')
    # Recovery must also permit a valid write in a different quota scope.
    fresh_root = f.node(instance='post-crash-root', limit=1)
    rec.call(prefix, 'valid_write_after_crash', f, f.node(fresh_root, instance='post-crash-valid'),
             'accepted', block='persistence_concurrency')


def freeze_sources(output):
    snapshot = output / 'source_snapshot'
    sources = [p for p in (ROOT / 'src').rglob('*.py') if '__pycache__' not in p.parts]
    local = [p for p in HERE.rglob('*') if p.is_file() and
             not any(s in p.relative_to(HERE).parts for s in ('results', '__pycache__', '.pytest_cache'))]
    records = {}
    for source in sources + local:
        rel = source.relative_to(ROOT) if source in sources else Path('e13') / source.relative_to(HERE)
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records[str(rel)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / 'source_hashes.json').write_text(json.dumps(records, indent=2))
    return records


def run(output, seeds, repeats):
    output.mkdir(parents=True, exist_ok=False)
    hashes = freeze_sources(output)
    config = {'schema': 'e13-authenticated-fanout-v1', 'seeds': seeds, 'repeats': repeats,
              'limits': [0, 1, 5, 20], 'designs': list(BACKENDS),
              'fixture_clock': NOW.isoformat(), 'python': sys.version,
              'platform': platform.platform(), 'sqlite': sqlite3.sqlite_version,
              'cryptography': cryptography.__version__, 'source_manifest_hash': digest(hashes),
              'command': sys.argv, 'pid': os.getpid(),
              'scope': 'Per immutable parent receipt, one local SQLite store; no global quota or malicious-child claim.'}
    (output / 'configuration.json').write_text(json.dumps(config, indent=2))
    rec = Recorder(output)
    start = time.monotonic()
    try:
        for design in BACKENDS:
            for seed in seeds:
                boundaries(rec, design, seed, repeats)
                def emit(store, scenario, factory, chain, expected, bad_output=False):
                    return rec.call(f'{design}-{seed}-{store}', scenario, factory,
                                    chain, expected, bad_output)
                composition_cases(design, emit)
                persistent_cases(rec, design, seed)
                print(f'completed {design} seed={seed}, events={rec.total}', flush=True)
        result = {'status': 'passed', 'unexpected_aborts': 0, 'expectation_mismatches': 0,
                  'elapsed_seconds': time.monotonic()-start, 'event_count': rec.total,
                  'counts': [{'block': k[0], 'design': k[1], 'kind': k[2], 'arm': k[3], 'outcome': k[4],
                              'count': v} for k, v in sorted(rec.counts.items())]}
    except BaseException:
        result = {'status': 'failed', 'event_count': rec.total, 'error': traceback.format_exc()}
        (output / 'summary.json').write_text(json.dumps(result, indent=2))
        raise
    finally:
        rec.file.close()
    result['events_sha256'] = hashlib.sha256((output / 'events.jsonl').read_bytes()).hexdigest()
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    return result


def replay(source, output):
    output.mkdir(parents=True, exist_ok=False)
    stores = {}
    counts = Counter()
    def store(name):
        if name not in stores:
            stores[name] = FanoutVerifier(output / (name + '.sqlite'))
        return stores[name]
    for line in (source / 'events.jsonl').open():
        row = json.loads(line)
        kind = row['kind']
        if kind == 'serial':
            v = None if row['control'] else store(row['store'])
            if v:
                assert digest(v.snapshot()) == row['state_before'], row['event_id']
            assert digest(row['request']) == row['request_hash']
            actual = evaluate(row['request'], v.database if v else None)
            assert actual == row['expected'] == row['actual'], row['event_id']
            if v:
                assert digest(v.snapshot()) == row['state_after'], row['event_id']
        elif kind == 'race':
            # Replay the observed linearization (winning commit first), not the
            # original OS schedule. Fresh-process races are measured by run().
            v = store(row['store'])
            assert digest(v.snapshot()) == row['state_before']
            reqs = {digest(req): req for req in row['requests']}
            for outcome in sorted(row['results'], key=lambda r: r['code'] != 'accepted'):
                assert evaluate(reqs[outcome['request_hash']], v.database) == outcome['code']
            assert digest(v.snapshot()) == row['state_after']
        elif kind == 'restart':
            assert evaluate(row['request'], store(row['store']).database) == row['expected']
        elif kind == 'crash':
            # Recorded crash is a no-op in the successful state history.
            assert digest(store(row['store']).snapshot()) == row['state_before'] == row['state_after']
        else:
            raise ValueError(kind)
        counts[kind] += 1
    report = {'status': 'passed', 'events': sum(counts.values()), 'counts': dict(counts),
              'source_events_sha256': hashlib.sha256((source / 'events.jsonl').read_bytes()).hexdigest(),
              'scope': 'Exact signed corpus and logical state replay; races linearized to recorded winners; crash records checked as unchanged committed state.'}
    (output / 'replay_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    execute = sub.add_parser('run')
    execute.add_argument('--out', type=Path, required=True)
    execute.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43])
    execute.add_argument('--repeats', type=int, default=10)
    repeat = sub.add_parser('replay')
    repeat.add_argument('--source', type=Path, required=True)
    repeat.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        if args.repeats < 1 or len(args.seeds) != len(set(args.seeds)):
            parser.error('repeats must be positive and seeds unique')
        run(args.out.resolve(), args.seeds, args.repeats)
    else:
        replay(args.source.resolve(), args.out.resolve())

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures import Factory, NOW, outputs
from scenarios import composition_cases
from verifier import AdmissionRejected, BACKENDS, FanoutVerifier
from fixtures import request
from run import racing, killed_writer


@pytest.mark.parametrize('design', BACKENDS)
@pytest.mark.parametrize('limit', [0, 1, 5, 20])
def test_signed_boundary_and_exact_state(tmp_path, design, limit):
    f = Factory(design)
    root = f.node(instance='root', limit=limit)
    v = FanoutVerifier(tmp_path / 'state.sqlite')
    for index in range(limit + 3):
        chain = f.node(root, instance=f'child-{index}')
        if index < limit:
            v.verify(chain=chain, outputs=outputs(chain), registry=f.registry, design=design, now=NOW)
        else:
            before = v.snapshot()
            with pytest.raises(AdmissionRejected) as error:
                v.verify(chain=chain, outputs=outputs(chain), registry=f.registry, design=design, now=NOW)
            assert error.value.code == 'fanout_limit'
            assert v.snapshot() == before
    assert len(v.snapshot()['edges']) == limit


@pytest.mark.parametrize('design', BACKENDS)
def test_adversarial_composition_sequences(tmp_path, design):
    verifiers = {}
    def emit(store, scenario, factory, chain, expected, bad_output=False):
        v = verifiers.setdefault(store, FanoutVerifier(tmp_path / (store + '.sqlite')))
        before = v.snapshot()
        data = outputs(chain)
        if bad_output:
            data[-1] = b'substituted'
        try:
            v.verify(chain=chain, outputs=data, registry=factory.registry, design=design, now=NOW)
            actual = 'accepted'
        except AdmissionRejected as error:
            actual = error.code
        assert actual == expected, scenario
        if actual != 'accepted':
            assert v.snapshot() == before, scenario
    composition_cases(design, emit)
    assert len(verifiers['shared'].snapshot()['edges']) == 3
    assert len(verifiers['rollback'].snapshot()['edges']) == 1
    assert len(verifiers['nonce'].snapshot()['edges']) == 2


@pytest.mark.parametrize('design', BACKENDS)
def test_reopen_store_and_external_leaf_target(tmp_path, design):
    f = Factory(design)
    root = f.node(instance='root', limit=1)
    chain = f.node(root, instance='child')
    path = tmp_path / 'persist.sqlite'
    v = FanoutVerifier(path)
    with pytest.raises(AdmissionRejected) as error:
        v.verify(chain=chain, outputs=outputs(chain), registry=f.registry, design=design,
                 now=NOW, expected_leaf='wrong-target')
    assert error.value.code == 'expected_leaf'
    assert v.snapshot() == {'receipts': [], 'edges': [], 'submissions': []}
    result = v.verify(chain=chain, outputs=outputs(chain), registry=f.registry, design=design, now=NOW)
    assert result['new_edges'] == 1
    reopened = FanoutVerifier(path)
    with pytest.raises(AdmissionRejected) as error:
        reopened.verify(chain=chain, outputs=outputs(chain), registry=f.registry, design=design, now=NOW)
    assert error.value.code == 'terminal_replay'
    assert reopened.snapshot() == v.snapshot()


@pytest.mark.parametrize('design', BACKENDS)
def test_process_races_and_killed_transaction(tmp_path, design):
    f = Factory(design)
    root = f.node(instance='root', limit=1)
    v = FanoutVerifier(tmp_path / 'race.sqlite')
    requests = [request(f.node(root, instance=f'child-{i}'), f) for i in range(4)]
    rows = racing(requests, v.database)
    assert sorted(row['code'] for row in rows) == ['accepted'] + ['fanout_limit']*3
    assert len(v.snapshot()['edges']) == 1
    before = v.snapshot()
    killed_writer(v.database)
    assert FanoutVerifier(v.database).snapshot() == before
    other = FanoutVerifier(tmp_path / 'duplicates.sqlite')
    rows = racing([requests[0]]*4, other.database)
    assert sorted(row['code'] for row in rows) == ['accepted'] + ['terminal_replay']*3
    assert len(other.snapshot()['edges']) == 1

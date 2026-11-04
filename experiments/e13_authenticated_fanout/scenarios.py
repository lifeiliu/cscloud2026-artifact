"""Predeclared adversarial sequences, shared by the experiment and smoke checks."""
from dataclasses import replace

from fixtures import Factory
from src.defenses.chain_designs.base import ChainNode


def composition_cases(design, emit):
    """emit(store_name, scenario, factory, chain, expected_code, bad_output)."""
    f = Factory(design)
    root = f.node(instance='root', limit=1)
    child = f.node(root, instance='child')
    # Real signatures are generated before each adversarial edit.
    bad_parent = ChainNode(replace(root[0].signed,
        payload=replace(root[0].signed.payload, max_children=100)))
    emit('auth', 'tampered_limit', f, [bad_parent, child[-1]], 'cryptographic_validation')
    bad_sig = ChainNode(replace(child[-1].signed, signature_hex='00' * 64))
    emit('auth', 'bad_primary', f, root + [bad_sig], 'cryptographic_validation')
    emit('auth', 'output_substitution', f, child, 'cryptographic_validation', True)
    bad_edge = f.resign(child[-1], parent_id='not-the-parent')
    emit('auth', 'signed_bad_ancestry', f, root + [bad_edge], 'cryptographic_validation')
    emit('auth', 'valid_after_bad_inputs', f, child, 'accepted')
    emit('auth', 'terminal_replay', f, child, 'terminal_replay')
    extra = f.node(root, instance='extra')
    emit('auth', 'overflow_after_bad_inputs', f, extra, 'fanout_limit')

    # Reject late in a multi-hop admission after staging earlier valid edges.
    f = Factory(design)
    root = f.node(instance='rollback-root', limit=1)
    middle = f.node(root, instance='middle', limit=0)
    leaf = f.node(middle, instance='blocked-grandchild', host='C')
    emit('rollback', 'late_multihop_failure', f, leaf, 'fanout_limit')
    emit('rollback', 'valid_after_rollback', f, middle, 'accepted')

    # A submitted leaf later serves as historical ancestry. The shared edge is
    # not charged again, and its nonce remains bound to the exact same receipt.
    f = Factory(design)
    root = f.node(instance='shared-root', limit=1)
    middle = f.node(root, instance='shared-middle', limit=2)
    emit('shared', 'submit_parent_first', f, middle, 'accepted')
    left = f.node(middle, instance='grandchild-left', host='C')
    right = f.node(middle, instance='grandchild-right', host='C')
    emit('shared', 'reuse_submitted_parent', f, left, 'accepted')
    emit('shared', 'reuse_shared_edge', f, right, 'accepted')
    third = f.node(middle, instance='grandchild-overflow', host='C')
    emit('shared', 'shared_parent_limit', f, third, 'fanout_limit')
    sibling = f.node(root, instance='root-overflow')
    emit('shared', 'root_limit_preserved', f, sibling, 'fanout_limit')

    f = Factory(design)
    root = f.node(instance='nonce-root', limit=5)
    first = f.node(root, instance='nonce-first', nonce='11' * 16)
    conflict = f.node(root, instance='nonce-conflict', nonce='11' * 16)
    emit('nonce', 'nonce_first', f, first, 'accepted')
    emit('nonce', 'signed_nonce_conflict', f, conflict, 'nonce_conflict')
    emit('nonce', 'valid_after_nonce_conflict', f, f.node(root, instance='nonce-fresh'), 'accepted')

    # instance_id is not a global quota namespace.
    f = Factory(design)
    root_a = f.node(instance='same-id', limit=1, host='A')
    root_c = f.node(instance='same-id', limit=1, host='C')
    emit('isolation', 'parent_A', f, f.node(root_a, instance='same-child-id'), 'accepted')
    emit('isolation', 'parent_C', f, f.node(root_c, instance='same-child-id'), 'accepted')
    emit('isolation', 'parent_A_overflow', f, f.node(root_a, instance='other-child'), 'fanout_limit')
    # Same signer and instance ID, new nonce: a distinct immutable receipt.
    root_a_new = f.node(instance='same-id', limit=1, host='A')
    emit('isolation', 'reissued_parent_separate_scope', f,
         f.node(root_a_new, instance='new-epoch-child'), 'accepted')

    for index, limit in enumerate([-1, True, 1.5, '2']):
        f = Factory(design)
        root = f.node(instance=f'invalid-policy-{index}', limit=limit)
        emit(f'policy-{index}', 'signed_invalid_policy', f,
             f.node(root, instance='child'), 'invalid_policy')
    f = Factory(design)
    root = f.node(instance='unlimited', limit=None)
    for index in range(3):
        emit('unlimited', 'null_limit', f, f.node(root, instance=f'child-{index}'), 'accepted')

    # Ancestors are revalidated even after their state has been recorded.
    f = Factory(design)
    root = f.node(instance='revoke-root', limit=2)
    emit('revocation', 'before_revoke', f, f.node(root, instance='before'), 'accepted')
    f.registry.revoke_key(root[0].signed.key_id)
    emit('revocation', 'revoked_reused_ancestor', f, f.node(root, instance='after'),
         'cryptographic_validation')

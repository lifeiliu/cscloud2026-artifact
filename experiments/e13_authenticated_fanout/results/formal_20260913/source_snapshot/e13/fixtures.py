"""Real Ed25519-signed fixtures; private keys never enter exported requests."""
from dataclasses import replace
from datetime import datetime, timezone

import bootstrap  # noqa: F401
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from src.core.agent_id_v2 import (BoundIDPayloadV2, ChainProof, LifecycleRecord,
                                 SignedBoundIDV2, make_lifecycle_new)
from src.defenses.chain_designs.base import ChainNode, DeployerRegistry
from src.runtime.multi_deployer import MultiDeployerHarness
from verifier import BACKENDS

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)


class Factory:
    def __init__(self, design):
        self.design = design
        self.backend = BACKENDS[design]()
        self.harness = MultiDeployerHarness.with_deployers(['A', 'B', 'C'])
        self.registry = DeployerRegistry()
        self.public_keys = {}
        for name in self.harness.names():
            key = self.harness.get_deployer(name).keypair
            self.registry.register_public(name, key.public)
            self.public_keys[name] = key.public_bytes.hex()

    def node(self, chain=(), *, instance='node', limit=None, host=None,
             nonce=None, output=None):
        host = host or ('A' if not chain else 'B')
        issuer = self.harness.get_deployer(host)
        output = output if output is not None else instance.encode()
        parent = chain[-1].signed.payload if chain else None
        names = tuple(dict.fromkeys((parent.deployer, host))) if parent else ()
        if not parent:
            proof = self.backend.build_proof_for_root()
        elif self.design == 'signed_dag':
            proof = ChainProof(design=self.design, co_signers=names)
        else:
            proof = self.backend.build_proof_for_extend(
                parent=parent, child_instance_id=instance,
                ancestors=[n.signed.payload for n in chain],
                deployers=self.harness.registry)
        payload = issuer.build_payload(instance_id=instance, system_id='e13',
                    output_bytes=output, binding_scope='per_tool_call',
                    chain_proof=proof, lifecycle=make_lifecycle_new(), now=NOW,
                    parent_id=parent.instance_id if parent else None,
                    ancestry_depth=len(chain), nonce=nonce)
        payload = replace(payload, max_children=limit)
        if parent and self.design == 'signed_dag':
            proof = self.backend.build_proof_for_extend(
                parent=parent, child_instance_id=instance,
                ancestors=[n.signed.payload for n in chain],
                deployers=self.harness.registry, co_signer_names=names,
                child_payload=payload)
            payload = replace(payload, chain_proof=proof)
        return list(chain) + [ChainNode(issuer.sign_payload(payload))]

    def resign(self, node, **changes):
        payload = replace(node.signed.payload, **changes)
        issuer = self.harness.get_deployer(payload.deployer)
        return ChainNode(issuer.sign_payload(payload))


def outputs(chain):
    return [n.signed.payload.instance_id.encode() for n in chain]


def request(chain, factory):
    return {'chain': [n.signed.to_wire() for n in chain],
            'outputs_hex': [b.hex() for b in outputs(chain)],
            'public_keys': factory.public_keys, 'design': factory.design,
            'revoked_key_ids': sorted(factory.registry._revoked_key_ids)}


def unpack(req):
    chain = []
    for item in req['chain']:
        payload = dict(item['payload'])
        proof = dict(payload['chain_proof'])
        for field in ('co_signers', 'signatures'):
            proof[field] = tuple(proof[field])
        payload['chain_proof'] = ChainProof(**proof)
        lifecycle = dict(payload['lifecycle'])
        lifecycle['constituents'] = tuple(lifecycle['constituents'])
        payload['lifecycle'] = LifecycleRecord(**lifecycle)
        payload['certifications'] = tuple(payload['certifications'])
        chain.append(ChainNode(SignedBoundIDV2(BoundIDPayloadV2(**payload),
                         item['signature_hex'], item['key_id'])))
    registry = DeployerRegistry()
    for name, key in req['public_keys'].items():
        registry.register_public(name, Ed25519PublicKey.from_public_bytes(bytes.fromhex(key)))
    for key in req.get('revoked_key_ids', []):
        registry.revoke_key(key)
    return chain, [bytes.fromhex(b) for b in req['outputs_hex']], registry

"""E13: authenticated, transactional per-parent-receipt fanout admission.

All callers use verify(); the original cryptographic verifier is mandatory.
State is shared by processes using ONE local SQLite file. This does not enforce
global quotas, action authorization, or limits across parent receipt reissuance.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import bootstrap  # noqa: F401
from cryptography.hazmat.primitives import serialization
from src.defenses.chain_designs import LinkedListBackend, MerkleBackend, SignedDAGBackend
from src.defenses.chain_designs.base import ChainVerifyError
from src.defenses.complete_chain_verifier import verify_complete_chain

BACKENDS = {'linked_list': LinkedListBackend, 'merkle': MerkleBackend,
            'signed_dag': SignedDAGBackend}


class AdmissionRejected(ChainVerifyError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(detail)


def receipt_identity(node, registry):
    """Bind a receipt and its nonce namespace to the full verified public key."""
    payload = node.signed.payload
    public = registry.get_public(payload.deployer)
    raw = public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    signer = hashlib.sha256(raw).hexdigest()
    identity = hashlib.sha256(
        b'e13-receipt-v1\x00' + bytes.fromhex(signer) + payload.to_canonical_json()
    ).hexdigest()
    return identity, signer, payload.nonce


class FanoutVerifier:
    def __init__(self, database: str | Path, *, initialize: bool = True):
        self.database = str(database)
        if self.database == ':memory:':
            raise ValueError('E13 requires a file-backed store')
        if initialize:
            Path(self.database).parent.mkdir(parents=True, exist_ok=True)
            db = self._connect()
            try:
                db.execute('PRAGMA journal_mode=WAL')
                db.executescript('''
                    CREATE TABLE IF NOT EXISTS meta(version INTEGER NOT NULL);
                    INSERT INTO meta SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM meta);
                    CREATE TABLE IF NOT EXISTS receipts(
                        receipt TEXT PRIMARY KEY, signer TEXT NOT NULL,
                        nonce TEXT NOT NULL, UNIQUE(signer, nonce));
                    CREATE TABLE IF NOT EXISTS edges(
                        parent TEXT NOT NULL, child TEXT NOT NULL,
                        PRIMARY KEY(parent, child));
                    CREATE TABLE IF NOT EXISTS submissions(leaf TEXT PRIMARY KEY);
                ''')
                if db.execute('SELECT version FROM meta').fetchall() != [(1,)]:
                    raise ValueError('unsupported E13 store version')
            finally:
                db.close()

    def _connect(self):
        db = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        db.execute('PRAGMA synchronous=FULL')
        return db

    def snapshot(self):
        db = self._connect()
        try:
            db.execute('BEGIN')
            result = {name: db.execute(f'SELECT * FROM {name} ORDER BY 1, 2' if
                       name != 'submissions' else 'SELECT * FROM submissions ORDER BY 1').fetchall()
                      for name in ('receipts', 'edges', 'submissions')}
            db.commit()
            return result
        finally:
            db.close()

    def verify(self, *, chain, outputs, registry, design: str,
               now: datetime | None = None, expected_leaf: str | None = None):
        # Historical ancestors are re-authenticated on EVERY presentation. The
        # legacy all-record single-use NonceStore is deliberately not passed:
        # E13 transactionally distinguishes receipt reuse from terminal replay.
        try:
            verify_complete_chain(chain=chain, outputs=outputs,
                                  backend=BACKENDS[design](), deployers=registry,
                                  now=now, nonces=None)
        except ChainVerifyError as exc:
            raise AdmissionRejected('cryptographic_validation', str(exc)) from exc
        identities = [receipt_identity(node, registry) for node in chain]
        ids = [item[0] for item in identities]
        if expected_leaf is not None and ids[-1] != expected_leaf:
            raise AdmissionRejected('expected_leaf', 'terminal receipt does not match caller target')
        for node in chain:
            limit = node.signed.payload.max_children
            if limit is not None and (type(limit) is not int or limit < 0):
                raise AdmissionRejected('invalid_policy', 'max_children must be null or a nonnegative integer')

        db = self._connect()
        try:
            # Cross-process writer serialization includes all checks and writes.
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM submissions WHERE leaf=?', (ids[-1],)).fetchone():
                raise AdmissionRejected('terminal_replay', 'terminal receipt already submitted')
            for identity, signer, nonce in identities:
                prior = db.execute('SELECT receipt FROM receipts WHERE signer=? AND nonce=?',
                                   (signer, nonce)).fetchone()
                if prior is not None and prior[0] != identity:
                    raise AdmissionRejected('nonce_conflict', 'signer reused nonce for another receipt')
                db.execute('INSERT OR IGNORE INTO receipts VALUES (?,?,?)',
                           (identity, signer, nonce))
            added = 0
            for index in range(1, len(chain)):
                parent, child = ids[index-1:index+1]
                if db.execute('SELECT 1 FROM edges WHERE parent=? AND child=?',
                              (parent, child)).fetchone():
                    continue
                limit = chain[index-1].signed.payload.max_children
                count = db.execute('SELECT COUNT(*) FROM edges WHERE parent=?',
                                   (parent,)).fetchone()[0]
                if limit is not None and count >= limit:
                    raise AdmissionRejected('fanout_limit', 'new child would exceed authenticated parent limit')
                db.execute('INSERT INTO edges VALUES (?,?)', (parent, child))
                added += 1
            db.execute('INSERT INTO submissions VALUES (?)', (ids[-1],))
            db.commit()
            return {'accepted': True, 'new_edges': added, 'leaf': ids[-1]}
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

"""Run one E11 signer behind an authenticated TCP listener.

The service deliberately reuses E10's signer operation handler.  Its private
Ed25519 key is generated in, and remains in, this container; the coordinator
receives only public metadata, receipts, and edge signatures.  The underlying
``multiprocessing.connection`` channel authenticates peers with an HMAC
challenge but is not a TLS transport.  It is therefore suitable for the
closed Docker bridge-network experiment only, not a production deployment.
"""

from __future__ import annotations

import argparse
import os
from multiprocessing.connection import Listener

from experiments.e10_multiprocess_cross_deployer.run import _signer_worker


def _authkey() -> bytes:
    value = os.environ.get("E11_RPC_AUTHKEY")
    if not value:
        raise RuntimeError("E11_RPC_AUTHKEY must be set")
    return value.encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployer", required=True)
    parser.add_argument("--port", type=int, default=6000)
    args = parser.parse_args()

    listener = Listener(("0.0.0.0", args.port), family="AF_INET", authkey=_authkey())
    try:
        connection = listener.accept()
        _signer_worker(args.deployer, connection)
    finally:
        listener.close()


if __name__ == "__main__":
    main()

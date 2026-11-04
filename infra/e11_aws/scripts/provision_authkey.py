"""Generate the non-exported E11 RPC authenticator in SSM Parameter Store.

The value is generated locally and sent directly to SecureString storage. It is
never written to Terraform variables, state, shell history, or this script's
output. EC2 nodes retrieve it through their instance role at bootstrap.
"""

from __future__ import annotations

import argparse
import secrets

import boto3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--parameter", required=True)
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    session.client("ssm").put_parameter(
        Name=args.parameter,
        Value=secrets.token_urlsafe(48),
        Type="SecureString",
        Overwrite=True,
        Tier="Standard",
    )
    print(f"Provisioned a fresh E11 RPC authenticator in {args.parameter}.")


if __name__ == "__main__":
    main()

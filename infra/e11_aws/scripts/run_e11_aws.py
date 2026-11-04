"""Run the AWS E11 coordinator through SSM and retrieve its raw result.

This runner never exposes the RPC credential. The coordinator reads it with
its EC2 role, writes the full result to a private S3 bucket, and prints only a
small JSON summary through SSM.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def _wait_online(ssm, instance_ids: list[str], timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    expected = set(instance_ids)
    while time.monotonic() < deadline:
        response = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": instance_ids}]
        )
        online = {
            item["InstanceId"]
            for item in response.get("InstanceInformationList", [])
            if item.get("PingStatus") == "Online"
        }
        if online == expected:
            return
        time.sleep(10)
    raise TimeoutError("not all E11 EC2 instances became SSM Online")


def _wait_command(ssm, command_id: str, instance_id: str, timeout_s: int) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            result = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        except ssm.exceptions.InvocationDoesNotExist:
            time.sleep(3)
            continue
        status = result["Status"]
        if status == "Success":
            return result
        if status in {"Cancelled", "TimedOut", "Failed", "Cancelling"}:
            raise RuntimeError(
                f"SSM command {command_id} ended {status}: {result.get('StandardErrorContent', '')}"
            )
        time.sleep(3)
    raise TimeoutError(f"SSM command {command_id} did not finish")


def _wait_signer_ready(ssm, instance_id: str, signer: str, timeout_s: int = 900) -> None:
    """Wait on the signer host itself, without consuming its one RPC connection."""
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={
            "commands": [
                "set -euo pipefail",
                "for i in {1..120}; do",
                "  state=$(docker inspect --format '{{.State.Running}}' e11-signer 2>/dev/null || true)",
                "  if [ \"$state\" = true ]; then exit 0; fi",
                "  sleep 5",
                "done",
                f"echo '{signer} signer did not become ready' >&2",
                "exit 1",
            ]
        },
        TimeoutSeconds=timeout_s,
        Comment=f"Wait for CSCloud E11 {signer} signer readiness",
    )
    _wait_command(ssm, response["Command"]["CommandId"], instance_id, timeout_s + 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--topology-json", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--auth-parameter", required=True)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    topology = json.loads(args.topology_json.read_text(encoding="utf-8"))
    hosts = topology["hosts"]
    parent = hosts["parent"]
    child = hosts["child"]
    coordinator = hosts["coordinator"]
    run_id = f"e11-aws-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    key = f"runs/{run_id}/results.json"

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    ssm = session.client("ssm")
    s3 = session.client("s3")
    _wait_online(ssm, [parent["instance_id"], child["instance_id"], coordinator["instance_id"]], 900)
    _wait_signer_ready(ssm, parent["instance_id"], "parent")
    _wait_signer_ready(ssm, child["instance_id"], "child")

    command = "\n".join(
        [
            "set -euo pipefail",
            "auth=$(aws ssm get-parameter --name '" + args.auth_parameter + "' --with-decryption --region '" + args.region + "' --query Parameter.Value --output text)",
            "runner_uid=$(docker run --rm '" + args.image + "' id -u)",
            "runner_gid=$(docker run --rm '" + args.image + "' id -g)",
            "install -d -o \"$runner_uid\" -g \"$runner_gid\" -m 0750 /var/lib/e11",
            "docker run --rm --network host -v /var/lib/e11:/results \\",
            "  -e E11_RPC_AUTHKEY=\"$auth\" \\",
            "  -e E11_PARENT_HOST='" + parent["private_ip"] + "' \\",
            "  -e E11_CHILD_HOST='" + child["private_ip"] + "' \\",
            "  -e E11_TRIALS='" + str(args.trials) + "' \\",
            "  -e E11_WARMUP='" + str(args.warmup) + "' \\",
            "  '" + args.image + "' python -m experiments.e11_container_cross_deployer.coordinator --out /results/results.json",
            "aws s3 cp /var/lib/e11/results.json 's3://" + args.bucket + "/" + key + "' --region '" + args.region + "'",
            "python3 - <<'PY'",
            "import json",
            "r=json.load(open('/var/lib/e11/results.json'))",
            "print(json.dumps({'valid_trials':r['valid_trials'],'aborted_trials':r['aborted_trials'],'issue_ms':r['cross_deployer_issue_ms'],'verify_ms':r['complete_verify_ms'],'faults':r['faults']},sort_keys=True))",
            "PY",
        ]
    )
    response = ssm.send_command(
        InstanceIds=[coordinator["instance_id"]],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [command]},
        TimeoutSeconds=3600,
        Comment=f"CSCloud E11 AWS formal run {run_id}",
    )
    invocation = _wait_command(ssm, response["Command"]["CommandId"], coordinator["instance_id"], 3900)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    s3.download_file(args.bucket, key, str(result_path))
    (args.output_dir / "ssm_summary.json").write_text(
        invocation.get("StandardOutputContent", ""), encoding="utf-8"
    )
    manifest = {
        "run_id": run_id,
        "bucket": args.bucket,
        "object_key": key,
        "coordinator_instance_id": coordinator["instance_id"],
        "ssm_command_id": response["Command"]["CommandId"],
        "runtime_image": args.image,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

# E11-AWS: three-host cloud deployment

This independent Terraform stack deploys the existing E11 signer protocol on
three EC2 hosts in three Availability Zones: parent signer, child signer, and
public-only verifier/coordinator. It reuses the Paper B guardrails but has its
own VPC, IAM role, ECR repository, evidence bucket, SSM parameter, Terraform
state, and tags.

## Boundaries

- No SSH key or port 22 is created. SSM is the only administration channel.
- Experimental RPC uses private IPs. Signer ingress on TCP/6000 is restricted
  to the coordinator security group; no signer port is public.
- Instances use public subnets only for outbound SSM/ECR/S3 access and have no
  security-group ingress from the Internet.
- The authenticated TCP channel is the current E11 transport, not TLS/mTLS or
  KMS. Those remain explicit next-step deployment enhancements.
- `enable_compute=false` is the checked-in default. Never apply an unchecked
  compute plan or a mutable image tag.

## Deployment phases

1. Authenticate a dedicated AWS CLI profile with permission to manage the
   isolated E11 ECR, EC2, IAM, S3, SSM, and VPC resources. The stack does not
   depend on the historical Paper B profile or account assignment.
2. Copy `terraform.tfvars.example` to ignored `terraform.tfvars`, set a short
   UTC expiry, then apply the reviewed bootstrap plan with compute disabled.
3. Build and push the runtime image by digest, provision a fresh SSM
   `SecureString` RPC authenticator, and generate a compute-enabled plan.
4. Apply that exact plan and invoke `scripts/run_e11_aws.py`; it waits for SSM,
   runs the coordinator, stores raw results in the private evidence bucket,
   and downloads a local evidence copy.
5. Review artifacts, then run `terraform destroy` before the tagged expiry.

## Safe post-run posture

The formal run leaves the three EC2 nodes active for inspection; the `Expiry`
tag is an audit deadline, not an automatic deletion mechanism. To remove only
the compute nodes while preserving the ECR image and private evidence bucket,
first review and then apply a no-compute plan:

```bash
AWS_PROFILE=<profile> terraform plan -var='enable_compute=false' \
  -var='runtime_image=' -out=stop_compute.tfplan
AWS_PROFILE=<profile> terraform apply stop_compute.tfplan
```

The ECR repository and versioned evidence bucket have destructive deletion
disabled. Archive or deliberately remove their contents before attempting a
full stack destroy.

## Local runner dependency

The runner and credential provisioner use an isolated local virtual
environment; the AWS CLI profile remains outside this repository:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Invoke `provision_authkey.py` and `run_e11_aws.py` with `.venv/bin/python`.

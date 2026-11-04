#!/usr/bin/env bash
set -euo pipefail

: "${AWS_PROFILE:?Set AWS_PROFILE to the IAM Identity Center profile}"
: "${AWS_REGION:=us-east-1}"
: "${IMAGE_TAG:?Set an immutable E11 runtime image tag}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
stack_dir="${repo_root}/infra/e11_aws"
repository="$(terraform -chdir="${stack_dir}" output -raw ecr_repository_url)"
registry="${repository%%/*}"

aws ecr get-login-password --profile "${AWS_PROFILE}" --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${registry}"

docker buildx build \
  --platform linux/amd64 \
  --file "${stack_dir}/images/runtime/Dockerfile" \
  --tag "${repository}:${IMAGE_TAG}" \
  --label "org.opencontainers.image.revision=$(git -C "${repo_root}" rev-parse HEAD)" \
  --provenance=true \
  --sbom=true \
  --push \
  "${repo_root}"

digest="$(aws ecr describe-images \
  --profile "${AWS_PROFILE}" \
  --region "${AWS_REGION}" \
  --repository-name "cscloud-e11-aws/runtime" \
  --image-ids "imageTag=${IMAGE_TAG}" \
  --query 'imageDetails[0].imageDigest' \
  --output text)"
printf '%s@%s\n' "${repository}" "${digest}"

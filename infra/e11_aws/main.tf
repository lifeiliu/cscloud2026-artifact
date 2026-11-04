data "aws_caller_identity" "current" {}

data "aws_ami" "al2023_x86_64" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }
}

data "aws_availability_zone" "selected" {
  for_each = toset(var.availability_zones)
  name     = each.value
}

locals {
  project = "cscloud-e11-aws"
  common_tags = {
    Project = local.project
    Owner   = var.owner
    Expiry  = var.expiry
    Purpose = "CSCloud E11 cloud experiment"
  }
  hosts = {
    parent      = { az_key = "a", role = "parent-signer", deployer = "platform_A" }
    child       = { az_key = "b", role = "child-signer", deployer = "platform_B" }
    coordinator = { az_key = "c", role = "public-verifier-coordinator", deployer = "" }
  }
  azs = { a = var.availability_zones[0], b = var.availability_zones[1], c = var.availability_zones[2] }
  subnets = {
    a = { az = local.azs.a, cidr = "10.81.1.0/24" }
    b = { az = local.azs.b, cidr = "10.81.2.0/24" }
    c = { az = local.azs.c, cidr = "10.81.3.0/24" }
  }
  active_hosts = var.enable_compute ? local.hosts : {}
}

resource "aws_ecr_repository" "runtime" {
  name                 = "cscloud-e11-aws/runtime"
  image_tag_mutability = "IMMUTABLE"
  # Do not silently delete experimental images during a Terraform destroy.
  force_delete = false
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_s3_bucket" "evidence" {
  bucket_prefix = "cscloud-e11-aws-evidence-"
  # Evidence must be explicitly archived or removed before the bucket can go.
  force_destroy = false
}

resource "aws_s3_bucket_public_access_block" "evidence" {
  bucket                  = aws_s3_bucket.evidence.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "evidence" {
  bucket = aws_s3_bucket.evidence.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "evidence" {
  bucket = aws_s3_bucket.evidence.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_ssm_parameter" "authkey_placeholder" {
  name  = "/cscloud/e11/aws/rpc-authkey"
  type  = "SecureString"
  value = "REPLACE_WITH_PROVISIONED_RANDOM_VALUE"

  lifecycle {
    ignore_changes = [value]
  }
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "node" {
  name               = "cscloud-e11-aws-node"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "node_runtime" {
  statement {
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    actions   = ["ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"]
    resources = [aws_ecr_repository.runtime.arn]
  }
  statement {
    actions   = ["ssm:GetParameter"]
    resources = [aws_ssm_parameter.authkey_placeholder.arn]
  }
  statement {
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.evidence.arn}/runs/*"]
  }
}

resource "aws_iam_role_policy" "node_runtime" {
  name   = "cscloud-e11-aws-runtime"
  role   = aws_iam_role.node.id
  policy = data.aws_iam_policy_document.node_runtime.json
}

resource "aws_iam_instance_profile" "node" {
  name = "cscloud-e11-aws-node"
  role = aws_iam_role.node.name
}

resource "aws_vpc" "e11" {
  cidr_block           = "10.81.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
}

resource "aws_internet_gateway" "e11" {
  vpc_id = aws_vpc.e11.id
}

resource "aws_subnet" "e11" {
  for_each                = local.subnets
  vpc_id                  = aws_vpc.e11.id
  availability_zone       = each.value.az
  cidr_block              = each.value.cidr
  map_public_ip_on_launch = true
  tags                    = { Name = "cscloud-e11-aws-${each.key}", ZoneId = data.aws_availability_zone.selected[each.value.az].zone_id }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.e11.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.e11.id
  }
}

resource "aws_route_table_association" "public" {
  for_each       = aws_subnet.e11
  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "signer" {
  name        = "cscloud-e11-aws-signer"
  description = "E11 signer: only coordinator may initiate experimental RPC"
  vpc_id      = aws_vpc.e11.id
}

resource "aws_security_group" "coordinator" {
  name        = "cscloud-e11-aws-coordinator"
  description = "E11 public-only verifier coordinator; no experimental ingress"
  vpc_id      = aws_vpc.e11.id
}

resource "aws_vpc_security_group_egress_rule" "signer_https" {
  security_group_id = aws_security_group.signer.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
  description       = "SSM, ECR, S3, and AWS API access"
}

resource "aws_vpc_security_group_egress_rule" "coordinator_https" {
  security_group_id = aws_security_group.coordinator.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
  description       = "SSM, ECR, S3, and AWS API access"
}

resource "aws_vpc_security_group_egress_rule" "coordinator_to_signer" {
  security_group_id            = aws_security_group.coordinator.id
  referenced_security_group_id = aws_security_group.signer.id
  from_port                    = 6000
  to_port                      = 6000
  ip_protocol                  = "tcp"
  description                  = "Authenticated E11 signer RPC"
}

resource "aws_vpc_security_group_ingress_rule" "signer_from_coordinator" {
  security_group_id            = aws_security_group.signer.id
  referenced_security_group_id = aws_security_group.coordinator.id
  from_port                    = 6000
  to_port                      = 6000
  ip_protocol                  = "tcp"
  description                  = "Authenticated E11 signer RPC"
}

resource "aws_instance" "e11" {
  for_each = local.active_hosts

  ami                                  = data.aws_ami.al2023_x86_64.id
  instance_type                        = var.instance_type
  subnet_id                            = aws_subnet.e11[each.value.az_key].id
  vpc_security_group_ids               = [each.key == "coordinator" ? aws_security_group.coordinator.id : aws_security_group.signer.id]
  iam_instance_profile                 = aws_iam_instance_profile.node.name
  associate_public_ip_address          = true
  instance_initiated_shutdown_behavior = "terminate"
  user_data_replace_on_change          = true
  user_data = templatefile("${path.module}/cloud-init.yaml.tftpl", {
    host_name      = "e11-${each.key}"
    host_role      = each.value.role
    deployer       = each.value.deployer
    runtime_image  = var.runtime_image
    auth_parameter = aws_ssm_parameter.authkey_placeholder.name
    aws_region     = var.aws_region
  })

  lifecycle {
    precondition {
      condition     = var.runtime_image != ""
      error_message = "enable_compute=true requires a pinned runtime_image."
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }
  credit_specification {
    cpu_credits = "standard"
  }
  root_block_device {
    volume_type           = "gp3"
    volume_size           = 20
    encrypted             = true
    delete_on_termination = true
  }
  tags = { Name = "cscloud-e11-aws-${each.key}", Role = each.value.role }
}

output "ecr_repository_url" { value = aws_ecr_repository.runtime.repository_url }
output "evidence_bucket" { value = aws_s3_bucket.evidence.id }
output "auth_parameter_name" { value = aws_ssm_parameter.authkey_placeholder.name }
output "compute_enabled" { value = var.enable_compute }
output "topology" {
  value = {
    region = var.aws_region
    hosts = {
      for name, instance in aws_instance.e11 : name => {
        instance_id       = instance.id
        private_ip        = instance.private_ip
        availability_zone = instance.availability_zone
        role              = local.hosts[name].role
      }
    }
  }
}

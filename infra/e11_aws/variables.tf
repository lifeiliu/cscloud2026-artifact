variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "owner" {
  description = "Resource owner, recorded as an AWS tag."
  type        = string
}

variable "expiry" {
  description = "Mandatory UTC cleanup deadline, recorded on every resource."
  type        = string
}

variable "enable_compute" {
  description = "Explicit gate. False creates no EC2 instances."
  type        = bool
  default     = false
}

variable "runtime_image" {
  description = "E11 image reference pinned by sha256 digest. Required only when compute is enabled."
  type        = string
  default     = ""

  validation {
    condition = (
      var.runtime_image == "" ||
      can(regex("^[^[:space:]]+@sha256:[0-9a-f]{64}$", var.runtime_image))
    )
    error_message = "runtime_image must be empty or an ECR image reference pinned by sha256."
  }
}

variable "availability_zones" {
  description = "Three distinct account-mapped AZ names."
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]

  validation {
    condition     = length(var.availability_zones) == 3 && length(toset(var.availability_zones)) == 3
    error_message = "E11-AWS requires three distinct Availability Zones."
  }
}

variable "instance_type" {
  description = "x86_64 instance type used for all three E11 hosts."
  type        = string
  default     = "t3.small"
}

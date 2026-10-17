variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "billing_account" {
  description = "Billing account ID, e.g. 0X0X0X-0X0X0X-0X0X0X"
  type        = string
}

variable "budget_amount" {
  description = "Monthly budget in the billing account's currency"
  type        = number
  default     = 20
}

variable "alert_email" {
  description = "Where drift alerts are emailed"
  type        = string
}

# --- Stage 6: SLOs and error-budget burn alerts for the serving API ---
#
# Cloud Run is deployed by Cloud Build, so it is referenced here by name only.
# SLIs are request-based, built from Cloud Run's own metrics (no app changes).

locals {
  run_filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.api_service_name}\""
}

resource "google_monitoring_custom_service" "api" {
  service_id   = "income-api"
  display_name = "income-api (Cloud Run)"
}

# Availability: a request is bad only if the server failed (5xx).
# 4xx (bad input, unauthenticated) are the caller's fault and don't spend budget.
resource "google_monitoring_slo" "availability" {
  service             = google_monitoring_custom_service.api.service_id
  slo_id              = "availability-99"
  display_name        = "99% of requests succeed (non-5xx), 28-day rolling"
  goal                = var.availability_goal
  rolling_period_days = 28

  request_based_sli {
    good_total_ratio {
      bad_service_filter   = "metric.type=\"run.googleapis.com/request_count\" AND ${local.run_filter} AND metric.labels.response_code_class=\"5xx\""
      total_service_filter = "metric.type=\"run.googleapis.com/request_count\" AND ${local.run_filter}"
    }
  }
}

# Latency: measured p95 was ~1.2 s and p99 ~11 s (cold starts with min-instances 0,
# plus the Gemini call in /predict-explained). 95% under 2 s is the honest target.
resource "google_monitoring_slo" "latency" {
  service             = google_monitoring_custom_service.api.service_id
  slo_id              = "latency-95-2s"
  display_name        = "95% of requests under ${var.latency_threshold_ms} ms, 28-day rolling"
  goal                = var.latency_goal
  rolling_period_days = 28

  request_based_sli {
    distribution_cut {
      distribution_filter = "metric.type=\"run.googleapis.com/request_latencies\" AND ${local.run_filter}"
      range {
        max = var.latency_threshold_ms
      }
    }
  }
}

# Multi-window, multi-burn-rate alerts (Google SRE workbook, ch. 5):
#   fast: 14.4x burn over 1h, confirmed over 5m -> ~2% of a 28-day budget gone in an hour
#   slow:  6x  burn over 6h, confirmed over 30m -> ~5% of the budget gone in six hours
locals {
  burn_alerts = {
    availability-fast = { slo = google_monitoring_slo.availability.id, rate = 14.4, long = "3600s", short = "300s" }
    availability-slow = { slo = google_monitoring_slo.availability.id, rate = 6, long = "21600s", short = "1800s" }
    latency-slow      = { slo = google_monitoring_slo.latency.id, rate = 6, long = "21600s", short = "1800s" }
  }
}

resource "google_monitoring_alert_policy" "slo_burn" {
  for_each     = local.burn_alerts
  display_name = "income-api SLO burn: ${each.key}"
  combiner     = "AND"

  conditions {
    display_name = "Burn rate > ${each.value.rate} over ${each.value.long}"
    condition_threshold {
      filter          = "select_slo_burn_rate(\"${each.value.slo}\", \"${each.value.long}\")"
      comparison      = "COMPARISON_GT"
      threshold_value = each.value.rate
      duration        = "0s"
    }
  }

  conditions {
    display_name = "Burn rate > ${each.value.rate} over ${each.value.short} (still happening)"
    condition_threshold {
      filter          = "select_slo_burn_rate(\"${each.value.slo}\", \"${each.value.short}\")"
      comparison      = "COMPARISON_GT"
      threshold_value = each.value.rate
      duration        = "0s"
    }
  }

  alert_strategy {
    auto_close = "86400s"
  }

  documentation {
    mime_type = "text/markdown"
    content   = <<-EOT
      The income-api error budget is burning ${each.value.rate}x faster than the SLO allows.
      1. Cloud Run > income-api > Revisions: did a deploy just happen? If so, shift traffic back to the previous revision.
      2. Logs Explorer: `resource.labels.service_name="income-api" severity>=ERROR`.
      3. `/health` shows the model_version and app_version that are serving.
      Low traffic caveat: at a few requests per hour, a single 5xx can trip the fast alert.
    EOT
  }

  notification_channels = [google_monitoring_notification_channel.email.id]
}

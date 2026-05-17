"""Admin registration for the Analysis app."""

from django.contrib import admin
from .models import AnalysisRecord, AlertLog


@admin.register(AnalysisRecord)
class AnalysisRecordAdmin(admin.ModelAdmin):
    list_display = ["id", "text_preview", "severity", "model_used", "alert_triggered", "created_at"]
    list_filter = ["model_used", "alert_triggered", "severity"]
    search_fields = ["text_preview"]
    readonly_fields = ["created_at"]
    ordering = ["-created_at"]


@admin.register(AlertLog)
class AlertLogAdmin(admin.ModelAdmin):
    list_display = ["id", "severity", "severity_status", "created_at"]
    list_filter = ["severity", "severity_status"]
    ordering = ["-created_at"]

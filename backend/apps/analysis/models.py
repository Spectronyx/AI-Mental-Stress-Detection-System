"""
Database models for the Analysis app.

AnalysisRecord — stores each text analysis result
AlertLog — stores triggered alerts linked to records
"""

from django.db import models


class AnalysisRecord(models.Model):
    """
    Stores a complete mental stress analysis result.
    """

    MODEL_CHOICES = [
        ("svm", "Support Vector Machine"),
        ("rf", "Random Forest"),
        ("lstm", "Bidirectional LSTM"),
    ]

    # Input
    text = models.TextField(help_text="Original input text (max 2000 chars)", max_length=2000)
    text_preview = models.CharField(max_length=200, blank=True)

    # Labels (stored as JSON lists)
    thematic_labels = models.JSONField(default=list, help_text="e.g. ['Anxiety', 'Depression']")
    categorical_labels = models.JSONField(default=list, help_text="e.g. ['Panic', 'Hopelessness']")
    trigger_labels = models.JSONField(default=list, help_text="e.g. ['Work', 'Financial']")

    # Scores
    thematic_scores = models.JSONField(default=dict)
    categorical_scores = models.JSONField(default=dict)
    trigger_scores = models.JSONField(default=dict)

    # Severity
    severity = models.IntegerField(default=1, help_text="1 (minimal) to 5 (critical)")
    severity_probabilities = models.JSONField(default=list)

    # Confidence
    confidence_score = models.FloatField(default=0.0)

    # Model metadata
    model_used = models.CharField(max_length=10, choices=MODEL_CHOICES, default="lstm")
    processing_time_ms = models.FloatField(default=0.0)

    # Alert
    alert_triggered = models.BooleanField(default=False)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Analysis Record"
        verbose_name_plural = "Analysis Records"
        indexes = [
            models.Index(fields=["created_at"]),
            models.Index(fields=["severity"]),
            models.Index(fields=["alert_triggered"]),
            models.Index(fields=["model_used"]),
        ]

    def __str__(self):
        return f"[{self.model_used.upper()}] Severity={self.severity} | {self.text_preview[:50]}"

    def save(self, *args, **kwargs):
        if not self.text_preview:
            self.text_preview = self.text[:200]
        super().save(*args, **kwargs)


class AlertLog(models.Model):
    """
    Log of triggered alerts for high/critical severity detections.
    """

    SEVERITY_CHOICES = [(i, str(i)) for i in range(1, 6)]

    record = models.OneToOneField(
        AnalysisRecord,
        on_delete=models.CASCADE,
        related_name="alert_log",
    )
    severity = models.IntegerField(choices=SEVERITY_CHOICES)
    severity_status = models.CharField(max_length=20)
    alert_message = models.TextField()
    recommendation = models.TextField()
    empathetic_response = models.TextField(blank=True, null=True)
    resources = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Alert Log"
        verbose_name_plural = "Alert Logs"

    def __str__(self):
        return f"Alert [Severity={self.severity}] at {self.created_at:%Y-%m-%d %H:%M}"

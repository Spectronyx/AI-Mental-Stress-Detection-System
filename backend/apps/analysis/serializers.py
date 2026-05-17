"""
DRF Serializers for the Analysis app.
"""

from rest_framework import serializers
from .models import AnalysisRecord, AlertLog


class AlertLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = AlertLog
        fields = [
            "severity", "severity_status", "alert_message",
            "recommendation", "empathetic_response", "resources", "created_at",
        ]


class AnalysisRecordSerializer(serializers.ModelSerializer):
    alert_log = AlertLogSerializer(read_only=True)

    class Meta:
        model = AnalysisRecord
        fields = [
            "id", "text_preview", "thematic_labels", "categorical_labels",
            "trigger_labels", "thematic_scores", "categorical_scores",
            "trigger_scores", "severity", "severity_probabilities",
            "confidence_score", "model_used", "processing_time_ms",
            "alert_triggered", "alert_log", "created_at",
        ]
        read_only_fields = fields


class AnalysisRequestSerializer(serializers.Serializer):
    """Input serializer for /api/analyze."""

    text = serializers.CharField(
        min_length=5,
        max_length=2000,
        help_text="Text to analyze (5–2000 characters)",
    )
    model = serializers.ChoiceField(
        choices=["svm", "rf", "lstm", "ensemble"],
        default="ensemble",
        required=False,
        help_text="ML model to use. Default: ensemble (weighted vote)",
    )
    threshold = serializers.FloatField(
        min_value=0.1,
        max_value=0.9,
        default=0.5,
        required=False,
        help_text="Probability threshold for multi-label classification. Default: 0.5",
    )


class AnalysisResponseSerializer(serializers.Serializer):
    """Response schema for /api/analyze (for Swagger docs)."""

    success = serializers.BooleanField()
    record_id = serializers.IntegerField(required=False)
    text_preview = serializers.CharField()
    model_used = serializers.CharField()
    processing_time_ms = serializers.FloatField()
    thematic_labels = serializers.ListField(child=serializers.CharField())
    thematic_scores = serializers.DictField()
    categorical_labels = serializers.ListField(child=serializers.CharField())
    categorical_scores = serializers.DictField()
    trigger_labels = serializers.ListField(child=serializers.CharField())
    trigger_scores = serializers.DictField()
    severity = serializers.IntegerField()
    confidence_score = serializers.FloatField()
    alert = serializers.DictField()
    disclaimer = serializers.CharField()


class ModelComparisonSerializer(serializers.Serializer):
    """Schema for /api/compare response."""

    model = serializers.CharField()
    accuracy = serializers.DictField()
    precision = serializers.DictField()
    recall = serializers.DictField()
    f1 = serializers.DictField()


class AlertThresholdSerializer(serializers.Serializer):
    """Request body for updating alert threshold."""

    threshold = serializers.IntegerField(min_value=1, max_value=5)

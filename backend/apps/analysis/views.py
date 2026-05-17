"""
Analysis API Views

Endpoints:
    POST /api/analyze          — Analyze text for stress indicators
    GET  /api/history          — Paginated analysis history
    GET  /api/history/<id>/    — Single record detail
    GET  /api/compare          — Model comparison metrics
    GET  /api/stats            — Dashboard statistics
    POST /api/threshold        — Update alert threshold
"""

import json
import logging
import os

from django.conf import settings
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page

from rest_framework import status
from rest_framework.generics import ListAPIView, RetrieveAPIView
from rest_framework.response import Response
from rest_framework.views import APIView
from drf_spectacular.utils import extend_schema, OpenApiParameter

from .models import AnalysisRecord, AlertLog
from .serializers import (
    AnalysisRequestSerializer,
    AnalysisResponseSerializer,
    AnalysisRecordSerializer,
    ModelComparisonSerializer,
    AlertThresholdSerializer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy pipeline loader
# ---------------------------------------------------------------------------

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        import sys
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        from ml.pipeline import get_pipeline
        _pipeline = get_pipeline(
            alert_threshold=getattr(settings, "ML_ALERT_THRESHOLD", 4)
        )
    return _pipeline


# ---------------------------------------------------------------------------
# /api/analyze
# ---------------------------------------------------------------------------

class AnalyzeView(APIView):
    """
    Analyze text for mental stress indicators.

    Submit text and receive multi-label classifications,
    severity level, confidence scores, and early warning alerts.
    """

    @extend_schema(
        request=AnalysisRequestSerializer,
        responses={200: AnalysisResponseSerializer},
        tags=["Analysis"],
        summary="Analyze text for mental stress",
        description=(
            "Runs the selected ML model on the input text and returns "
            "thematic labels, categorical labels, triggers, severity (1-5), "
            "confidence scores, and an alert payload.\n\n"
            "⚠️ **Disclaimer**: Not a medical diagnostic tool."
        ),
    )
    def post(self, request):
        serializer = AnalysisRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        text = serializer.validated_data["text"]
        model_name = serializer.validated_data.get("model", getattr(settings, "ML_DEFAULT_MODEL", "ensemble"))
        threshold = serializer.validated_data.get("threshold", 0.5)

        try:
            pipeline = _get_pipeline()
            result = pipeline.analyze(text, model=model_name, threshold=threshold)
        except FileNotFoundError:
            return Response(
                {
                    "error": "ML models not found. Please run the training script first.",
                    "hint": "cd backend && python ml/trainer.py",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as exc:
            logger.exception("Analysis failed: %s", exc)
            return Response(
                {"error": "Analysis failed. Check server logs for details."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if not result.get("success"):
            return Response(
                {"error": result.get("error", "Unknown error")},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        # Persist to database
        record = self._save_record(result, text)
        result["record_id"] = record.pk if record else None

        return Response(result, status=status.HTTP_200_OK)

    def _save_record(self, result: dict, original_text: str):
        """Persist analysis result to database."""
        try:
            alert_data = result.get("alert", {})
            record = AnalysisRecord.objects.create(
                text=original_text[:2000],
                text_preview=original_text[:200],
                thematic_labels=result.get("thematic_labels", []),
                categorical_labels=result.get("categorical_labels", []),
                trigger_labels=result.get("trigger_labels", []),
                thematic_scores=result.get("thematic_scores", {}),
                categorical_scores=result.get("categorical_scores", {}),
                trigger_scores=result.get("trigger_scores", {}),
                severity=result.get("severity", 1),
                severity_probabilities=result.get("severity_probabilities", []),
                confidence_score=result.get("confidence_score", 0.0),
                model_used=result.get("model_used", "lstm"),
                processing_time_ms=result.get("processing_time_ms", 0.0),
                alert_triggered=alert_data.get("alert_triggered", False),
            )

            if alert_data.get("alert_triggered"):
                AlertLog.objects.create(
                    record=record,
                    severity=alert_data.get("severity", 1),
                    severity_status=alert_data.get("severity_status", ""),
                    alert_message=alert_data.get("alert_message", ""),
                    recommendation=alert_data.get("recommendation", ""),
                    empathetic_response=alert_data.get("empathetic_response"),
                    resources=alert_data.get("resources", []),
                )

            return record
        except Exception as e:
            logger.error("Failed to save analysis record: %s", e)
            return None


# ---------------------------------------------------------------------------
# /api/history
# ---------------------------------------------------------------------------

class AnalysisHistoryListView(ListAPIView):
    """
    Retrieve paginated analysis history.
    Supports filtering by severity, model, and alert status.
    """

    serializer_class = AnalysisRecordSerializer
    filterset_fields = ["model_used", "alert_triggered", "severity"]
    ordering_fields = ["created_at", "severity", "confidence_score"]
    ordering = ["-created_at"]
    search_fields = ["text_preview"]

    @extend_schema(tags=["History"], summary="List analysis history")
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        qs = AnalysisRecord.objects.prefetch_related("alert_log")

        # Optional date filters
        since = self.request.query_params.get("since")
        until = self.request.query_params.get("until")
        if since:
            qs = qs.filter(created_at__gte=since)
        if until:
            qs = qs.filter(created_at__lte=until)

        return qs


class AnalysisHistoryDetailView(RetrieveAPIView):
    """Retrieve a single analysis record by ID."""

    queryset = AnalysisRecord.objects.prefetch_related("alert_log").all()
    serializer_class = AnalysisRecordSerializer

    @extend_schema(tags=["History"], summary="Get a single analysis record")
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)


# ---------------------------------------------------------------------------
# /api/compare  — Model comparison metrics
# ---------------------------------------------------------------------------

class ModelCompareView(APIView):
    """
    Return cached model evaluation metrics (from evaluator.py output).
    """

    @extend_schema(
        tags=["Models"],
        summary="Get model comparison metrics",
        description="Returns 5-fold CV metrics for SVM, RF, and LSTM models.",
    )
    @method_decorator(cache_page(60 * 60))  # Cache 1 hour
    def get(self, request):
        results_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "evaluation_results.json",
        )
        if not os.path.exists(results_path):
            return Response(
                {
                    "error": "Evaluation results not found.",
                    "hint": "Run: python backend/ml/evaluator.py",
                },
                status=status.HTTP_404_NOT_FOUND,
            )
        with open(results_path) as f:
            data = json.load(f)

        formatted = [
            {"model": m.upper(), **metrics}
            for m, metrics in data.items()
        ]
        return Response({"results": formatted}, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# /api/stats  — Dashboard stats
# ---------------------------------------------------------------------------

class DashboardStatsView(APIView):
    """Return aggregate statistics for the dashboard."""

    @extend_schema(tags=["Analysis"], summary="Get dashboard statistics")
    @method_decorator(cache_page(60))  # Cache 1 minute
    def get(self, request):
        from django.db.models import Count, Avg

        total = AnalysisRecord.objects.count()
        alerts = AnalysisRecord.objects.filter(alert_triggered=True).count()
        avg_severity = AnalysisRecord.objects.aggregate(avg=Avg("severity"))["avg"] or 0

        # Severity distribution
        severity_dist = (
            AnalysisRecord.objects.values("severity")
            .annotate(count=Count("id"))
            .order_by("severity")
        )

        # Model usage
        model_usage = (
            AnalysisRecord.objects.values("model_used")
            .annotate(count=Count("id"))
        )

        # Recent trend (last 10)
        recent = list(
            AnalysisRecord.objects.values("created_at", "severity")
            .order_by("-created_at")[:10]
        )
        recent.reverse()

        return Response({
            "total_analyses": total,
            "alerts_triggered": alerts,
            "average_severity": round(avg_severity, 2),
            "severity_distribution": list(severity_dist),
            "model_usage": list(model_usage),
            "recent_trend": [
                {
                    "timestamp": r["created_at"].isoformat(),
                    "severity": r["severity"],
                }
                for r in recent
            ],
        })


# ---------------------------------------------------------------------------
# /api/threshold  — Update alert threshold
# ---------------------------------------------------------------------------

class AlertThresholdView(APIView):
    """Update the alert severity threshold at runtime."""

    @extend_schema(
        request=AlertThresholdSerializer,
        tags=["Analysis"],
        summary="Update alert threshold",
    )
    def post(self, request):
        serializer = AlertThresholdSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        threshold = serializer.validated_data["threshold"]
        try:
            pipeline = _get_pipeline()
            pipeline._get_alert_engine().update_threshold(threshold)
            return Response({"message": f"Alert threshold updated to {threshold}"})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

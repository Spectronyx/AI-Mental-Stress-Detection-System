"""Health check app views and URLs."""

from django.urls import path
from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema
import os
import sys


@extend_schema(tags=["Health"], summary="Service health check")
@api_view(["GET"])
def health_check(request):
    """Returns service health and ML model availability."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    try:
        from ml.pipeline import AnalysisPipeline
        pipeline = AnalysisPipeline()
        model_status = pipeline.health_check()
    except Exception:
        model_status = {"error": "Could not check model status"}

    return Response({
        "status": "ok",
        "service": "Mental Stress Detection API",
        "version": "1.0.0",
        "models": model_status,
        "disclaimer": "This service is for research purposes only. Not a medical tool.",
    })

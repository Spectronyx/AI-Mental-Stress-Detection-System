"""URL routing for the Analysis app."""

from django.urls import path
from . import views

urlpatterns = [
    path("analyze", views.AnalyzeView.as_view(), name="analyze"),
    path("history", views.AnalysisHistoryListView.as_view(), name="history-list"),
    path("history/<int:pk>/", views.AnalysisHistoryDetailView.as_view(), name="history-detail"),
    path("compare", views.ModelCompareView.as_view(), name="model-compare"),
    path("stats", views.DashboardStatsView.as_view(), name="dashboard-stats"),
    path("threshold", views.AlertThresholdView.as_view(), name="alert-threshold"),
]

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from unittest.mock import patch, MagicMock
from .models import AnalysisRecord, AlertLog

class AnalysisViewTests(APITestCase):
    """Test suite for the Analysis app API endpoints."""

    def setUp(self):
        # Create a sample analysis record for list/detail/stats tests
        self.record = AnalysisRecord.objects.create(
            text="I feel extremely stressed out and anxious.",
            text_preview="I feel extremely stressed out...",
            thematic_labels=["Stress"],
            categorical_labels=[],
            trigger_labels=[],
            thematic_scores={"Stress": 0.85},
            categorical_scores={},
            trigger_scores={},
            severity=3,
            severity_probabilities=[0.05, 0.1, 0.1, 0.7, 0.05],
            confidence_score=0.85,
            model_used="lstm",
            processing_time_ms=12.5,
            alert_triggered=False
        )

    @patch('apps.analysis.views._get_pipeline')
    def test_analyze_endpoint_success(self, mock_get_pipeline):
        """Test that POST to /api/analyze processes text and persists record."""
        mock_pipeline = MagicMock()
        mock_pipeline.analyze.return_value = {
            "success": True,
            "text_preview": "I feel absolutely hopeless...",
            "model_used": "ensemble (svm, rf, lstm)",
            "processing_time_ms": 15.2,
            "thematic_labels": ["Depression"],
            "thematic_scores": {"Depression": 0.92, "Normal": 0.08},
            "categorical_labels": [],
            "categorical_scores": {},
            "trigger_labels": [],
            "trigger_scores": {},
            "severity": 4,
            "severity_probabilities": [0.01, 0.02, 0.05, 0.02, 0.9],
            "confidence_score": 0.92,
            "alert": {
                "alert_triggered": True,
                "severity": 4,
                "severity_status": "HIGH RISK",
                "alert_message": "High severity indicators detected.",
                "recommendation": "Consider contacting support services.",
                "resources": ["National Helpline: 988"]
            },
            "disclaimer": "Test disclaimer."
        }
        mock_get_pipeline.return_value = mock_pipeline

        url = reverse('analyze')
        data = {"text": "I feel absolutely hopeless. Nothing seems to work out for me anymore."}
        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['success'], True)
        self.assertEqual(response.data['severity'], 4)
        self.assertEqual(response.data['thematic_labels'], ["Depression"])

        # Check that it saved to database and linked the alert
        new_record = AnalysisRecord.objects.get(severity=4)
        self.assertEqual(new_record.confidence_score, 0.92)
        self.assertTrue(new_record.alert_triggered)
        
        alert_log = AlertLog.objects.get(record=new_record)
        self.assertEqual(alert_log.severity_status, "HIGH RISK")

    def test_analyze_endpoint_invalid_data(self):
        """Test that POST with short text or missing text fails."""
        url = reverse('analyze')
        
        # Too short text (min_length=5)
        response = self.client.post(url, {"text": "Sad"}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('text', response.data)

        # Missing text
        response = self.client.post(url, {}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_history_list_endpoint(self):
        """Test retrieving paginated analysis history."""
        url = reverse('history-list')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('results', response.data)
        self.assertEqual(len(response.data['results']), 1)
        self.assertEqual(response.data['results'][0]['id'], self.record.id)

    def test_dashboard_stats_endpoint(self):
        """Test retrieving dashboard statistics."""
        url = reverse('dashboard-stats')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['total_analyses'], 1)
        self.assertEqual(response.data['average_severity'], 3.0)
        self.assertEqual(response.data['alerts_triggered'], 0)

    @patch('apps.analysis.views._get_pipeline')
    def test_update_alert_threshold(self, mock_get_pipeline):
        """Test changing alert threshold."""
        mock_pipeline = MagicMock()
        mock_alert_engine = MagicMock()
        mock_pipeline._get_alert_engine.return_value = mock_alert_engine
        mock_get_pipeline.return_value = mock_pipeline

        url = reverse('alert-threshold')
        response = self.client.post(url, {"threshold": 4}, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("message", response.data)
        mock_alert_engine.update_threshold.assert_called_once_with(4)

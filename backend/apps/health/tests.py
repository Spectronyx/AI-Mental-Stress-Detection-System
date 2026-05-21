from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from unittest.mock import patch

class HealthCheckViewTests(APITestCase):
    """Test suite for the API service health-check endpoint."""

    @patch('ml.pipeline.AnalysisPipeline.health_check')
    def test_health_check_success(self, mock_health_check):
        """Test that the health endpoint returns service state and ML availability."""
        mock_health_check.return_value = {
            "svm": True,
            "rf": True,
            "lstm": False,
            "feature_extractor": True,
            "label_encoders": True,
            "all_ready": False
        }

        url = reverse('health-check')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'ok')
        self.assertEqual(response.data['service'], 'Mental Stress Detection API')
        self.assertEqual(response.data['version'], '1.0.0')
        self.assertEqual(response.data['models']['svm'], True)
        self.assertEqual(response.data['models']['all_ready'], False)

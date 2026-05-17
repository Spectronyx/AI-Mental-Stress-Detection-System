"""
Alert Engine — Early Warning System for Mental Stress Detection

Triggers alerts when severity >= threshold (default: 4)
Generates warning messages and recommendations.
Optional LLM integration for empathetic responses.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "alert_threshold": 4,  # Severity >= this → alert
    "critical_threshold": 5,
    "llm_enabled": False,   # Set True to enable LLM responses
}

SEVERITY_MESSAGES = {
    1: {
        "status": "minimal",
        "message": "Your text indicates minimal stress. You seem to be doing well!",
        "recommendation": "Continue with your regular self-care routine.",
        "color": "#10b981",
        "icon": "✅",
    },
    2: {
        "status": "mild",
        "message": "Your text suggests mild emotional distress.",
        "recommendation": "Consider journaling your thoughts, light exercise, or talking to a friend.",
        "color": "#84cc16",
        "icon": "🟡",
    },
    3: {
        "status": "moderate",
        "message": "Moderate levels of stress detected in your text.",
        "recommendation": "It may help to speak with someone you trust. Mindfulness and rest are important.",
        "color": "#f59e0b",
        "icon": "🟠",
    },
    4: {
        "status": "high",
        "message": "⚠️ High stress indicators detected. This is an early warning sign.",
        "recommendation": "We strongly recommend reaching out to a mental health professional or counselor. You don't have to handle this alone.",
        "color": "#ef4444",
        "icon": "🔴",
    },
    5: {
        "status": "critical",
        "message": "🚨 Critical stress markers detected. Immediate attention suggested.",
        "recommendation": "Please reach out to a mental health crisis line or professional immediately. If you are in India, call iCall: 9152987821. In the US: 988 Suicide & Crisis Lifeline.",
        "color": "#dc2626",
        "icon": "🆘",
    },
}

THEMATIC_RESPONSES = {
    "Normal": "You appear to be in a balanced state. Keep taking care of yourself.",
    "Depression": "Feelings of depression are valid and treatable. Please consider speaking to a professional.",
    "Suicidal": "🚨 We detect indicators of suicidal ideation. Please reach out to a crisis line immediately. You are not alone.",
    "Anxiety": "Your text reflects significant anxiety. Remember: anxiety is manageable with the right support.",
    "Bipolar": "Living with bipolar disorder has its challenges. Professional guidance can make a real difference.",
    "Stress": "High stress levels detected. Consider relaxation techniques, exercise, or talking to someone you trust.",
    "Personality Disorder": "Navigating personality-related challenges is tough. Therapy and support groups can help.",
}


class AlertEngine:
    """
    Evaluates stress severity and generates alerts, recommendations,
    and optionally empathetic LLM responses.
    """

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._llm_client = None

        if self.config.get("llm_enabled"):
            self._init_llm()

    def _init_llm(self):
        """Initialise LLM client (OpenAI/Gemini stub)."""
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning(
                "LLM enabled but no API key found. Set OPENAI_API_KEY or GEMINI_API_KEY."
            )
            self.config["llm_enabled"] = False
            return
        try:
            import openai
            self._llm_client = openai.OpenAI(api_key=api_key)
            logger.info("LLM client initialised (OpenAI).")
        except ImportError:
            logger.warning("openai package not installed. LLM disabled.")
            self.config["llm_enabled"] = False

    def check(
        self,
        severity: int,
        thematic_labels: list[str] = None,
        original_text: str = None,
    ) -> dict:
        """
        Evaluate severity and build alert payload.

        Args:
            severity: Integer 1–5
            thematic_labels: List of detected thematic labels
            original_text: Original input text (used for LLM)

        Returns:
            Alert dict with status, message, recommendation, alert_triggered
        """
        severity = max(1, min(5, int(severity)))
        info = SEVERITY_MESSAGES[severity]
        threshold = self.config["alert_threshold"]

        alert_triggered = severity >= threshold

        result = {
            "severity": severity,
            "severity_status": info["status"],
            "alert_triggered": alert_triggered,
            "alert_message": info["message"],
            "recommendation": info["recommendation"],
            "severity_color": info["color"],
            "severity_icon": info["icon"],
            "resources": self._crisis_resources() if severity >= 4 else [],
            "thematic_commentary": self._thematic_commentary(thematic_labels or []),
            "empathetic_response": None,
        }

        if alert_triggered and self.config.get("llm_enabled") and original_text:
            result["empathetic_response"] = self._generate_empathetic_response(
                original_text, severity, thematic_labels or []
            )

        return result

    def _thematic_commentary(self, labels: list[str]) -> str:
        """Return a contextual comment based on detected thematic labels."""
        if not labels:
            return ""
        responses = [THEMATIC_RESPONSES.get(label, "") for label in labels if label in THEMATIC_RESPONSES]
        return " ".join(filter(None, responses))

    def _crisis_resources(self) -> list[dict]:
        return [
            {"name": "iCall (India)", "contact": "9152987821", "type": "phone"},
            {"name": "Vandrevala Foundation Helpline", "contact": "1860-2662-345", "type": "phone"},
            {"name": "988 Lifeline (USA)", "contact": "988", "type": "phone"},
            {"name": "Crisis Text Line (USA)", "contact": "Text HOME to 741741", "type": "text"},
        ]

    def _generate_empathetic_response(
        self, text: str, severity: int, thematic_labels: list[str]
    ) -> str:
        """
        Generate an empathetic response via LLM.

        This is a **stub** — replace with actual API call.
        The system prompt is designed to be safe and non-harmful.
        """
        if not self._llm_client:
            return self._fallback_empathetic_response(severity, thematic_labels)

        system_prompt = (
            "You are a compassionate mental wellness AI assistant. "
            "Your role is to provide warm, supportive, and non-judgmental responses. "
            "IMPORTANT SAFETY RULES: "
            "1. Never provide medical diagnoses. "
            "2. Always encourage professional help for severe distress. "
            "3. Never minimize the user's feelings. "
            "4. Keep responses under 100 words. "
            "5. End with a gentle encouragement."
        )
        user_prompt = (
            f"A person wrote: '{text}'\n"
            f"Detected stress indicators: {', '.join(thematic_labels)}.\n"
            f"Severity level: {severity}/5.\n"
            "Provide a compassionate, supportive response:"
        )

        try:
            response = self._llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            return self._fallback_empathetic_response(severity, thematic_labels)

    def _fallback_empathetic_response(self, severity: int, labels: list[str]) -> str:
        """Pre-written empathetic response when LLM is unavailable."""
        label_str = " and ".join(labels) if labels else "emotional distress"
        if severity <= 2:
            return f"It sounds like you're navigating some challenges with {label_str}. You're not alone, and it's okay to reach out."
        elif severity == 3:
            return f"What you're feeling with {label_str} is completely valid. Consider talking to someone you trust — sharing can make a meaningful difference."
        else:
            return (
                f"I can hear how difficult things are right now. Feelings of {label_str} can be overwhelming. "
                "Please know that professional support is available and you deserve to feel better. "
                "Reaching out is a sign of strength, not weakness."
            )

    def update_threshold(self, threshold: int) -> None:
        """Dynamically update the alert threshold."""
        if not 1 <= threshold <= 5:
            raise ValueError("Threshold must be between 1 and 5")
        self.config["alert_threshold"] = threshold
        logger.info("Alert threshold updated to %d", threshold)

"""
Configuration settings for Shark Tank Analysis System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Langfuse Configuration
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-...")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-...")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")  # 🇪🇺 EU region
# LANGFUSE_BASE_URL = "https://us.cloud.langfuse.com"  # 🇺🇸 US region
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"

# Analysis Configuration
ANALYSIS_CONFIG = {
    "max_analysis_history": 50,
    "confidence_threshold": 0.7,
    "success_probability_thresholds": {
        "high": 0.7,
        "medium": 0.4,
        "low": 0.0
    },
    "equity_sweet_spot": {
        "min": 10,
        "max": 30
    },
    "ask_amount_ranges": {
        "low": 50000,
        "medium": 500000,
        "high": 2000000
    }
}

# Visualization Configuration
VIZ_CONFIG = {
    "colors": {
        "primary": "#1f77b4",
        "success": "#2e7d32",
        "warning": "#ff9800",
        "error": "#d32f2f",
        "info": "#2196f3"
    },
    "chart_height": 400,
    "max_categories": 10
}

# File Upload Configuration
UPLOAD_CONFIG = {
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": [".csv", ".txt", ".md", ".json"],
    "max_rows_preview": 100
}

# Report Configuration
REPORT_CONFIG = {
    "include_visualizations": True,
    "include_recommendations": True,
    "include_risk_analysis": True,
    "max_recommendations": 10
}

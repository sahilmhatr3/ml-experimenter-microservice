"""
ML Experiment Microservice - Main application entry point.
"""

import uvicorn
from ml_service.app import app

if __name__ == "__main__":
    uvicorn.run(
        "ml_service.app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )

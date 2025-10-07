"""
FastAPI application for ML Experiment Microservice.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from datetime import datetime

from .models import MLJobRequest, MLJobResponse
from .experiment_runner import ExperimentRunner
from .artifact_manager import ArtifactManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_service")

# Create FastAPI application
app = FastAPI(
    title="ML Experiment Microservice",
    description="Machine learning experiment execution service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
experiment_runner = ExperimentRunner()
artifact_manager = ArtifactManager()

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ml-experiment-microservice",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health/ready")
async def readiness_check():
    """Readiness check endpoint."""
    return {
        "status": "ready",
        "service": "ml-experiment-microservice",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/datasets")
async def list_datasets():
    """List available datasets."""
    return {
        "datasets": [
            {"name": "iris", "description": "Iris flower dataset"},
            {"name": "wine", "description": "Wine quality dataset"},
            {"name": "breast_cancer", "description": "Breast cancer dataset"},
            {"name": "digits", "description": "Handwritten digits dataset"}
        ]
    }


@app.get("/models")
async def list_models():
    """List supported model types."""
    return {
        "models": [
            {"name": "linear", "description": "Linear Regression/Classification"},
            {"name": "random_forest", "description": "Random Forest"},
            {"name": "neural_network", "description": "Neural Network (MLP)"},
            {"name": "svm", "description": "Support Vector Machine"},
            {"name": "decision_tree", "description": "Decision Tree"}
        ]
    }


@app.post("/execute", response_model=MLJobResponse)
async def execute_ml_job(request: MLJobRequest):
    """
    Execute a machine learning experiment.
    
    Args:
        request: ML job request with configuration
        
    Returns:
        ML job response with results and artifacts
    """
    try:
        logger.info(f"Executing ML job {request.job_id}")
        
        # Run the experiment
        result = await experiment_runner.run_experiment(request)
        
        # Generate artifacts
        artifacts = await artifact_manager.generate_artifacts(
            job_id=request.job_id,
            result=result,
            config=request.payload
        )
        
        # Create response
        response = MLJobResponse(
            status="completed",
            result=result,
            artifacts=artifacts
        )
        
        logger.info(f"ML job {request.job_id} completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"ML job {request.job_id} failed: {e}")
        return MLJobResponse(
            status="failed",
            error=str(e)
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": "internal_error"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

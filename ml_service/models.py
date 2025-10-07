"""
Data models for ML Experiment Microservice.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class ModelType(str, Enum):
    """Supported ML model types."""
    LINEAR = "linear"
    RANDOM_FOREST = "random_forest"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"
    DECISION_TREE = "decision_tree"


class DatasetType(str, Enum):
    """Supported datasets."""
    IRIS = "iris"
    WINE = "wine"
    BREAST_CANCER = "breast_cancer"
    DIGITS = "digits"
    CUSTOM = "custom"


class ArtifactType(str, Enum):
    """Artifact types."""
    MODEL = "model"
    PLOT = "plot"
    METRICS = "metrics"
    LOG = "log"
    DATA = "data"


class MLJobRequest(BaseModel):
    """Request model for ML job execution."""
    
    job_id: str
    type: str = "ml_experiment"
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class ArtifactInfo(BaseModel):
    """Information about a generated artifact."""
    
    name: str
    type: ArtifactType
    storage_location: str
    description: Optional[str] = None
    service_id: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MLJobResponse(BaseModel):
    """Response model for ML job execution."""
    
    status: str
    result: Optional[Dict[str, Any]] = None
    artifacts: List[ArtifactInfo] = Field(default_factory=list)
    error: Optional[str] = None


class ExperimentConfig(BaseModel):
    """Configuration for ML experiment."""
    
    model_config = {"protected_namespaces": ()}
    
    model_type: ModelType
    dataset: DatasetType
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = Field(default=42)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    artifact_dependencies: List[str] = Field(default_factory=list)

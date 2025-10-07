"""
ML Experiment Runner - Core logic for running machine learning experiments.
"""

import asyncio
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import joblib
import hashlib
import os

from .models import MLJobRequest, ModelType, DatasetType, ExperimentConfig

logger = logging.getLogger("experiment_runner")


class ExperimentRunner:
    """Runs machine learning experiments."""
    
    def __init__(self):
        self.datasets = {
            DatasetType.IRIS: load_iris,
            DatasetType.WINE: load_wine,
            DatasetType.BREAST_CANCER: load_breast_cancer,
            DatasetType.DIGITS: load_digits
        }
        
        self.models = {
            ModelType.LINEAR: {
                "classifier": LogisticRegression,
                "regressor": LinearRegression
            },
            ModelType.RANDOM_FOREST: {
                "classifier": RandomForestClassifier,
                "regressor": RandomForestRegressor
            },
            ModelType.NEURAL_NETWORK: {
                "classifier": MLPClassifier,
                "regressor": MLPRegressor
            },
            ModelType.SVM: {
                "classifier": SVC,
                "regressor": SVR
            },
            ModelType.DECISION_TREE: {
                "classifier": DecisionTreeClassifier,
                "regressor": DecisionTreeRegressor
            }
        }
    
    async def run_experiment(self, request: MLJobRequest) -> Dict[str, Any]:
        """
        Run a machine learning experiment.
        
        Args:
            request: ML job request
            
        Returns:
            Experiment results
        """
        try:
            # Parse configuration
            config = self._parse_config(request.payload)
            
            # Load dataset
            X, y, dataset_info = await self._load_dataset(config.dataset)
            
            # Determine if classification or regression
            is_classification = self._is_classification_task(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=config.test_size, 
                random_state=config.random_state,
                stratify=y if is_classification else None
            )
            
            # Create and train model
            model = await self._create_model(config.model_type, is_classification, config.parameters)
            
            # Train model
            logger.info(f"Training {config.model_type} model on {config.dataset} dataset")
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self._evaluate_model(model, X_train, y_train, is_classification)
            test_score = self._evaluate_model(model, X_test, y_test, is_classification)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Generate predictions for analysis
            y_pred = model.predict(X_test)
            
            # Create experiment ID
            experiment_id = f"exp_{request.job_id[:8]}"
            
            # Save model
            model_path = f"artifacts/model_{experiment_id}.pkl"
            os.makedirs("artifacts", exist_ok=True)
            joblib.dump(model, model_path)
            
            # Calculate model size
            model_size = os.path.getsize(model_path)
            
            # Generate checksum
            with open(model_path, 'rb') as f:
                model_checksum = hashlib.sha256(f.read()).hexdigest()
            
            # Prepare results
            results = {
                "experiment_id": experiment_id,
                "model_type": config.model_type,
                "dataset": config.dataset,
                "is_classification": is_classification,
                "train_score": float(train_score),
                "test_score": float(test_score),
                "cv_mean": float(cv_scores.mean()),
                "cv_std": float(cv_scores.std()),
                "model_size_bytes": model_size,
                "model_checksum": model_checksum,
                "model_path": model_path,
                "dataset_info": dataset_info,
                "parameters": config.parameters,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "features": X.shape[1],
                "classes": len(np.unique(y)) if is_classification else None
            }
            
            # Add classification-specific metrics
            if is_classification:
                results.update({
                    "accuracy": float(test_score),
                    "classification_report": classification_report(y_test, y_pred, output_dict=True)
                })
            else:
                results.update({
                    "mse": float(test_score),
                    "rmse": float(np.sqrt(test_score))
                })
            
            logger.info(f"Experiment {experiment_id} completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
    
    def _parse_config(self, payload: Dict[str, Any]) -> ExperimentConfig:
        """Parse experiment configuration from payload."""
        return ExperimentConfig(
            model_type=ModelType(payload.get("model_type", "linear")),
            dataset=DatasetType(payload.get("dataset", "iris")),
            test_size=payload.get("test_size", 0.2),
            random_state=payload.get("random_state", 42),
            parameters=payload.get("parameters", {}),
            artifact_dependencies=payload.get("artifact_dependencies", [])
        )
    
    async def _load_dataset(self, dataset_type: DatasetType) -> tuple:
        """Load dataset and return features, targets, and info."""
        if dataset_type == DatasetType.CUSTOM:
            raise ValueError("Custom datasets not yet supported")
        
        loader_func = self.datasets[dataset_type]
        data = loader_func()
        
        X = data.data
        y = data.target
        
        dataset_info = {
            "name": dataset_type,
            "samples": X.shape[0],
            "features": X.shape[1],
            "target_names": data.target_names.tolist() if hasattr(data, 'target_names') and data.target_names is not None else None,
            "feature_names": data.feature_names if hasattr(data, 'feature_names') and data.feature_names is not None else None,
            "description": data.DESCR[:200] + "..." if hasattr(data, 'DESCR') and data.DESCR is not None else None
        }
        
        return X, y, dataset_info
    
    def _is_classification_task(self, y: np.ndarray) -> bool:
        """Determine if task is classification or regression."""
        unique_values = np.unique(y)
        return len(unique_values) <= 20 and all(isinstance(val, (int, np.integer)) for val in unique_values)
    
    async def _create_model(self, model_type: ModelType, is_classification: bool, parameters: Dict[str, Any]):
        """Create model instance based on type and task."""
        model_class = self.models[model_type]["classifier" if is_classification else "regressor"]
        
        # Set default parameters
        default_params = self._get_default_parameters(model_type, is_classification)
        default_params.update(parameters)
        
        return model_class(**default_params)
    
    def _get_default_parameters(self, model_type: ModelType, is_classification: bool) -> Dict[str, Any]:
        """Get default parameters for model type."""
        defaults = {
            ModelType.LINEAR: {"random_state": 42},
            ModelType.RANDOM_FOREST: {"random_state": 42, "n_estimators": 100},
            ModelType.NEURAL_NETWORK: {"random_state": 42, "max_iter": 1000},
            ModelType.SVM: {"random_state": 42},
            ModelType.DECISION_TREE: {"random_state": 42}
        }
        return defaults.get(model_type, {})
    
    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray, is_classification: bool) -> float:
        """Evaluate model performance."""
        y_pred = model.predict(X)
        
        if is_classification:
            return accuracy_score(y, y_pred)
        else:
            return mean_squared_error(y, y_pred)

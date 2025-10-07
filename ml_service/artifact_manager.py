"""
Artifact Manager - Handles generation and management of ML experiment artifacts.
"""

import os
import logging
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib

from .models import ArtifactInfo, ArtifactType

logger = logging.getLogger("artifact_manager")


class ArtifactManager:
    """Manages artifact generation and storage."""
    
    def __init__(self):
        self.artifacts_dir = "artifacts"
        os.makedirs(self.artifacts_dir, exist_ok=True)
    
    async def generate_artifacts(self, job_id: str, result: Dict[str, Any], config: Dict[str, Any]) -> List[ArtifactInfo]:
        """
        Generate artifacts from experiment results.
        
        Args:
            job_id: Job identifier
            result: Experiment results
            config: Experiment configuration
            
        Returns:
            List of artifact information
        """
        artifacts = []
        experiment_id = result.get("experiment_id", f"exp_{job_id[:8]}")
        
        # 1. Model artifact (already saved by experiment runner)
        model_artifact = ArtifactInfo(
            name=f"model_{experiment_id}.pkl",
            type=ArtifactType.MODEL,
            storage_location=result["model_path"],
            description=f"Trained {result['model_type']} model on {result['dataset']} dataset",
            service_id="ml-service-1",
            size_bytes=result["model_size_bytes"],
            checksum=result["model_checksum"],
            tags=["production", "v1.0", result["model_type"], result["dataset"]],
            metadata={
                "model_type": result["model_type"],
                "dataset": result["dataset"],
                "test_score": result["test_score"],
                "cv_mean": result["cv_mean"],
                "is_classification": result["is_classification"]
            }
        )
        artifacts.append(model_artifact)
        
        # 2. Metrics artifact
        metrics_artifact = await self._create_metrics_artifact(experiment_id, result)
        artifacts.append(metrics_artifact)
        
        # 3. Performance plot artifact
        plot_artifact = await self._create_performance_plot(experiment_id, result)
        artifacts.append(plot_artifact)
        
        # 4. Experiment log artifact
        log_artifact = await self._create_experiment_log(experiment_id, result, config)
        artifacts.append(log_artifact)
        
        logger.info(f"Generated {len(artifacts)} artifacts for experiment {experiment_id}")
        return artifacts
    
    async def _create_metrics_artifact(self, experiment_id: str, result: Dict[str, Any]) -> ArtifactInfo:
        """Create metrics artifact."""
        metrics_data = {
            "experiment_id": experiment_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model_type": result["model_type"],
            "dataset": result["dataset"],
            "train_score": result["train_score"],
            "test_score": result["test_score"],
            "cv_mean": result["cv_mean"],
            "cv_std": result["cv_std"],
            "training_samples": result["training_samples"],
            "test_samples": result["test_samples"],
            "features": result["features"],
            "is_classification": result["is_classification"]
        }
        
        # Add classification-specific metrics
        if result["is_classification"]:
            metrics_data.update({
                "accuracy": result["accuracy"],
                "classes": result["classes"]
            })
        else:
            metrics_data.update({
                "mse": result["mse"],
                "rmse": result["rmse"]
            })
        
        # Save metrics to JSON file
        import json
        metrics_path = f"{self.artifacts_dir}/metrics_{experiment_id}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Calculate file info
        size_bytes = os.path.getsize(metrics_path)
        with open(metrics_path, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()
        
        return ArtifactInfo(
            name=f"metrics_{experiment_id}.json",
            type=ArtifactType.METRICS,
            storage_location=metrics_path,
            description=f"Performance metrics for {result['model_type']} experiment",
            service_id="ml-service-1",
            size_bytes=size_bytes,
            checksum=checksum,
            tags=["metrics", result["model_type"], result["dataset"]],
            metadata=metrics_data
        )
    
    async def _create_performance_plot(self, experiment_id: str, result: Dict[str, Any]) -> ArtifactInfo:
        """Create performance visualization plot."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Performance Analysis - {result['model_type'].upper()} on {result['dataset'].upper()}", fontsize=16)
        
        # 1. Score comparison
        scores = ['Train', 'Test', 'CV Mean']
        values = [result['train_score'], result['test_score'], result['cv_mean']]
        axes[0, 0].bar(scores, values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 0].set_title('Model Performance Scores')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Dataset info
        dataset_info = [
            f"Samples: {result['training_samples'] + result['test_samples']}",
            f"Features: {result['features']}",
            f"Train/Test: {result['training_samples']}/{result['test_samples']}"
        ]
        if result['is_classification']:
            dataset_info.append(f"Classes: {result['classes']}")
        
        axes[0, 1].text(0.1, 0.5, '\n'.join(dataset_info), 
                       transform=axes[0, 1].transAxes, fontsize=12, verticalalignment='center')
        axes[0, 1].set_title('Dataset Information')
        axes[0, 1].axis('off')
        
        # 3. Performance metrics
        if result['is_classification']:
            metrics_text = f"Accuracy: {result['accuracy']:.3f}\nCV Mean: {result['cv_mean']:.3f}\nCV Std: {result['cv_std']:.3f}"
        else:
            metrics_text = f"MSE: {result['mse']:.3f}\nRMSE: {result['rmse']:.3f}\nCV Mean: {result['cv_mean']:.3f}"
        
        axes[1, 0].text(0.1, 0.5, metrics_text, 
                       transform=axes[1, 0].transAxes, fontsize=12, verticalalignment='center')
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].axis('off')
        
        # 4. Model info
        model_info = f"Model: {result['model_type']}\nDataset: {result['dataset']}\nExperiment: {experiment_id}"
        axes[1, 1].text(0.1, 0.5, model_info, 
                       transform=axes[1, 1].transAxes, fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Experiment Info')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.artifacts_dir}/performance_{experiment_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate file info
        size_bytes = os.path.getsize(plot_path)
        with open(plot_path, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()
        
        return ArtifactInfo(
            name=f"performance_{experiment_id}.png",
            type=ArtifactType.PLOT,
            storage_location=plot_path,
            description=f"Performance visualization for {result['model_type']} experiment",
            service_id="ml-service-1",
            size_bytes=size_bytes,
            checksum=checksum,
            tags=["visualization", "performance", result["model_type"], result["dataset"]],
            metadata={
                "plot_type": "performance_analysis",
                "model_type": result["model_type"],
                "dataset": result["dataset"]
            }
        )
    
    async def _create_experiment_log(self, experiment_id: str, result: Dict[str, Any], config: Dict[str, Any]) -> ArtifactInfo:
        """Create experiment log artifact."""
        log_content = f"""
ML Experiment Log
=================

Experiment ID: {experiment_id}
Timestamp: {datetime.utcnow().isoformat()}
Model Type: {result['model_type']}
Dataset: {result['dataset']}

Configuration:
- Test Size: {config.get('test_size', 0.2)}
- Random State: {config.get('random_state', 42)}
- Parameters: {config.get('parameters', {})}

Results:
- Training Score: {result['train_score']:.4f}
- Test Score: {result['test_score']:.4f}
- CV Mean: {result['cv_mean']:.4f}
- CV Std: {result['cv_std']:.4f}

Dataset Info:
- Total Samples: {result['training_samples'] + result['test_samples']}
- Features: {result['features']}
- Training Samples: {result['training_samples']}
- Test Samples: {result['test_samples']}
- Is Classification: {result['is_classification']}

Model Performance:
"""
        
        if result['is_classification']:
            log_content += f"- Accuracy: {result['accuracy']:.4f}\n"
            log_content += f"- Classes: {result['classes']}\n"
        else:
            log_content += f"- MSE: {result['mse']:.4f}\n"
            log_content += f"- RMSE: {result['rmse']:.4f}\n"
        
        log_content += f"""
Artifacts Generated:
- Model: {result['model_path']}
- Metrics: metrics_{experiment_id}.json
- Plot: performance_{experiment_id}.png
- Log: experiment_log_{experiment_id}.txt

Experiment completed successfully.
"""
        
        # Save log
        log_path = f"{self.artifacts_dir}/experiment_log_{experiment_id}.txt"
        with open(log_path, 'w') as f:
            f.write(log_content)
        
        # Calculate file info
        size_bytes = os.path.getsize(log_path)
        with open(log_path, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()
        
        return ArtifactInfo(
            name=f"experiment_log_{experiment_id}.txt",
            type=ArtifactType.LOG,
            storage_location=log_path,
            description=f"Detailed experiment log for {result['model_type']} experiment",
            service_id="ml-service-1",
            size_bytes=size_bytes,
            checksum=checksum,
            tags=["log", "experiment", result["model_type"], result["dataset"]],
            metadata={
                "log_type": "experiment_summary",
                "model_type": result["model_type"],
                "dataset": result["dataset"]
            }
        )

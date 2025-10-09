# ML Experiment Microservice

A microservice for running machine learning experiments, integrated with the MCP Orchestrator platform.

## Features

- REST API for ML job execution
- Support for scikit-learn models (linear, random forest, neural networks)
- Built-in datasets (iris, wine, breast cancer)
- Artifact generation (models, plots, metrics)
- Integration with MCP Orchestrator

## Quick Start

```bash
pip install -r requirements.txt
python main.py

```

## API Endpoints

- `POST /execute` - Execute ML experiment
- `GET /health` - Health check
- `GET /datasets` - List available datasets
- `GET /models` - List supported model types
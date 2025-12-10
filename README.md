# BAT-OptiCure-MLOps

**OptiCure** is an MLOps demo project for British American Tobacco (BAT) designed to showcase best practices in machine learning operations, from data generation through model training and deployment.

## Project Overview

This project demonstrates:
- **Data Generation**: Synthetic data pipeline for training
- **Model Training**: Machine learning model development and evaluation
- **API Deployment**: Flask-based REST API for model serving
- **CI/CD Integration**: GitHub Actions for automated testing and deployment

## Project Structure

```
BAT-OptiCure-MLOps/
├── .github/
│   └── workflows/
│       └── mlops.yml           # CI/CD pipeline configuration
├── docs/
│   └── HLD.md                  # High Level Design document
├── src/
│   ├── generate_data.py        # Data generation script
│   ├── train.py                # Model training script
│   └── app.py                  # Flask API application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

### Design Doc

- High Level Design: see `docs/HLD.md` for architecture, scope, data flow, CI/CD, and runbook details.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd BAT-OptiCure-MLOps
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Data

```bash
python src/generate_data.py
```

### 5. Train Model

```bash
python src/train.py
```

### 6. Run API Server

```bash
python src/app.py
```

The API will be available at `http://localhost:5000`

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment. The pipeline is configured in `.github/workflows/mlops.yml` and includes:
- Code quality checks
- Unit tests
- Model training and evaluation
- Deployment to staging/production

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Make predictions |
| `/model/info` | GET | Get model information |

## Contributing

1. Create a feature branch
2. Make your changes
3. Commit and push
4. Create a Pull Request

## License

Proprietary - British American Tobacco (BAT)

## Contact

For questions or support, contact the MLOps team.

# ML Data Pipeline

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 14+](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

A **production-ready full-stack platform** for training, deploying, and managing machine learning models with an intuitive web interface.

[Features](#features) • [Quick Start](#quick-start) • [Architecture](#architecture) • [API Documentation](#api-documentation) • [Contributing](#contributing)

</div>

---

## Overview

ML Data Pipeline is an end-to-end machine learning platform designed to streamline the workflow from data ingestion through model deployment. Built with modern technologies, it provides a seamless experience for data scientists and ML engineers to:

- **Upload & explore** datasets (CSV, JSON, Parquet)
- **Preprocess data** with automated cleaning and feature engineering
- **Train models** using various algorithms (Linear Regression, Random Forest, Gradient Boosting, Neural Networks)
- **Evaluate & compare** model performance with interactive visualizations
- **Deploy models** to production with one click
- **Monitor** predictions and model performance in real-time
- **Version control** all models and datasets for reproducibility

---

## Features

### 🎯 Core Capabilities
- ✅ **Multi-format Data Support**: CSV, JSON, Parquet, Excel
- ✅ **Interactive Data Exploration**: Statistical summaries, correlations, distributions
- ✅ **Automated Preprocessing**: Handling missing values, encoding, scaling
- ✅ **Multiple ML Algorithms**: Regression, Classification, Clustering
- ✅ **Hyperparameter Optimization**: Grid search, random search
- ✅ **Model Comparison**: Side-by-side performance metrics
- ✅ **Real-time Predictions**: Batch and single-instance inference
- ✅ **Model Versioning**: Track all model iterations
- ✅ **REST API**: Production-ready endpoints with OpenAPI documentation
- ✅ **Role-based Access Control**: User management and permissions
- ✅ **Experiment Tracking**: Monitor training runs and metrics
- ✅ **Export Functionality**: Download models, predictions, reports

### 🏗️ Technical Stack

**Frontend**
- Next.js 14 with React 18
- TypeScript for type safety
- TailwindCSS for styling
- Recharts for data visualization
- SWR for data fetching
- Zustand for state management

**Backend**
- FastAPI (Python 3.10+)
- Pydantic for data validation
- SQLAlchemy ORM
- Alembic for database migrations
- Celery for async job processing

**Data Science & ML**
- scikit-learn for traditional ML
- TensorFlow/PyTorch for deep learning
- Pandas for data manipulation
- NumPy for numerical computing
- Matplotlib/Seaborn for visualization

**Infrastructure**
- PostgreSQL for data storage
- Redis for caching and job queues
- Docker & Docker Compose
- GitHub Actions for CI/CD
- AWS/GCP ready deployment configs

---

## Quick Start

### Prerequisites
- Docker & Docker Compose (recommended)
- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- Redis 7+

### Installation (Docker - Recommended)

```bash
# Clone the repository
git clone https://github.com/rey344/ml-data-pipeline.git
cd ml-data-pipeline

# Copy environment file
cp .env.example .env

# Start all services
docker-compose up -d

# Initialize database
docker-compose exec api python -m app.commands.init_db

# Access the application
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

### Manual Installation

**Backend Setup**
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env

# Run migrations
alembic upgrade head

# Start FastAPI server
uvicorn app.main:app --reload
```

**Frontend Setup**
```bash
cd frontend

# Install dependencies
npm install

# Set environment variables
cp .env.example .env.local

# Start Next.js development server
npm run dev
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Next.js)                │
│  ├─ Dashboard: Overview & quick stats                       │
│  ├─ Data Management: Upload, explore, preprocess            │
│  ├─ Model Training: Algorithm selection, hyperparameter     │
│  ├─ Evaluation: Metrics, visualizations, comparisons        │
│  └─ Deployment: Model serving, monitoring                   │
└────────────────────┬────────────────────────────────────────┘
                     │ REST API (OpenAPI/Swagger)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   API Layer (FastAPI)                       │
│  ├─ /api/auth: Authentication & authorization               │
│  ├─ /api/datasets: Data management                          │
│  ├─ /api/models: Model CRUD operations                      │
│  ├─ /api/training: Training jobs                            │
│  ├─ /api/predictions: Inference endpoints                   │
│  └─ /api/monitoring: Performance metrics                    │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌─────────┐ ┌──────────┐ ┌─────────┐
   │ Database│ │  Redis   │ │ Celery  │
   │ (PG)    │ │ (Cache)  │ │ (Jobs)  │
   └─────────┘ └──────────┘ └─────────┘
        ▲            │            ▲
        └────────────┼────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   ┌──────────────┐         ┌──────────────┐
   │ Model Storage│         │   Artifacts  │
   │   (S3/GCS)   │         │   (Logs)     │
   └──────────────┘         └──────────────┘
```

---

## Project Structure

```
ml-data-pipeline/
├── backend/                    # FastAPI application
│   ├── app/
│   │   ├── api/               # API route handlers
│   │   ├── models/            # SQLAlchemy models
│   │   ├── schemas/           # Pydantic schemas
│   │   ├── services/          # Business logic
│   │   ├── ml/                # ML pipeline code
│   │   ├── commands/          # CLI commands
│   │   └── main.py            # FastAPI app
│   ├── tests/                 # Unit & integration tests
│   ├── requirements.txt        # Python dependencies
│   └── Dockerfile
│
├── frontend/                   # Next.js application
│   ├── app/                   # Next.js app directory
│   │   ├── (auth)/            # Authentication routes
│   │   ├── (dashboard)/       # Dashboard routes
│   │   ├── api/               # API routes
│   │   └── layout.tsx         # Root layout
│   ├── components/            # Reusable React components
│   ├── hooks/                 # Custom React hooks
│   ├── lib/                   # Utilities & helpers
│   ├── public/                # Static assets
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml          # Multi-container setup
├── .github/
│   └── workflows/             # CI/CD pipelines
└── README.md
```

---

## API Documentation

Full API documentation is available at `/docs` endpoint when the backend is running.

### Example Endpoints

```bash
# Upload dataset
POST /api/datasets
Content-Type: multipart/form-data

# List datasets
GET /api/datasets

# Train model
POST /api/models/train
{"dataset_id": "123", "algorithm": "random_forest", "parameters": {...}}

# Get predictions
POST /api/predictions
{"model_id": "456", "data": [[...], [...]]}

# Model metrics
GET /api/models/{model_id}/metrics
```

See [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) for complete details.

---

## Usage Examples

### Training a Model

```python
import requests

# 1. Upload dataset
with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/datasets',
        files={'file': f},
        data={'name': 'iris_dataset'}
    )
    dataset_id = response.json()['id']

# 2. Train model
response = requests.post(
    'http://localhost:8000/api/models/train',
    json={
        'dataset_id': dataset_id,
        'algorithm': 'random_forest',
        'parameters': {
            'n_estimators': 100,
            'max_depth': 10
        },
        'test_split': 0.2
    }
)
model_id = response.json()['id']

# 3. Get predictions
response = requests.post(
    f'http://localhost:8000/api/models/{model_id}/predict',
    json={'data': [[5.1, 3.5, 1.4, 0.2]]}
)
print(response.json())
```

---

## Development

### Running Tests

```bash
# Backend tests
cd backend
pip install pytest pytest-cov
pytest tests/ -v --cov=app

# Frontend tests
cd frontend
npm test -- --coverage
```

### Code Quality

```bash
# Backend
cd backend
flake8 app/
black app/ --check
mypy app/

# Frontend
cd frontend
npm run lint
npm run format
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "Add new table"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## Deployment

### Docker Deployment

```bash
# Build images
docker-compose build

# Deploy
docker-compose up -d
```

### Cloud Deployment

- **AWS**: See [deployment/aws/README.md](./deployment/aws/)
- **GCP**: See [deployment/gcp/README.md](./deployment/gcp/)
- **Heroku**: See [deployment/heroku/README.md](./deployment/heroku/)

---

## Performance Metrics

- **Model Training**: <5 minutes for typical datasets (< 100k rows)
- **Prediction Latency**: <100ms per instance
- **Throughput**: 1000+ predictions/second with auto-scaling
- **Data Upload**: Supports files up to 5GB
- **Model Storage**: Versioned storage with deduplication

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For detailed contribution guidelines, see [CONTRIBUTING.md](./CONTRIBUTING.md).

---

## Roadmap

- [ ] AutoML capabilities
- [ ] Multi-GPU training support
- [ ] Model explainability (SHAP, LIME)
- [ ] Advanced monitoring & alerting
- [ ] A/B testing framework
- [ ] Integration with MLflow
- [ ] Support for LLMs (fine-tuning, inference)
- [ ] Graph neural networks support
- [ ] Time series forecasting tools
- [ ] Mobile app for predictions

---

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## Support

- 📖 [Documentation](./docs/)
- 🐛 [Report Issues](https://github.com/rey344/ml-data-pipeline/issues)
- 💬 [Discussions](https://github.com/rey344/ml-data-pipeline/discussions)
- 📧 Email: support@example.com

---

## Authors & Contributors

- **rey344** - Initial work - [@rey344](https://github.com/rey344)
See [CONTRIBUTORS.md](./CONTRIBUTORS.md) for all contributors.

---

<div align="center">

**[⬆ back to top](#ml-data-pipeline)**

Made with ❤️ for the data science community

</div>

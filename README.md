
# FluPredict Prototype Repository

This repository contains a minimal prototype scaffold for the FluPredict application specified in your dissertation.
It includes:
- Backend (FastAPI) with a simple predict endpoint and placeholder LSTM + XGBoost training script.
- Frontend (Vite + React minimal) that calls the backend predict endpoint.
- Dockerfile and docker-compose for local development.

## Quickstart (local)

1. Build & run with Docker Compose (recommended):
   ```bash
   docker-compose up --build
   ```
   - Backend available at http://localhost:8000
   - Frontend dev server available at http://localhost:5173

2. Generate placeholder models (if you want to avoid training inside the container):
   ```bash
   cd backend
   python model_train.py
   ```

3. Use the API:
   - POST /predict with JSON `{ "lag1": 1200, "lag2": 1180, "weekofyear": 40 }`

## Notes
- This is a **starter** scaffold. Replace data ingestion and the toy training with your ETL and real datasets.
- For production: add authentication, HTTPS, secrets management, model registry (MLflow), CI/CD, and monitoring as recommended in the dissertation.


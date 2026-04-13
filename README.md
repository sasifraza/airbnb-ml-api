# Airbnb Price Prediction API

## Overview
This project is an end-to-end machine learning API for Airbnb price prediction.

It includes:
- data preprocessing
- model training with scikit-learn
- FastAPI inference endpoint
- Docker containerization
- Azure Container Apps deployment

## Tech Stack
- Python
- scikit-learn
- FastAPI
- Docker
- Azure Container Apps

## Live API
Base URL: `https://airbnb-real-api.purplecoast-442677a6.eastus.azurecontainerapps.io`

Swagger Docs: `https://airbnb-real-api.purplecoast-442677a6.eastus.azurecontainerapps.io/docs`

## API Endpoints

### Health Check
`GET /`

Example response:
```json
{"message": "API is running"}

Prediction Endpoint

POST /predict

Example request:

{
  "minimum_nights": 3,
  "number_of_reviews": 25,
  "reviews_per_month": 1.8,
  "calculated_host_listings_count": 2,
  "availability_365": 180
}

Example response:

{
  "predicted_price": 117.965
}

Workflow

Data → preprocessing → model training → saved model → FastAPI → Docker → Azure Container Apps

Notes
Large model artifacts and raw data are excluded from the repository due to size constraints

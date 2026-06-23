# Deployment Guide

## Build Docker Image

```bash
docker build -t leaf-disease-api .
```

## Run Docker Container

```bash
docker run -p 8000:8000 leaf-disease-api
```

## Health Check

Open:

http://localhost:8000/health

Expected Response:

```json
{
  "status": "healthy"
}
```

## Prediction Endpoint

Open:

http://localhost:8000/docs

Use the POST `/predict` endpoint to upload a leaf image.

Example Response:

```json
{
  "class": "Tomato___healthy",
  "confidence": 81.61,
  "inference_ms": 23.45
}
```

## API Documentation

Swagger UI:

http://localhost:8000/docs

```
```

## API Usage

### Start Server

```bash
uvicorn api.main:app --reload
```

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "healthy"
}
```

### Prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
-F "file=@predict_image/image1.jpg"
```

Example Response:

```json
{
  "class": "Pepper,_bell___Bacterial_spot",
  "confidence": 85.99,
  "inference_ms": 24.31
}
```

### OpenAPI Documentation

```text
http://127.0.0.1:8000/docs
```

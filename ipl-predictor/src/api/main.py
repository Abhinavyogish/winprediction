"""FastAPI service for IPL predictor."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="IPL Predictor API", version="0.1.0")


@app.get("/health")
def health():
    """Health check endpoint."""

    return {"status": "ok"}


@app.post("/predict")
def predict():
    """Stub prediction endpoint."""

    return {"message": "prediction placeholder"}


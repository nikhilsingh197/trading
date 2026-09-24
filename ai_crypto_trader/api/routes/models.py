"""Machine learning models and artifact inspection API routes."""
from __future__ import annotations

import json
from pathlib import Path
from fastapi import APIRouter

router = APIRouter()


@router.get("/models")
async def list_trained_models():
    """Return all discovered trained ML models with validation summaries."""
    models_dir = Path("data/models")
    if not models_dir.exists():
        return {"models": []}

    results = []
    for d in sorted(models_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "training_summary.json").exists():
            with open(d / "training_summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
            results.append({
                "directory": d.name,
                "model_type": summary.get("model_type"),
                "symbol": summary.get("symbol"),
                "timeframe": summary.get("timeframe"),
                "target": summary.get("target"),
                "horizon": summary.get("horizon"),
                "cv_metrics": summary.get("cv_metrics"),
                "test_metrics": summary.get("test_metrics"),
                "top_features": list(summary.get("feature_importances", {}).items())[:5],
                "created_at": summary.get("created_at"),
            })
    return {"models": results}

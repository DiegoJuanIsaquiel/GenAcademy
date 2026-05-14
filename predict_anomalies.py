import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_ml_pipeline import build_feature_store


def load_pickle(path: str):
    return joblib.load(path)


def predict_with_supervised(model, feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in feature_df.columns if c != "timestamp"]
    X = feature_df[feature_cols].copy()
    preds = model.predict(X)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    else:
        scores = preds.astype(float)

    out = feature_df[["timestamp"]].copy()
    out["prediction"] = preds
    out["score"] = scores
    return out


def predict_with_unsupervised(bundle, feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = bundle["feature_columns"]
    X = feature_df[feature_cols].copy()
    transformed = bundle["preprocessor"].transform(X)
    preds = np.where(bundle["model"].predict(transformed) == -1, 1, 0)
    scores = -bundle["model"].score_samples(transformed)

    out = feature_df[["timestamp"]].copy()
    out["prediction"] = preds
    out["score"] = scores
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Caminho do .pkl gerado no treino")
    parser.add_argument("--output", default="predictions.csv", help="Arquivo de saída")
    args = parser.parse_args()

    feature_df = build_feature_store().sort_values("timestamp").reset_index(drop=True)
    artifact = load_pickle(args.model_path)

    if isinstance(artifact, dict) and "preprocessor" in artifact and "model" in artifact:
        result = predict_with_unsupervised(artifact, feature_df)
    else:
        result = predict_with_supervised(artifact, feature_df)

    result.to_csv(args.output, index=False)
    print(json.dumps({"output": args.output, "rows": len(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

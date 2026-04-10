import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier


# ============================================================
# CONFIGURAÇÕES
# ============================================================
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "data-lake")
MLFLOW_CLIENT_TRACKING_URI = os.getenv("MLFLOW_CLIENT_TRACKING_URI", os.getenv("MLFLOW_TRACKING_SERVER_URI", "http://localhost:3000"))
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "cloudtrail-anomaly-detection")

LOCAL_GOLD_BASE = Path(os.getenv("LOCAL_GOLD_BASE", "/app/data-lake/gold"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/artifacts"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))


# ============================================================
# IO E DESCOBERTA DE DADOS
# ============================================================
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="us-east-1",
    )


def list_objects(prefix: str) -> List[str]:
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def read_csv_or_parquet_from_s3(key: str) -> pd.DataFrame:
    s3 = get_s3_client()
    response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    data = response["Body"].read()
    buffer = io.BytesIO(data)
    if key.lower().endswith(".parquet"):
        return pd.read_parquet(buffer)
    if key.lower().endswith(".csv"):
        return pd.read_csv(buffer, sep=None, engine="python")
    raise ValueError(f"Formato não suportado para leitura: {key}")


def read_csv_or_parquet_local(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, sep=None, engine="python")
    raise ValueError(f"Formato não suportado para leitura: {path}")


def find_latest_local_file(domain: str, basename: str) -> Optional[Path]:
    candidates = list(LOCAL_GOLD_BASE.glob(f"{domain}/dt=*/{basename}.parquet"))
    candidates += list(LOCAL_GOLD_BASE.glob(f"{domain}/dt=*/{basename}.csv"))
    if not candidates:
        return None
    return sorted(candidates)[-1]


def find_latest_s3_key(domain: str, basename: str) -> Optional[str]:
    keys = list_objects(f"gold/{domain}/")
    filtered = [
        k for k in keys if k.endswith(f"/{basename}.parquet") or k.endswith(f"/{basename}.csv")
    ]
    if not filtered:
        return None
    return sorted(filtered)[-1]


def load_domain_dataset(domain: str, basename: str) -> pd.DataFrame:
    local_candidate = find_latest_local_file(domain, basename)
    if local_candidate and local_candidate.exists():
        print(f"Lendo localmente: {local_candidate}")
        return read_csv_or_parquet_local(local_candidate)

    s3_candidate = find_latest_s3_key(domain, basename)
    if s3_candidate:
        print(f"Lendo do MinIO: s3://{BUCKET_NAME}/{s3_candidate}")
        return read_csv_or_parquet_from_s3(s3_candidate)

    raise FileNotFoundError(
        f"Dataset não encontrado para domain={domain}, basename={basename}."
    )


# ============================================================
# NORMALIZAÇÃO
# ============================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_boolean(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype(int)
    )


def normalize_datetime(df: pd.DataFrame, possible_cols: List[str]) -> pd.Series:
    for col in possible_cols:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                return parsed
    raise ValueError(f"Nenhuma coluna de data válida encontrada em {possible_cols}")


# ============================================================
# FEATURE STORE TABULAR A PARTIR DA GOLD
# ============================================================
def build_feature_store() -> pd.DataFrame:
    cost = normalize_columns(load_domain_dataset("cost", "cost"))
    perf = normalize_columns(load_domain_dataset("performance", "performance"))
    sec = normalize_columns(load_domain_dataset("security", "security"))

    cost["timestamp"] = normalize_datetime(cost, ["timestamp", "eventTime"])
    perf["timestamp"] = normalize_datetime(perf, ["timestamp", "eventTime"])
    sec["timestamp"] = normalize_datetime(sec, ["timestamp", "eventTime"])

    # ---------------- COST ----------------
    if "service" in cost.columns:
        cost["service"] = cost["service"].fillna("unknown").astype(str).str.lower()
        pivot_requests = cost.pivot_table(
            index="timestamp",
            columns="service",
            values="requests",
            aggfunc="sum",
            fill_value=0,
        )
        pivot_requests.columns = [f"requests_service_{c}" for c in pivot_requests.columns]

        pivot_cost = cost.pivot_table(
            index="timestamp",
            columns="service",
            values="cost",
            aggfunc="sum",
            fill_value=0,
        )
        pivot_cost.columns = [f"cost_service_{c}" for c in pivot_cost.columns]

        cost_agg = cost.groupby("timestamp", as_index=False).agg(
            total_cost=("cost", "sum"),
            total_requests_cost=("requests", "sum"),
        )
        if "unique_events" in cost.columns:
            extra = cost.groupby("timestamp", as_index=False).agg(unique_events_cost=("unique_events", "sum"))
            cost_agg = cost_agg.merge(extra, on="timestamp", how="left")
        cost_agg = cost_agg.merge(pivot_requests.reset_index(), on="timestamp", how="left")
        cost_agg = cost_agg.merge(pivot_cost.reset_index(), on="timestamp", how="left")
    else:
        cost_agg = cost.groupby("timestamp", as_index=False).agg(
            total_cost=("cost", "sum"),
            total_requests_cost=("requests", "sum"),
        )

    # ---------------- PERFORMANCE ----------------
    perf_agg_map = {
        "requests": "sum",
        "unique_users": "sum",
    }
    if "unique_resources" in perf.columns:
        perf_agg_map["unique_resources"] = "sum"
    if "requests_growth_pct" in perf.columns:
        perf_agg_map["requests_growth_pct"] = "mean"
    if "rolling_requests_3" in perf.columns:
        perf_agg_map["rolling_requests_3"] = "mean"

    perf_agg = perf.groupby("timestamp", as_index=False).agg(perf_agg_map)
    perf_agg = perf_agg.rename(columns={
        "requests": "total_requests_perf",
        "unique_users": "total_unique_users",
        "unique_resources": "total_unique_resources",
        "requests_growth_pct": "requests_growth_pct",
        "rolling_requests_3": "rolling_requests_3",
    })

    # ---------------- SECURITY ----------------
    sec_agg_map = {"requests": "sum"}
    if "rare_events_count" in sec.columns:
        sec_agg_map["rare_events_count"] = "sum"
    if "rare_events_ratio_pct" in sec.columns:
        sec_agg_map["rare_events_ratio_pct"] = "mean"

    sec_agg = sec.groupby("timestamp", as_index=False).agg(sec_agg_map).rename(columns={
        "requests": "total_requests_security",
        "rare_events_count": "rare_events_count",
        "rare_events_ratio_pct": "rare_events_ratio_pct",
    })

    if "dominant_event" in sec.columns:
        dominant = (
            sec.groupby("timestamp")["dominant_event"]
            .agg(lambda x: x.dropna().astype(str).mode().iloc[0] if not x.dropna().empty else "unknown")
            .reset_index()
        )
        sec_agg = sec_agg.merge(dominant, on="timestamp", how="left")

    # ---------------- MERGE ----------------
    df = cost_agg.merge(perf_agg, on="timestamp", how="outer")
    df = df.merge(sec_agg, on="timestamp", how="outer")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---------------- FEATURES TEMPORAIS ----------------
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # ---------------- FEATURES DERIVADAS ----------------
    if "total_requests_perf" in df.columns and "rolling_requests_3" in df.columns:
        df["requests_vs_rolling_ratio"] = np.where(
            df["rolling_requests_3"].fillna(0) > 0,
            df["total_requests_perf"].fillna(0) / df["rolling_requests_3"].replace(0, np.nan),
            0,
        )
        df["requests_vs_rolling_ratio"] = df["requests_vs_rolling_ratio"].replace([np.inf, -np.inf], 0).fillna(0)

    if "total_cost" in df.columns and "total_requests_cost" in df.columns:
        df["avg_cost_per_request"] = np.where(
            df["total_requests_cost"].fillna(0) > 0,
            df["total_cost"].fillna(0) / df["total_requests_cost"].replace(0, np.nan),
            0,
        )
        df["avg_cost_per_request"] = df["avg_cost_per_request"].replace([np.inf, -np.inf], 0).fillna(0)

    if "total_requests_security" in df.columns and "total_requests_perf" in df.columns:
        df["security_to_perf_ratio"] = np.where(
            df["total_requests_perf"].fillna(0) > 0,
            df["total_requests_security"].fillna(0) / df["total_requests_perf"].replace(0, np.nan),
            0,
        )
        df["security_to_perf_ratio"] = df["security_to_perf_ratio"].replace([np.inf, -np.inf], 0).fillna(0)

    return df


# ============================================================
# TARGET (RÓTULO FRACO)
# ============================================================
def build_target() -> pd.DataFrame:
    perf = normalize_columns(load_domain_dataset("performance", "performance"))
    sec = normalize_columns(load_domain_dataset("security", "security"))

    perf["timestamp"] = normalize_datetime(perf, ["timestamp", "eventTime"])
    sec["timestamp"] = normalize_datetime(sec, ["timestamp", "eventTime"])

    target_df = pd.DataFrame({"timestamp": sorted(set(perf["timestamp"].dropna()) | set(sec["timestamp"].dropna()))})

    # Regras vindas da gold (rótulo fraco / weak supervision)
    perf_flag = pd.DataFrame({"timestamp": perf["timestamp"].copy()})
    perf_flag["perf_anomaly"] = 0

    if "performance_status" in perf.columns:
        perf_flag["perf_anomaly"] = perf["performance_status"].astype(str).str.lower().isin(
            ["crescimento_abrupto", "degradado", "pico", "anômalo", "anomalo"]
        ).astype(int)
    if "is_peak" in perf.columns:
        perf_flag["perf_anomaly"] = np.maximum(perf_flag["perf_anomaly"], normalize_boolean(perf["is_peak"]))

    perf_flag = perf_flag.groupby("timestamp", as_index=False)["perf_anomaly"].max()

    sec_flag = pd.DataFrame({"timestamp": sec["timestamp"].copy()})
    sec_flag["sec_anomaly"] = 0

    if "risk_level" in sec.columns:
        sec_flag["sec_anomaly"] = sec["risk_level"].astype(str).str.lower().isin(
            ["medio", "médio", "alto", "critico", "crítico"]
        ).astype(int)
    if "is_unusual_access" in sec.columns:
        sec_flag["sec_anomaly"] = np.maximum(sec_flag["sec_anomaly"], normalize_boolean(sec["is_unusual_access"]))
    if "rare_events_ratio_pct" in sec.columns:
        sec_flag["sec_anomaly"] = np.maximum(sec_flag["sec_anomaly"], (pd.to_numeric(sec["rare_events_ratio_pct"], errors="coerce").fillna(0) >= 5).astype(int))

    sec_flag = sec_flag.groupby("timestamp", as_index=False)["sec_anomaly"].max()

    target_df = target_df.merge(perf_flag, on="timestamp", how="left")
    target_df = target_df.merge(sec_flag, on="timestamp", how="left")
    target_df[["perf_anomaly", "sec_anomaly"]] = target_df[["perf_anomaly", "sec_anomaly"]].fillna(0)
    target_df["is_anomaly"] = ((target_df["perf_anomaly"] == 1) | (target_df["sec_anomaly"] == 1)).astype(int)
    return target_df


# ============================================================
# CONJUNTO FINAL
# ============================================================
def build_modeling_dataset() -> pd.DataFrame:
    features_df = build_feature_store()
    target_df = build_target()

    df = features_df.merge(target_df[["timestamp", "is_anomaly"]], on="timestamp", how="left")
    df["is_anomaly"] = df["is_anomaly"].fillna(0).astype(int)

    # Remove colunas que causam vazamento do target
    leakage_cols = {
        "performance_status",
        "is_peak",
        "risk_level",
        "reason",
        "is_unusual_access",
        "anomaly_score",
        "requests_zscore",
        "perf_anomaly",
        "sec_anomaly",
    }
    keep_cols = [c for c in df.columns if c not in leakage_cols]
    df = df[keep_cols]

    # Remove linhas sem sinal suficiente
    signal_cols = [c for c in df.columns if c not in ["timestamp", "dominant_event", "is_anomaly"]]
    df = df.dropna(subset=signal_cols, how="all").reset_index(drop=True)
    return df


# ============================================================
# TREINO / TESTE TEMPORAL
# ============================================================
def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    split_idx = max(1, min(split_idx, len(df) - 1))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# ============================================================
# MODELOS
# ============================================================
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def get_supervised_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=6,
                        max_iter=200,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def train_isolation_forest(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer):
    transformed_train = preprocessor.fit_transform(X_train)
    contamination = max(0.01, min(float(y_train.mean()), 0.3))
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(transformed_train)
    return preprocessor, model


# ============================================================
# AVALIAÇÃO
# ============================================================
def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass
    return metrics


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# EXECUÇÃO COM MLFLOW
# ============================================================
def run_experiments() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_CLIENT_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = build_modeling_dataset()

    train_df, test_df = temporal_train_test_split(df, TEST_SIZE)

    feature_cols = [c for c in df.columns if c not in ["timestamp", "is_anomaly"]]
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df["is_anomaly"].copy()
    y_test = test_df["is_anomaly"].copy()

    preprocessor = build_preprocessor(X_train)
    supervised_models = get_supervised_models(preprocessor)

    run_summary = []

    # ---------------- SUPERVISIONADOS ----------------
    for model_name, pipeline in supervised_models.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("problem_type", "binary_classification_with_weak_labels")
            mlflow.set_tag("dataset_layer", "gold")
            mlflow.set_tag("split_strategy", "temporal_holdout")
            mlflow.set_tag("label_strategy", "weak_supervision_from_gold_rules")

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_rows", len(df))
            mlflow.log_param("train_rows", len(train_df))
            mlflow.log_param("test_rows", len(test_df))
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("positive_rate_train", float(y_train.mean()))
            mlflow.log_param("positive_rate_test", float(y_test.mean()))

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            if hasattr(pipeline, "predict_proba"):
                y_score = pipeline.predict_proba(X_test)[:, 1]
            elif hasattr(pipeline, "decision_function"):
                y_score = pipeline.decision_function(X_test)
            else:
                y_score = None

            metrics = compute_metrics(y_test, y_pred, y_score)
            mlflow.log_metrics(metrics)

            report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
            cm = confusion_matrix(y_test, y_pred).tolist()

            run_dir = OUTPUT_DIR / model_name
            run_dir.mkdir(parents=True, exist_ok=True)

            model_pkl_path = run_dir / f"{model_name}.pkl"
            metadata_path = run_dir / f"{model_name}_metadata.json"
            report_path = run_dir / f"{model_name}_classification_report.json"
            cm_path = run_dir / f"{model_name}_confusion_matrix.json"

            joblib.dump(pipeline, model_pkl_path)
            save_json({
                "model_name": model_name,
                "feature_columns": feature_cols,
                "target": "is_anomaly",
                "problem_type": "binary_classification_with_weak_labels",
            }, metadata_path)
            save_json(report, report_path)
            save_json({"confusion_matrix": cm}, cm_path)

            mlflow.log_artifact(str(model_pkl_path), artifact_path="pickle_model")
            mlflow.log_artifact(str(metadata_path), artifact_path="metadata")
            mlflow.log_artifact(str(report_path), artifact_path="reports")
            mlflow.log_artifact(str(cm_path), artifact_path="reports")
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            run_summary.append({"model": model_name, **metrics})

    # ---------------- NÃO SUPERVISIONADO ----------------
    with mlflow.start_run(run_name="isolation_forest"):
        mlflow.set_tag("problem_type", "unsupervised_anomaly_detection")
        mlflow.set_tag("dataset_layer", "gold")
        mlflow.set_tag("split_strategy", "temporal_holdout")
        mlflow.set_tag("label_strategy", "evaluation_with_weak_labels_only")

        preproc_iso = build_preprocessor(X_train)
        fitted_preproc, iso_model = train_isolation_forest(X_train, y_train, preproc_iso)

        X_test_transformed = fitted_preproc.transform(X_test)
        pred_raw = iso_model.predict(X_test_transformed)
        y_pred = np.where(pred_raw == -1, 1, 0)
        anomaly_score = -iso_model.score_samples(X_test_transformed)

        metrics = compute_metrics(y_test, y_pred, anomaly_score)
        mlflow.log_param("model_name", "isolation_forest")
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("contamination", max(0.01, min(float(y_train.mean()), 0.3)))
        mlflow.log_metrics(metrics)

        run_dir = OUTPUT_DIR / "isolation_forest"
        run_dir.mkdir(parents=True, exist_ok=True)

        artifact_bundle = {
            "preprocessor": fitted_preproc,
            "model": iso_model,
            "feature_columns": feature_cols,
            "target": "is_anomaly",
            "problem_type": "unsupervised_anomaly_detection",
        }
        model_pkl_path = run_dir / "isolation_forest.pkl"
        report_path = run_dir / "isolation_forest_classification_report.json"
        cm_path = run_dir / "isolation_forest_confusion_matrix.json"

        joblib.dump(artifact_bundle, model_pkl_path)
        save_json(classification_report(y_test, y_pred, zero_division=0, output_dict=True), report_path)
        save_json({"confusion_matrix": confusion_matrix(y_test, y_pred).tolist()}, cm_path)

        mlflow.log_artifact(str(model_pkl_path), artifact_path="pickle_model")
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(cm_path), artifact_path="reports")

        run_summary.append({"model": "isolation_forest", **metrics})

    summary_path = OUTPUT_DIR / "run_summary.csv"
    pd.DataFrame(run_summary).sort_values("f1", ascending=False).to_csv(summary_path, index=False)
    print("\nResumo dos experimentos:")
    print(pd.read_csv(summary_path).sort_values("f1", ascending=False).to_string(index=False))
    print(f"\nArquivo resumo salvo em: {summary_path}")


if __name__ == "__main__":
    run_experiments()

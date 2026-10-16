"""
Anomaly Detection Pipeline - Base Version
-----------------------------------------
A simple, modular pipeline with four stages:

    1. Ingest      -> load data from CSV or generate synthetic data
    2. Preprocess  -> clean and scale features
    3. Detect      -> run multiple detectors (Z-Score, IQR, Isolation Forest)
    4. Report      -> combine votes, flag anomalies, save results

Usage:
    python anomaly_detection_pipeline.py                  # synthetic demo data
    python anomaly_detection_pipeline.py --input data.csv # your own CSV
"""

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("anomaly-pipeline")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    contamination: float = 0.02   # expected fraction of anomalies (Isolation Forest)
    min_votes: int = 2            # detectors that must agree to flag a row
    random_state: int = 42


# ---------------------------------------------------------------------------
# Stage 1: Ingest
# ---------------------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    log.info("Loading data from %s", path)
    return pd.read_csv(path)


def generate_synthetic_data(n_rows: int = 1000, n_anomalies: int = 20, seed: int = 42) -> pd.DataFrame:
    """Simulate server metrics with a few injected anomalies."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "cpu_usage": rng.normal(45, 8, n_rows),
        "memory_usage": rng.normal(60, 10, n_rows),
        "latency_ms": rng.normal(120, 20, n_rows),
        "error_rate": np.abs(rng.normal(1.0, 0.4, n_rows)),
    })
    idx = rng.choice(n_rows, n_anomalies, replace=False)
    df.loc[idx, "cpu_usage"] += rng.uniform(35, 50, n_anomalies)
    df.loc[idx, "latency_ms"] += rng.uniform(250, 400, n_anomalies)
    df.loc[idx, "error_rate"] += rng.uniform(5, 10, n_anomalies)
    df["is_injected"] = False
    df.loc[idx, "is_injected"] = True
    log.info("Generated %d rows with %d injected anomalies", n_rows, n_anomalies)
    return df


# ---------------------------------------------------------------------------
# Stage 2: Preprocess
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Fill missing values with the median and standardize features."""
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    scaled = StandardScaler().fit_transform(X)
    return pd.DataFrame(scaled, columns=feature_cols, index=df.index)


# ---------------------------------------------------------------------------
# Stage 3: Detectors (each returns a boolean Series: True = anomaly)
# ---------------------------------------------------------------------------
def detect_zscore(X: pd.DataFrame, cfg: PipelineConfig) -> pd.Series:
    return (X.abs() > cfg.zscore_threshold).any(axis=1)


def detect_iqr(X: pd.DataFrame, cfg: PipelineConfig) -> pd.Series:
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - cfg.iqr_multiplier * iqr, q3 + cfg.iqr_multiplier * iqr
    return ((X < lower) | (X > upper)).any(axis=1)


def detect_isolation_forest(X: pd.DataFrame, cfg: PipelineConfig) -> pd.Series:
    model = IsolationForest(contamination=cfg.contamination, random_state=cfg.random_state)
    preds = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    return pd.Series(preds == -1, index=X.index)


DETECTORS = {
    "zscore": detect_zscore,
    "iqr": detect_iqr,
    "isolation_forest": detect_isolation_forest,
}


# ---------------------------------------------------------------------------
# Stage 4: Combine + Report
# ---------------------------------------------------------------------------
def run_detectors(X: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    results = pd.DataFrame(index=X.index)
    for name, fn in DETECTORS.items():
        results[name] = fn(X, cfg)
        log.info("%-18s flagged %d rows", name, int(results[name].sum()))
    results["votes"] = results[list(DETECTORS)].sum(axis=1)
    results["is_anomaly"] = results["votes"] >= cfg.min_votes
    return results


def report(df: pd.DataFrame, results: pd.DataFrame, output_path: str) -> pd.DataFrame:
    final = pd.concat([df, results], axis=1)
    anomalies = final[final["is_anomaly"]]
    log.info("Total rows: %d | Anomalies flagged: %d (%.2f%%)",
             len(final), len(anomalies), 100 * len(anomalies) / len(final))

    if "is_injected" in final.columns:  # only present in synthetic demo
        tp = int((final["is_anomaly"] & final["is_injected"]).sum())
        fp = int((final["is_anomaly"] & ~final["is_injected"]).sum())
        fn = int((~final["is_anomaly"] & final["is_injected"]).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        log.info("Evaluation -> precision: %.2f | recall: %.2f", precision, recall)

    final.to_csv(output_path, index=False)
    log.info("Results saved to %s", output_path)
    return anomalies


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def run_pipeline(input_path=None, output_path="anomaly_results.csv", cfg=None) -> pd.DataFrame:
    cfg = cfg or PipelineConfig()
    df = load_csv(input_path) if input_path else generate_synthetic_data()

    feature_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "is_injected"]
    log.info("Using features: %s", feature_cols)

    X = preprocess(df, feature_cols)
    results = run_detectors(X, cfg)
    return report(df, results, output_path)


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection Pipeline")
    parser.add_argument("--input", help="Path to input CSV (numeric columns are used)")
    parser.add_argument("--output", default="anomaly_results.csv", help="Path for results CSV")
    parser.add_argument("--min-votes", type=int, default=2, help="Detectors that must agree (1-3)")
    args = parser.parse_args()

    cfg = PipelineConfig(min_votes=args.min_votes)
    anomalies = run_pipeline(args.input, args.output, cfg)
    print("\nTop flagged rows:")
    print(anomalies.sort_values("votes", ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()

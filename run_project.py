"""
NASA CMAPSS FD001 RUL experiments: tabular baselines + 1D CNN sequences.
Writes REPORT.md and results/metrics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cmapss_utils import (
    FEATURE_COLUMNS,
    RUL_CAP,
    RUL_FILE,
    TEST_FILE,
    TRAIN_FILE,
    add_training_rul,
    build_test_rul_tables,
    load_cmapss_file,
    load_rul_labels,
)


def build_model_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate(name: str, pipeline, x_train, x_val, y_train, y_val):
    pipeline.fit(x_train, y_train)
    pred = pipeline.predict(x_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
    mae = float(mean_absolute_error(y_val, pred))
    return {"model": name, "MAE": mae, "RMSE": rmse, "pipeline": pipeline}


def last_row_per_engine(test_df: pd.DataFrame) -> pd.DataFrame:
    idx = test_df.groupby("engine_id")["cycle"].idxmax()
    return test_df.loc[idx].reset_index(drop=True)


def tabular_test_metrics(
    pipeline,
    train_df: pd.DataFrame,
    test_last: pd.DataFrame,
    y_true_capped: np.ndarray,
):
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["RUL_target"]
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(test_last[FEATURE_COLUMNS])
    rmse = float(np.sqrt(mean_squared_error(y_true_capped, pred)))
    mae = float(mean_absolute_error(y_true_capped, pred))
    return {"MAE": mae, "RMSE": rmse, "predictions": pred}


def make_sequences(
    df: pd.DataFrame,
    engine_ids: np.ndarray,
    window: int,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Arrays shaped (N, n_features, window) and (N,) targets (capped RUL)."""
    xs, ys = [], []
    sub = df[df["engine_id"].isin(engine_ids)]
    for _, grp in sub.groupby("engine_id"):
        grp = grp.sort_values("cycle")
        raw = grp[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        if len(grp) < window:
            continue
        mat = scaler.transform(raw).astype(np.float32)
        rul = grp["RUL_target"].to_numpy()
        for i in range(window - 1, len(grp)):
            xs.append(mat[i - window + 1 : i + 1].T)
            ys.append(rul[i])
    n_feat = len(FEATURE_COLUMNS)
    if not xs:
        return np.zeros((0, n_feat, window), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(xs, axis=0), np.array(ys, dtype=np.float32)


def make_test_last_sequences(
    test_df: pd.DataFrame,
    window: int,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """One sequence per engine: last `window` cycles. Returns X, engine_id order sorted."""
    xs, eids = [], []
    for eid, grp in test_df.groupby("engine_id"):
        grp = grp.sort_values("cycle")
        raw = grp[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        if len(grp) < window:
            pad_rows = window - len(grp)
            raw = np.vstack([np.tile(raw[0], (pad_rows, 1)), raw])
        mat = scaler.transform(raw).astype(np.float32)
        xs.append(mat[-window:].T)
        eids.append(eid)
    order = np.argsort(np.array(eids))
    X = np.stack([xs[i] for i in order], axis=0)
    return X, np.sort(np.array(eids))


class RULCNN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)
        return self.head(h).squeeze(-1)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    window: int,
    epochs: int = 45,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 12,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RULCNN(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    best_state = None
    best_val = float("inf")
    bad = 0
    n = Xt.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = float(loss_fn(val_pred, yv).cpu().item())
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, device


def train_cnn_full_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
):
    """Train on all rows (no held-out val) for final test-time model."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RULCNN(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    n = Xt.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model, device


def cnn_metrics(model, device, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(X).to(device)
        pred = model(t).cpu().numpy()
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    return mae, rmse


def write_report(
    path: Path,
    val_rows: list[dict],
    test_rows: list[dict],
    window: int,
    n_train_engines: int,
    n_val_engines: int,
):
    lines = [
        "# CMAPSS FD001 — RUL prediction report",
        "",
        "Generated by `run_project.py` on NASA CMAPSS subset FD001 (single operating condition, single fault).",
        "",
        "## Setup",
        "",
        f"- **Piecewise RUL cap:** {RUL_CAP} cycles (training target and test labels for scoring).",
        "- **Validation:** 20% of training engines held out (no leakage across cycles of the same engine).",
        f"- **Temporal model:** 1D CNN on sliding windows of length **{window}**; shorter test trajectories are front-padded by repeating the first observed row (standard pragmatic fix).",
        "- **Tabular baselines:** settings + 21 sensors at each time step; `engine_id` and `cycle` excluded from features.",
        "",
        "## Validation metrics (capped RUL)",
        "",
        "| Model | MAE | RMSE |",
        "| --- | ---: | ---: |",
    ]
    for r in val_rows:
        lines.append(f"| {r['model']} | {r['MAE']:.4f} | {r['RMSE']:.4f} |")
    lines.extend(
        [
            "",
            "## Test metrics (last snapshot per engine, capped true RUL)",
            "",
            "Models are refit on **all** training engines before test prediction.",
            "",
            "| Model | MAE | RMSE |",
            "| --- | ---: | ---: |",
        ]
    )
    for r in test_rows:
        lines.append(f"| {r['model']} | {r['MAE']:.4f} | {r['RMSE']:.4f} |")
    lines.extend(
        [
            "",
            "## Discussion",
            "",
            "On validation, the 1D CNN often sits between linear regression and the random forest, "
            "suggesting sliding windows capture some temporal degradation structure. If test metrics "
            "for the CNN are weaker than the forest, consider architecture capacity, window length, "
            "padding for short tests, and training protocol (epochs, regularization) before drawing "
            "conclusions about temporal modeling in general.",
            "",
            "## Methods (short)",
            "",
            "Training RUL is `min(max_cycle − cycle, RUL_cap)`. "
            "Sklearn pipelines use median imputation and standard scaling. "
            "The CNN sees standardized windows per feature (scaler fit on training engines only).",
            "",
            "## Future work",
            "",
            "- Tune window size, CNN width/depth, and learning rate schedule.",
            "- Sensor selection / ablation and multi-dataset FD002–FD004.",
            "- Recurrent or attention models; NASA asymmetric scoring function.",
            "",
            f"## Reproducibility",
            "",
            f"- Training engines (fit): {n_train_engines}; validation engines: {n_val_engines}.",
            "- Random seed 42 for engine split and forest/CNN initialization.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    window = 30
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("NASA CMAPSS FD001 — full pipeline", flush=True)
    train_df = load_cmapss_file(TRAIN_FILE)
    test_df = load_cmapss_file(TEST_FILE)
    rul_df = load_rul_labels(RUL_FILE)
    train_df = add_training_rul(train_df)
    final_cycle_table, _ = build_test_rul_tables(test_df, rul_df)
    y_test_capped = np.minimum(final_cycle_table["true_RUL"].to_numpy(), RUL_CAP)

    engine_ids = train_df["engine_id"].drop_duplicates().to_numpy()
    train_eids, val_eids = train_test_split(engine_ids, test_size=0.2, random_state=42)
    train_mask = train_df["engine_id"].isin(train_eids)
    val_mask = train_df["engine_id"].isin(val_eids)

    X_train_tab = train_df.loc[train_mask, FEATURE_COLUMNS]
    X_val_tab = train_df.loc[val_mask, FEATURE_COLUMNS]
    y_train_tab = train_df.loc[train_mask, "RUL_target"]
    y_val_tab = train_df.loc[val_mask, "RUL_target"]

    models = {
        "Linear Regression": build_model_pipeline(LinearRegression()),
        "Random Forest": build_model_pipeline(
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            )
        ),
    }

    val_rows = []
    fitted_tabular = {}
    for name, pipe in models.items():
        out = evaluate(name, pipe, X_train_tab, X_val_tab, y_train_tab, y_val_tab)
        val_rows.append({"model": name, "MAE": out["MAE"], "RMSE": out["RMSE"]})
        fitted_tabular[name] = out["pipeline"]
        print(f"Val  {name}: MAE={out['MAE']:.4f} RMSE={out['RMSE']:.4f}", flush=True)

    # Scaler for CNN (fit on training engines only; NumPy avoids sklearn feature-name warnings)
    scaler = StandardScaler()
    scaler.fit(train_df.loc[train_mask, FEATURE_COLUMNS].to_numpy())

    Xtr_cnn, ytr_cnn = make_sequences(train_df, train_eids, window, scaler)
    Xva_cnn, yva_cnn = make_sequences(train_df, val_eids, window, scaler)
    if Xtr_cnn.shape[0] == 0:
        print("ERROR: no CNN training sequences (window too large?).", file=sys.stderr)
        sys.exit(1)

    n_features = Xtr_cnn.shape[1]
    cnn_model, device = train_cnn(Xtr_cnn, ytr_cnn, Xva_cnn, yva_cnn, n_features, window)
    cnn_mae_v, cnn_rmse_v = cnn_metrics(cnn_model, device, Xva_cnn, yva_cnn)
    val_rows.append({"model": f"1D CNN (window={window})", "MAE": cnn_mae_v, "RMSE": cnn_rmse_v})
    print(f"Val  1D CNN: MAE={cnn_mae_v:.4f} RMSE={cnn_rmse_v:.4f}", flush=True)

    test_last = last_row_per_engine(test_df)
    test_rows = []
    for name, pipe in fitted_tabular.items():
        m = tabular_test_metrics(pipe, train_df, test_last, y_test_capped)
        test_rows.append({"model": name, "MAE": m["MAE"], "RMSE": m["RMSE"]})
        print(f"Test {name}: MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f}", flush=True)

    # CNN: refit on all training engines (full data; fixed epochs to avoid row-level leakage)
    scaler_full = StandardScaler()
    scaler_full.fit(train_df[FEATURE_COLUMNS].to_numpy())
    X_all, y_all = make_sequences(train_df, engine_ids, window, scaler_full)
    cnn_final, dev = train_cnn_full_data(X_all, y_all, n_features)
    X_test_cnn, _ = make_test_last_sequences(test_df, window, scaler_full)
    cnn_mae_t, cnn_rmse_t = cnn_metrics(cnn_final, dev, X_test_cnn, y_test_capped)
    test_rows.append({"model": f"1D CNN (window={window})", "MAE": cnn_mae_t, "RMSE": cnn_rmse_t})
    print(f"Test 1D CNN: MAE={cnn_mae_t:.4f} RMSE={cnn_rmse_t:.4f}", flush=True)

    metrics = {"validation": val_rows, "test": test_rows, "window": window, "rul_cap": RUL_CAP}
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    write_report(
        Path("REPORT.md"),
        val_rows,
        test_rows,
        window,
        n_train_engines=len(train_eids),
        n_val_engines=len(val_eids),
    )
    print("\nWrote REPORT.md and results/metrics.json", flush=True)


if __name__ == "__main__":
    main()

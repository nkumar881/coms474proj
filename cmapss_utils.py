"""NASA CMAPSS FD001 loading and RUL label helpers."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("CMAPSSData")
TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"

# Common piecewise-linear RUL cap for FD001 (see PHM08 / literature).
RUL_CAP = 125

FEATURE_COLUMNS = [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]


def build_column_names():
    base_columns = ["engine_id", "cycle"]
    setting_columns = [f"setting_{i}" for i in range(1, 4)]
    sensor_columns = [f"sensor_{i}" for i in range(1, 22)]
    return base_columns + setting_columns + sensor_columns


def load_cmapss_file(file_path: Path) -> pd.DataFrame:
    """Load a CMAPSS text file and drop empty columns caused by trailing spaces."""
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")

    expected_columns = len(build_column_names())
    if df.shape[1] != expected_columns:
        raise ValueError(
            f"{file_path} has {df.shape[1]} columns after cleanup; "
            f"expected {expected_columns}."
        )

    df.columns = build_column_names()
    return df


def add_training_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute RUL for every training row as max_cycle(engine) - current_cycle."""
    train_df = train_df.copy()
    max_cycle_per_engine = train_df.groupby("engine_id")["cycle"].transform("max")
    train_df["RUL"] = max_cycle_per_engine - train_df["cycle"]
    train_df["RUL_target"] = np.minimum(train_df["RUL"].to_numpy(), RUL_CAP)
    return train_df


def build_test_rul_tables(test_df: pd.DataFrame, rul_df: pd.DataFrame):
    """Align one RUL value per test engine; derive row-level RUL for diagnostics."""
    final_cycle_table = (
        test_df.groupby("engine_id", as_index=False)["cycle"]
        .max()
        .rename(columns={"cycle": "final_observed_cycle"})
        .sort_values("engine_id")
        .reset_index(drop=True)
    )

    final_cycle_table["true_RUL"] = rul_df["true_RUL"].to_numpy()
    final_cycle_table["failure_cycle"] = (
        final_cycle_table["final_observed_cycle"] + final_cycle_table["true_RUL"]
    )

    test_with_rul = test_df.merge(
        final_cycle_table[["engine_id", "final_observed_cycle", "true_RUL", "failure_cycle"]],
        on="engine_id",
        how="left",
    )
    test_with_rul["RUL_if_labeled"] = test_with_rul["failure_cycle"] - test_with_rul["cycle"]
    return final_cycle_table, test_with_rul


def load_rul_labels(rul_path: Path) -> pd.DataFrame:
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")
    rul_df.columns = ["true_RUL"]
    return rul_df

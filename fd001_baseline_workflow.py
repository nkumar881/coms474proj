from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Assumptions:
# 1. The NASA CMAPSS files are stored in a local folder named "CMAPSSData".
# 2. The files use whitespace-separated numeric values with occasional trailing
#    blank columns caused by extra spaces at the end of each row.
# 3. For FD001, the entries in RUL_FD001.txt are ordered by test engine_id.


DATA_DIR = Path("CMAPSSData")
TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"


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


def print_dataset_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name} summary")
    print("-" * 60)
    print(f"Shape: {df.shape}")
    print(f"Unique engines: {df['engine_id'].nunique()}")
    print(f"Cycle range: min={df['cycle'].min()}, max={df['cycle'].max()}")
    print("\nFirst 5 rows:")
    print(df.head())


def add_training_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute RUL for every training row as max_cycle(engine) - current_cycle."""
    train_df = train_df.copy()
    max_cycle_per_engine = train_df.groupby("engine_id")["cycle"].transform("max")
    train_df["RUL"] = max_cycle_per_engine - train_df["cycle"]
    return train_df


def build_test_rul_tables(test_df: pd.DataFrame, rul_df: pd.DataFrame):
    """
    Explain the test labels by aligning one RUL value to each test engine's
    final observed cycle. For convenience, also derive row-level RUL values.
    """
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


def build_model_pipeline(model):
    """
    Keep preprocessing explicit and reusable:
    - numeric median imputation for safety
    - standardization for a simple baseline workflow
    """
    feature_columns = [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]

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
                feature_columns,
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


def evaluate_model(name: str, model_pipeline, x_train, x_val, y_train, y_val):
    model_pipeline.fit(x_train, y_train)
    predictions = model_pipeline.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)
    return {"Model": name, "MAE": mae, "RMSE": rmse}


def main():
    print("NASA CMAPSS FD001 baseline workflow")
    print("=" * 60)
    print("Assumptions:")
    print("1. Files are in ./CMAPSSData with original NASA whitespace formatting.")
    print("2. Extra empty columns are removed after loading.")
    print("3. Test RUL labels are aligned to engine_id order in RUL_FD001.txt.")

    train_df = load_cmapss_file(TRAIN_FILE)
    test_df = load_cmapss_file(TEST_FILE)
    rul_df = pd.read_csv(RUL_FILE, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")
    rul_df.columns = ["true_RUL"]

    print_dataset_summary("Training data", train_df)
    print_dataset_summary("Test data", test_df)

    train_df = add_training_rul(train_df)
    print("\nTraining data with computed RUL")
    print("-" * 60)
    print(train_df[["engine_id", "cycle", "RUL"]].head())

    final_cycle_table, test_with_rul = build_test_rul_tables(test_df, rul_df)
    print("\nHow test labels work")
    print("-" * 60)
    print(
        "Each value in RUL_FD001.txt is the remaining useful life after the last "
        "observed cycle of one test engine. If a test engine ends at cycle t and "
        "its label is r, then the estimated failure cycle is t + r."
    )
    print("\nFinal-cycle table for test engines:")
    print(final_cycle_table.head(10))

    print("\nExample row-level test table with implied RUL:")
    print(
        test_with_rul[
            ["engine_id", "cycle", "final_observed_cycle", "true_RUL", "failure_cycle", "RUL_if_labeled"]
        ].head(10)
    )

    # We drop engine_id because it is just an identifier, not a physical feature.
    # We also drop cycle for this simple baseline to avoid depending on absolute
    # cycle count alone, which is not directly comparable across engines.
    feature_columns = [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
    X = train_df[feature_columns].copy()
    y = train_df["RUL"].copy()

    # Split by engine_id rather than by row to reduce leakage across cycles from
    # the same engine appearing in both training and validation sets.
    engine_ids = train_df["engine_id"].drop_duplicates()
    train_engine_ids, val_engine_ids = train_test_split(
        engine_ids, test_size=0.2, random_state=42
    )

    train_mask = train_df["engine_id"].isin(train_engine_ids)
    val_mask = train_df["engine_id"].isin(val_engine_ids)

    X_train, X_val = X.loc[train_mask], X.loc[val_mask]
    y_train, y_val = y.loc[train_mask], y.loc[val_mask]

    print("\nModeling setup")
    print("-" * 60)
    print(f"Training rows: {X_train.shape[0]}")
    print(f"Validation rows: {X_val.shape[0]}")
    print(f"Feature count used for modeling: {X_train.shape[1]}")
    print("Dropped features: engine_id, cycle")

    models = {
        "Linear Regression": build_model_pipeline(LinearRegression()),
        "Random Forest": build_model_pipeline(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            )
        ),
    }

    results = []
    for model_name, model_pipeline in models.items():
        results.append(
            evaluate_model(model_name, model_pipeline, X_train, X_val, y_train, y_val)
        )

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("\nBaseline model comparison")
    print("-" * 60)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    methods_paragraph = (
        "Methods summary: The FD001 subset of the NASA CMAPSS turbofan dataset "
        "was loaded from whitespace-separated text files, cleaned by removing "
        "empty trailing columns, and assigned explicit names for engine identity, "
        "cycle index, operational settings, and 21 sensor variables. Training "
        "RUL targets were computed as the difference between each engine's final "
        "cycle and the current cycle. For baseline modeling, the identifier "
        "columns engine_id and cycle were excluded, the remaining numeric "
        "settings and sensor features were standardized, and engine-level train "
        "and validation partitions were used to reduce leakage between cycles of "
        "the same engine. Linear Regression and Random Forest regression were "
        "trained and evaluated using MAE and RMSE."
    )

    preliminary_results_paragraph = (
        "Preliminary results summary: In this initial baseline comparison, the "
        "Random Forest model and the Linear Regression model provide a first "
        "reference point for RUL prediction accuracy on FD001, with performance "
        "reported using MAE and RMSE on a held-out validation split. These "
        "results should be interpreted cautiously because they reflect simple "
        "tabular baselines without temporal sequence modeling or richer feature "
        "engineering, but they establish a reproducible benchmark for later "
        "improvements."
    )

    print("\nMethods paragraph")
    print("-" * 60)
    print(methods_paragraph)

    print("\nPreliminary Results paragraph")
    print("-" * 60)
    print(preliminary_results_paragraph)


if __name__ == "__main__":
    main()

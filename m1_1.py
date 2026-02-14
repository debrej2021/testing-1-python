from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


LOG = logging.getLogger("csv_cleaner")


EXPECTED_COLUMNS = ["SizeSqFt", "Bedrooms", "AgeYears", "Price"]


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_path: Path
    drop_duplicates: bool
    verbose: bool
    random_seed: int = 42
    test_size: float = 0.2


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def parse_args(argv: list[str]) -> Config:
    p = argparse.ArgumentParser(
        description="Validate + clean house_prices.csv and export a cleaned CSV."
    )
    p.add_argument("--input", required=True, help="Path to input CSV")
    p.add_argument(
        "--output",
        default="output/house_prices_cleaned.csv",
        help="Path to write cleaned CSV (default: output/house_prices_cleaned.csv)",
    )
    p.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Drop duplicate rows",
    )
    p.add_argument("--verbose", action="store_true", help="Enable debug logs")
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for train/test split (default: 42)"
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )

    args = p.parse_args(argv)
    configure_logging(args.verbose)

    return Config(
        input_path=Path(args.input),
        output_path=Path(args.output),
        drop_duplicates=bool(args.drop_duplicates),
        verbose=bool(args.verbose),
        random_seed=args.seed,
        test_size=args.test_size,
    )


def validate_input_file(cfg: Config) -> None:
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.input_path}")

    if cfg.input_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Only .csv files are supported, got '{cfg.input_path.suffix}'"
        )

    if cfg.input_path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {cfg.input_path}")


def load_csv(cfg: Config) -> pd.DataFrame:
    LOG.info("Reading CSV: %s", cfg.input_path)
    df = pd.read_csv(cfg.input_path)
    LOG.info("Loaded rows=%d cols=%d", len(df), len(df.columns))
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if extra:
        LOG.warning("Extra columns found (will keep them): %s", extra)


def clean_data(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Keep expected columns in consistent order
    df = df.copy()
    df = df[EXPECTED_COLUMNS]

    # Ensure numeric types
    for col in EXPECTED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with nulls after conversion
    before = len(df)
    df = df.dropna()
    after = len(df)
    if after != before:
        LOG.warning("Dropped %d rows due to invalid numeric values", before - after)

    # Filter rows that violate constraints (instead of raising)
    bad_size = df["SizeSqFt"] <= 0
    if bad_size.any():
        LOG.warning("Filtered %d rows with SizeSqFt <= 0", bad_size.sum())
        df = df[~bad_size]

    bad_bed = df["Bedrooms"] <= 0
    if bad_bed.any():
        LOG.warning("Filtered %d rows with Bedrooms <= 0", bad_bed.sum())
        df = df[~bad_bed]

    bad_age = df["AgeYears"] < 0
    if bad_age.any():
        LOG.warning("Filtered %d rows with AgeYears < 0", bad_age.sum())
        df = df[~bad_age]

    bad_price = df["Price"] <= 0
    if bad_price.any():
        LOG.warning("Filtered %d rows with Price <= 0", bad_price.sum())
        df = df[~bad_price]

    if df.empty:
        raise ValueError("No valid rows remain after cleaning")

    if cfg.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        LOG.info("Dropped %d duplicate rows", before - len(df))

    # Sort for deterministic output
    df = df.sort_values(EXPECTED_COLUMNS).reset_index(drop=True)

    return df


def write_cleaned_csv(df: pd.DataFrame, cfg: Config) -> None:
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_path, index=False)
    LOG.info("Wrote cleaned CSV: %s", cfg.output_path)


def split_data(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.random_seed
    )
    stem = cfg.output_path.stem
    parent = cfg.output_path.parent

    train_path = parent / f"{stem}_train.csv"
    test_path = parent / f"{stem}_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    LOG.info(
        "Split: train=%d rows (%s), test=%d rows (%s)",
        len(train_df), train_path, len(test_df), test_path,
    )
    return train_df, test_df


def evaluate_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    features = ["SizeSqFt", "Bedrooms", "AgeYears"]
    target = "Price"

    model = LinearRegression()
    model.fit(train_df[features], train_df[target])

    predictions = model.predict(test_df[features])
    r2 = r2_score(test_df[target], predictions)
    mae = mean_absolute_error(test_df[target], predictions)
    rmse = np.sqrt(mean_squared_error(test_df[target], predictions))

    LOG.info("Model metrics — R²: %.4f, MAE: %.2f, RMSE: %.2f", r2, mae, rmse)
    return {"r2": r2, "mae": mae, "rmse": rmse}


def main(argv: list[str]) -> int:
    try:
        cfg = parse_args(argv)
        validate_input_file(cfg)

        df = load_csv(cfg)
        validate_schema(df)

        cleaned = clean_data(df, cfg)
        write_cleaned_csv(cleaned, cfg)

        train_df, test_df = split_data(cleaned, cfg)
        evaluate_model(train_df, test_df)

        LOG.info("Done. Clean rows=%d", len(cleaned))
        return 0
    except Exception as ex:
        LOG.error("Failed: %s", ex, exc_info=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

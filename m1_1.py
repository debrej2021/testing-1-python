from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LOG = logging.getLogger("csv_cleaner")


EXPECTED_COLUMNS = ["SizeSqFt", "Bedrooms", "AgeYears", "Price"]


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_path: Path
    drop_duplicates: bool
    verbose: bool


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

    args = p.parse_args(argv)
    configure_logging(args.verbose)

    return Config(
        input_path=Path(args.input),
        output_path=Path(args.output),
        drop_duplicates=bool(args.drop_duplicates),
        verbose=bool(args.verbose),
    )


def validate_input_file(cfg: Config) -> None:
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.input_path}")

    if cfg.input_path.suffix.lower() != ".csv":
        raise ValueError("Only .csv files are supported.")


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

    # Basic sanity checks
    if (df["SizeSqFt"] <= 0).any():
        raise ValueError("Invalid data: SizeSqFt must be > 0")
    if (df["Bedrooms"] <= 0).any():
        raise ValueError("Invalid data: Bedrooms must be > 0")
    if (df["AgeYears"] < 0).any():
        raise ValueError("Invalid data: AgeYears must be >= 0")
    if (df["Price"] <= 0).any():
        raise ValueError("Invalid data: Price must be > 0")

    if cfg.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        LOG.info("Dropped %d duplicate rows", before - len(df))

    return df


def write_cleaned_csv(df: pd.DataFrame, cfg: Config) -> None:
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_path, index=False)
    LOG.info("Wrote cleaned CSV: %s", cfg.output_path)


def main(argv: list[str]) -> int:
    try:
        cfg = parse_args(argv)
        validate_input_file(cfg)

        df = load_csv(cfg)
        validate_schema(df)

        cleaned = clean_data(df, cfg)
        write_cleaned_csv(cleaned, cfg)

        LOG.info("Done. Clean rows=%d", len(cleaned))
        return 0
    except Exception as ex:
        LOG.error("Failed: %s", ex, exc_info=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

from __future__ import annotations

import pandas as pd
import pytest

from m1_1 import (
    EXPECTED_COLUMNS,
    Config,
    clean_data,
    evaluate_model,
    parse_args,
    split_data,
    validate_input_file,
    validate_schema,
    write_cleaned_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path, **overrides):
    defaults = {
        "input_path": tmp_path / "input.csv",
        "output_path": tmp_path / "output.csv",
        "drop_duplicates": False,
        "verbose": False,
        "random_seed": 42,
        "test_size": 0.2,
    }
    defaults.update(overrides)
    return Config(**defaults)


def _valid_df():
    return pd.DataFrame(
        {
            "SizeSqFt": [1000, 1500],
            "Bedrooms": [2, 3],
            "AgeYears": [10, 5],
            "Price": [80000, 120000],
        }
    )


def _large_valid_df():
    """Return a DataFrame large enough for meaningful train/test split."""
    return pd.DataFrame(
        {
            "SizeSqFt": [800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600],
            "Bedrooms": [1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
            "AgeYears": [20, 15, 10, 8, 5, 3, 2, 1, 0, 12],
            "Price": [50000, 80000, 95000, 120000, 140000, 160000, 200000, 220000, 260000, 150000],
        }
    )


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_required_input(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_defaults(self):
        cfg = parse_args(["--input", "data.csv"])
        assert str(cfg.input_path) == "data.csv"
        assert str(cfg.output_path) == "output/house_prices_cleaned.csv"
        assert cfg.drop_duplicates is False
        assert cfg.verbose is False
        assert cfg.random_seed == 42
        assert cfg.test_size == 0.2

    def test_all_flags(self):
        cfg = parse_args([
            "--input", "in.csv",
            "--output", "out.csv",
            "--drop-duplicates",
            "--verbose",
            "--seed", "123",
            "--test-size", "0.3",
        ])
        assert str(cfg.input_path) == "in.csv"
        assert str(cfg.output_path) == "out.csv"
        assert cfg.drop_duplicates is True
        assert cfg.verbose is True
        assert cfg.random_seed == 123
        assert cfg.test_size == 0.3


# ---------------------------------------------------------------------------
# validate_input_file
# ---------------------------------------------------------------------------

class TestValidateInputFile:
    def test_missing_file(self, tmp_path):
        cfg = _make_config(tmp_path, input_path=tmp_path / "no_such.csv")
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_input_file(cfg)

    def test_wrong_extension(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("a,b\n1,2\n")
        cfg = _make_config(tmp_path, input_path=f)
        with pytest.raises(ValueError, match="Only .csv"):
            validate_input_file(cfg)

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        cfg = _make_config(tmp_path, input_path=f)
        with pytest.raises(ValueError, match="empty"):
            validate_input_file(cfg)

    def test_valid_file(self, tmp_path):
        f = tmp_path / "ok.csv"
        f.write_text("a\n1\n")
        cfg = _make_config(tmp_path, input_path=f)
        validate_input_file(cfg)  # should not raise


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_valid_schema(self):
        validate_schema(_valid_df())  # no exception

    def test_missing_column(self):
        df = _valid_df().drop(columns=["Price"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_extra_columns_warns(self):
        df = _valid_df()
        df["Extra"] = 1
        validate_schema(df)  # warns but does not raise


# ---------------------------------------------------------------------------
# clean_data
# ---------------------------------------------------------------------------

class TestCleanData:
    def test_valid_data_unchanged(self, tmp_path):
        cfg = _make_config(tmp_path)
        result = clean_data(_valid_df(), cfg)
        assert len(result) == 2
        assert list(result.columns) == EXPECTED_COLUMNS

    def test_coerces_non_numeric(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = pd.DataFrame(
            {
                "SizeSqFt": [1000, 1500],
                "Bedrooms": [2, 3],
                "AgeYears": [10, 5],
                "Price": ["not_a_number", 120000],  # string in numeric column
            }
        )
        result = clean_data(df, cfg)
        assert len(result) == 1  # bad row dropped

    def test_invalid_rows_filtered_not_raised(self, tmp_path):
        """Bad rows are filtered out with warnings, not raised as errors."""
        cfg = _make_config(tmp_path)
        df = pd.DataFrame(
            {
                "SizeSqFt": [1000, 1500],
                "Bedrooms": [2, 3],
                "AgeYears": [10, 5],
                "Price": [-100, 120000],  # first row has negative price
            }
        )
        result = clean_data(df, cfg)
        assert len(result) == 1
        assert result.iloc[0]["Price"] == 120000

    def test_all_invalid_rows_raises(self, tmp_path):
        """If ALL rows are invalid, ValueError is raised."""
        cfg = _make_config(tmp_path)
        df = pd.DataFrame(
            {
                "SizeSqFt": [-1, 0],
                "Bedrooms": [2, 3],
                "AgeYears": [10, 5],
                "Price": [80000, 120000],
            }
        )
        with pytest.raises(ValueError, match="No valid rows remain"):
            clean_data(df, cfg)

    def test_negative_price_filtered(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = _valid_df()
        df.loc[0, "Price"] = -100
        result = clean_data(df, cfg)
        assert len(result) == 1  # bad row filtered, not raised

    def test_zero_bedrooms_filtered(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = _valid_df()
        df.loc[0, "Bedrooms"] = 0
        result = clean_data(df, cfg)
        assert len(result) == 1

    def test_negative_age_filtered(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = _valid_df()
        df.loc[0, "AgeYears"] = -1
        result = clean_data(df, cfg)
        assert len(result) == 1

    def test_zero_sqft_filtered(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = _valid_df()
        df.loc[0, "SizeSqFt"] = 0
        result = clean_data(df, cfg)
        assert len(result) == 1

    def test_drop_duplicates(self, tmp_path):
        cfg = _make_config(tmp_path, drop_duplicates=True)
        df = pd.concat([_valid_df(), _valid_df()], ignore_index=True)
        result = clean_data(df, cfg)
        assert len(result) == 2

    def test_deterministic_order(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = _valid_df().iloc[::-1].reset_index(drop=True)  # reversed
        r1 = clean_data(df, cfg)
        r2 = clean_data(_valid_df(), cfg)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_split_reproducibility(self, tmp_path):
        """Same seed produces identical splits."""
        cfg = _make_config(tmp_path)
        df = _large_valid_df()

        train1, test1 = split_data(df, cfg)
        train2, test2 = split_data(df, cfg)

        pd.testing.assert_frame_equal(
            train1.reset_index(drop=True), train2.reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            test1.reset_index(drop=True), test2.reset_index(drop=True)
        )

    def test_split_writes_files(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = _large_valid_df()
        split_data(df, cfg)

        assert (tmp_path / "output_train.csv").exists()
        assert (tmp_path / "output_test.csv").exists()

    def test_split_sizes(self, tmp_path):
        cfg = _make_config(tmp_path, test_size=0.3)
        df = _large_valid_df()
        train, test = split_data(df, cfg)
        assert len(train) + len(test) == len(df)


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def test_returns_metrics(self, tmp_path):
        cfg = _make_config(tmp_path)
        df = _large_valid_df()
        train, test = split_data(df, cfg)
        metrics = evaluate_model(train, test)

        assert "r2" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0


# ---------------------------------------------------------------------------
# write_cleaned_csv
# ---------------------------------------------------------------------------

class TestWriteCleanedCSV:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "sub" / "out.csv"
        cfg = _make_config(tmp_path, output_path=out)
        write_cleaned_csv(_valid_df(), cfg)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) == 2

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "out.csv"
        cfg = _make_config(tmp_path, output_path=out)
        write_cleaned_csv(_valid_df(), cfg)
        assert out.exists()

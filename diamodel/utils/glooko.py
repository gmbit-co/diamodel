#!/usr/bin/env python3
"""
Glooko data import script.

Reads CGM and bolus data from a Glooko export folder and outputs
a normalized CSV with aligned time-series data.
"""

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd


def load_cgm(file_path: Path) -> pd.DataFrame:
    """Load CGM data from Glooko export CSV."""
    if not file_path.exists():
        raise FileNotFoundError(f"CGM data file not found: {file_path}")

    # Detect CGM units from column names
    headers = pd.read_csv(file_path, skiprows=1, nrows=0).columns
    if "CGM Glucose Value (mg/dl)" in headers:
        col_name = "CGM Glucose Value (mg/dl)"
        convert_to_mmol = True
    else:
        col_name = "CGM Glucose Value (mmol/l)"
        convert_to_mmol = False

    df = pd.read_csv(
        file_path,
        dtype={col_name: "float64"},
        skiprows=1,
        parse_dates=["Timestamp"],
    )
    df.rename(
        columns={
            "Timestamp": "timestamp",
            col_name: "cgm",
        },
        inplace=True,
    )
    if convert_to_mmol:
        df["cgm"] = round(df["cgm"] / 18.0156, 1)  # to mmol/L
    return df[["timestamp", "cgm"]]


def load_bolus(file_path: Path) -> pd.DataFrame:
    """Load bolus data from Glooko export CSV."""
    if not file_path.exists():
        raise FileNotFoundError(f"Bolus data file not found: {file_path}")
    df = pd.read_csv(
        file_path,
        dtype={
            "Carbs Input (g)": "float64",
            "Insulin Delivered (U)": "float64",
        },
        skiprows=1,
        parse_dates=["Timestamp"],
    )
    df.rename(
        columns={
            "Timestamp": "timestamp",
            "Carbs Input (g)": "carbs",
            "Insulin Delivered (U)": "insulin",
        },
        inplace=True,
    )
    return df[["timestamp", "carbs", "insulin"]]


def load_dataset(glooko_export_dir: str | Path, dt: int = 5) -> pd.DataFrame:
    """Load and create aligned time-series dataset from Glooko export folder."""
    export_dir = Path(glooko_export_dir)
    cgm_df = load_cgm(export_dir / "cgm_data_1.csv")
    bolus_df = load_bolus(export_dir / "Insulin data" / "bolus_data_1.csv")

    start_time = cgm_df["timestamp"].min()

    def ts_to_tick(ts):
        return round((ts - start_time).total_seconds() / 60 / dt)

    def tick_to_ts(tick):
        return start_time + timedelta(minutes=tick * dt)

    n = ts_to_tick(cgm_df["timestamp"].max())

    data_df = pd.DataFrame({"tick": range(n + 1)})
    data_df.set_index("tick", inplace=True, drop=False)
    data_df["timestamp"] = data_df.index.map(tick_to_ts)

    cgm_ticks = cgm_df["timestamp"].map(ts_to_tick)
    data_df.loc[cgm_ticks, "cgm"] = cgm_df["cgm"].values

    bolus_ticks = bolus_df["timestamp"].map(ts_to_tick)
    bolus_grouped = bolus_df.groupby(bolus_ticks)[["carbs", "insulin"]].sum()
    data_df.loc[bolus_grouped.index, "carbs"] = bolus_grouped["carbs"].values
    data_df.loc[bolus_grouped.index, "insulin"] = bolus_grouped["insulin"].values

    return data_df


def main():
    parser = argparse.ArgumentParser(
        description="Import Glooko export data and create normalized CSV"
    )
    parser.add_argument(
        "glooko_export_dir",
        type=str,
        help="Path to Glooko export folder",
    )
    parser.add_argument(
        "--dt",
        type=int,
        default=5,
        help="Tick spacing in minutes (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data.csv",
        help="Output CSV file path (default: data.csv)",
    )

    args = parser.parse_args()

    print(f"Loading dataset from: {args.glooko_export_dir}")
    data_df = load_dataset(args.glooko_export_dir, args.dt)
    print(f"  Created {len(data_df)} time points")

    print(f"Saving to: {args.output}")
    data_df.to_csv(args.output, index=False)
    print("Done!")


if __name__ == "__main__":
    main()

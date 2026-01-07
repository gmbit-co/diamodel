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
    df = pd.read_csv(
        file_path,
        dtype={"CGM Glucose Value (mmol/l)": "float64"},
        skiprows=1,
        parse_dates=["Timestamp"],
    )
    df.rename(
        columns={
            "Timestamp": "timestamp",
            "CGM Glucose Value (mmol/l)": "cgm",
        },
        inplace=True,
    )
    return df[["timestamp", "cgm"]]


def load_bolus(file_path: Path) -> pd.DataFrame:
    """Load bolus data from Glooko export CSV."""
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


def create_dataset(cgm_df: pd.DataFrame, bolus_df: pd.DataFrame, dt: int) -> pd.DataFrame:
    """Create aligned time-series dataset from CGM and bolus data."""
    start_time = cgm_df["timestamp"].min()

    def ts_to_tick(ts):
        return round((ts - start_time).total_seconds() / 60 / dt)

    def tick_to_ts(tick):
        return start_time + timedelta(minutes=tick * dt)

    n = ts_to_tick(cgm_df["timestamp"].max())

    data_df = pd.DataFrame({"tick": range(n + 1)})
    data_df.set_index("tick", inplace=True)
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

    export_dir = Path(args.glooko_export_dir)
    cgm_path = export_dir / "cgm_data_1.csv"
    bolus_path = export_dir / "Insulin data" / "bolus_data_1.csv"

    if not cgm_path.exists():
        raise FileNotFoundError(f"CGM data file not found: {cgm_path}")
    if not bolus_path.exists():
        raise FileNotFoundError(f"Bolus data file not found: {bolus_path}")

    print(f"Loading CGM data from: {cgm_path}")
    cgm_df = load_cgm(cgm_path)
    print(f"  Loaded {len(cgm_df)} CGM readings")

    print(f"Loading bolus data from: {bolus_path}")
    bolus_df = load_bolus(bolus_path)
    print(f"  Loaded {len(bolus_df)} bolus events")

    print(f"Creating aligned dataset with dt={args.dt} minutes...")
    data_df = create_dataset(cgm_df, bolus_df, args.dt)
    print(f"  Created {len(data_df)} time points")

    print(f"Saving to: {args.output}")
    data_df.to_csv(args.output, index=False)
    print("Done!")


if __name__ == "__main__":
    main()

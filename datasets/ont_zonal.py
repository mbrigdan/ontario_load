import pandas as pd
import numpy as np
import pathlib
from pathlib import Path

# description of the data
HEADERS_ROW = 3

META_COLUMNS = ["ts"]
TARGET_COLUMNS = ["Ontario Demand","Northwest", "Northeast", "Ottawa", "East",
             "Toronto", "Essa", "Bruce", "Southwest", "Niagara", "West"]

def _load_data_to_df(year: int) -> pd.DataFrame:
    """

    """
    parent = Path(__file__).parent.resolve()
    df = pd.read_csv(parent/"ontario_zonal"/f"PUB_DemandZonal_{year}.csv", header=HEADERS_ROW)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df.loc[df["Hour"] == 24, "Date"] += pd.Timedelta(days=1)
    df.loc[df["Hour"] == 24, "Hour"] = 0
    df["Hour"] = df["Hour"].astype(str).str.zfill(2)
    df["datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Hour"],
                                    format="%Y-%m-%d %H")
    df["ts"] = df["datetime"].values.astype(np.int64) // 10 ** 9

    df = df.set_index("datetime")
    return df


def get_single_series(year: int):
    """
    Returns the time-series for the whole province only
    Index is datetime in 2020-01-01 01:00:00 format
    """
    df = _load_data_to_df(year)
    df.rename(columns={"Ontario Demand": "value"}, inplace=True)
    return df["value"]

def get_series_with_ts(year: int):
    df = _load_data_to_df(year)
    df.rename(columns={"Ontario Demand": "value"}, inplace=True)
    return df["value"], df["ts"]

def get_all_zone_data():
    df = _load_data_to_df()

    df = df[TARGET_COLUMNS + META_COLUMNS]
    return df
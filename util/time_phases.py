import numpy as np
import pandas as pd

from enum import Enum

class CycleLengths(float, Enum):
    """
    Enum for cycle lengths, in seconds
    """
    MINUTE = 60
    HOUR = MINUTE * 60
    DAY = HOUR * 24
    WEEK = DAY * 7
    MONTH = DAY * 30.4368
    YEAR = 365.2425 * DAY

CYCLE_LENGTH_NAMES = {
    CycleLengths.MINUTE: 'minute',
    CycleLengths.HOUR: 'hour',
    CycleLengths.DAY: 'day',
    CycleLengths.WEEK: 'week',
    CycleLengths.MONTH: 'month',
    CycleLengths.YEAR: 'year'
}


def _make_phases(series: pd.Series, cycle_length: float) -> (pd.Series, pd.Series):
    phase_sin = np.sin(2 * np.pi * series / cycle_length)
    phase_cos = np.cos(2 * np.pi * series / cycle_length)
    return phase_sin, phase_cos


def add_time_phases(df: pd.DataFrame, lengths: list[CycleLengths]) -> pd.DataFrame:
    for cycle_length in lengths:
        phase_sin, phase_cos = _make_phases(df["ts"], cycle_length)
        name = CYCLE_LENGTH_NAMES[cycle_length]
        df[f"{name}_sin"] = phase_sin
        df[f"{name}_cos"] = phase_cos

    return df
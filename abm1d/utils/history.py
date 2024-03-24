from typing import Iterable

import pandas as pd


def make(columns: Iterable[str]) -> pd.DataFrame:
    columns = ["timestamp"] + list(columns)
    return pd.DataFrame({column: pd.Series(dtype=float) for column in columns})


def append(df: pd.DataFrame, timestamp: float, values: dict[str, float]) -> None:
    idx = len(df)
    if len(df) > 0:
        last = df.iloc[-1]["timestamp"]
        if timestamp < last:
            raise RuntimeError("History timestamps must monotonically increase")
        elif timestamp == last:
            idx = len(df) - 1
    df.loc[idx] = {"timestamp": timestamp, **values}


def offset_forward(df: pd.DataFrame, offset: float) -> float:
    return df["timestamp"].iloc[0] + offset


def offset_backward(df: pd.DataFrame, offset: float) -> float:
    return df["timestamp"].iloc[-1] - offset


def index_absolute(df: pd.DataFrame, timestamp: float) -> int:
    index = df["timestamp"].searchsorted(timestamp, side="right") - 1
    index = min(max(index, 0), len(df) - 1)
    return index


def index_forward(df: pd.DataFrame, offset: float) -> int:
    return index_absolute(df, offset_forward(df, offset))


def index_backward(df: pd.DataFrame, offset: float) -> int:
    return index_absolute(df, offset_backward(df, offset))


def slice_absolute(
    df: pd.DataFrame, timestamp_start: float | None, timestamp_stop: float | None
) -> pd.DataFrame:
    if timestamp_start is None:
        index_start = None
    else:
        index_start = index_absolute(df, timestamp_start)
    if timestamp_stop is None:
        index_stop = None
    else:
        index_stop = index_absolute(df, timestamp_stop)
    return df.iloc[index_start:index_stop]


def slice_relative(
    df: pd.DataFrame, offset_start: float | None, offset_stop: float | None
) -> pd.DataFrame:
    if offset_start is None:
        timestamp_start = None
    else:
        if offset_start < 0.0:
            timestamp_start = offset_backward(df, -offset_start)
        else:
            timestamp_start = offset_forward(df, offset_start)
    if offset_stop is None:
        timestamp_stop = None
    else:
        if offset_stop < 0.0:
            timestamp_stop = offset_backward(df, -offset_stop)
        else:
            timestamp_stop = offset_forward(df, offset_stop)
    return slice_absolute(df, timestamp_start, timestamp_stop)

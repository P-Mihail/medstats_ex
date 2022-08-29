from typing import List

import pandas as pd


def combine_rows(df: pd.DataFrame, rows: List) -> pd.DataFrame:
    """
    Объединяет несколько строк датафрейма (суммируя) в одну. 
    """
    df = df.copy()

    df.loc[" + ".join(map(str, rows))] = df.loc[rows].sum(0)
    df.drop(rows, inplace=True)

    return df


def combine_cols(df: pd.DataFrame, cols: List) -> pd.DataFrame:
    """
    Объединяет несколько столбцов датафрейма (суммируя) в один. 
    """
    df = df.copy()

    df[" + ".join(map(str, cols))] = df[cols].sum(1)
    df.drop(cols, inplace=True, axis=1)

    return df

from typing import List, Union

import pandas as pd
from scipy.stats import t


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


def add_CI(df: pd.DataFrame, p: int, nms: List[Union[int, str]] = ["n", "mean", "std"]) -> pd.DataFrame:
    """
    Добавляет к датафрейму доверительный интервал P%
    nms - список названий столбцов содержащих число наблюдений, среднее значение и стандартное отклонение.
    Вместо имени столбца с числом наблюдений может быть передано число для случая когда такого столбца нет, а размер выборок одинаков для всех строк.
    """
    del_col_n = False

    if type(nms[0]) == int:
        df["n"] = nms[0]
        nms[0] = "n"
        del_col_n = True

    # Margin of error
    df[f"MoE_{p}"] = t.ppf((1+p/100)/2, df=df[nms[0]]-1,
                           scale=df[nms[2]]/df[nms[0]]**.5)
    df[[f"{p}% LB", f"{p}% RB"]] = df.apply(lambda row: t.interval(p/100, row[nms[0]]-1, loc=row[nms[1]], scale=row[nms[2]] / row[nms[0]]**.5),
                                            axis=1,
                                            result_type='expand')
    if del_col_n:
        df.drop(columns=["n"], inplace=True)

    return df

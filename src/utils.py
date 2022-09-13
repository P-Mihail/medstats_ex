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
    df_ = df[nms[1:]].rename(columns={nms[1]: "mean", nms[2]: "std"})

    if type(nms[0]) == int:
        df_["n"] = nms[0]
    else:
        df_["n"] = df[nms[0]]

    # Margin of error
    df_[f"MoE_{p}"] = t.ppf((1+p/100)/2, df=df_["n"]-1,
                            scale=df_["std"]/df_["n"]**.5)
    df_[[f"{p}% LB", f"{p}% RB"]] = df_.apply(lambda row: t.interval(p/100, row["n"]-1, loc=row["mean"], scale=row["std"] / row["n"]**.5),
                                              axis=1,
                                              result_type='expand')

    return df.join(df_[[f"MoE_{p}", f"{p}% LB", f"{p}% RB"]])

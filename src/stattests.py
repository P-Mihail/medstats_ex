from typing import Union
import numpy as np
import pandas as pd
import scipy.stats


# ДВЕ ГРУППЫ

# критерий Стьюдента


def t_test(
    size: Union[int, list, np.ndarray, pd.Series],
    means: Union[list, np.ndarray, pd.Series],
    stds: Union[list, np.ndarray, pd.Series],
    silent: bool = True,
) -> None:
    """
    Проверка гипотезы, что средние двух групп равны (исслед. факторы не оказ. никакого влияния и полученные различия случайны).
    """

    assert len(means) == 2, "len(means) != 2"
    assert len(stds) == 2, "len(stds) != 2"

    if type(size) == int:
        n = np.array([size, size])
    else:
        n = np.array(size)
    assert len(n) == 2, "len(n) != 2"

    # Число степеней свободы:
    v = sum(n) - 2

    # Объединенная оценка дисперсии
    S2 = ((np.array(n) - 1) * np.power(stds, 2)).sum() / v

    t = (means[0] - means[1]) / np.sqrt(S2 * sum(n) / n[0] / n[1])

    # p-уровень значимости
    p = 2 * (scipy.stats.t.cdf(-abs(t), v))

    # Определение критического уровня t
    t01 = scipy.stats.t.ppf(q=1 - 0.01 / 2, df=v)  # two-sided
    t05 = scipy.stats.t.ppf(q=1 - 0.05 / 2, df=v)

    if not silent:
        print(f"Объединенная оценка дисперсии: {S2:.3f}")
        print(f"Число степеней свободы: {v}\n")
        print(f"t = {t:.3f}")
        print(f"p = {p:.3f}")
        print(
            f"Критический значение для заданного числа степеней сввободы и уровней значимости 0.01 и 0.05: t01 = {t01:.3f}, t05 = {t05:.3f}\n"
        )

    if abs(t) > t01:
        print(
            f"Различия статистически значимы. P = {p:.3f} < 0.01 (|t = {t:.3f}| > t01 = {t01:.3f})"
        )
    elif abs(t) < t05:
        print(
            f"Различия статистически не значимы. P = {p:.3f} > 0.05 (|t = {t:.3f}| < t05 = {t05:.3f})"
        )
    else:
        print(
            f"Пограничный случай, есть основания задуматься над наличием различий. 0.01 > P = {p:.3f} > 0.05 (t05 = {t05:.3f} > |t = {t:.3f}| > t01 = {t01:.3f})"
        )


# НЕСКОЛЬКО ГРУПП, ПРОВЕРКА ГИПОТЕЗЫ, ЧТО ВСЕ СРЕДНИЕ РАВНЫ


def f_test(
    size: Union[int, list, np.ndarray, pd.Series],
    means: Union[list, np.ndarray, pd.Series],
    stds: Union[list, np.ndarray, pd.Series],
    silent: bool = True,
) -> None:
    """
    Проверка гипотезы, что все средние равны (исслед. факторы не оказ. никакого влияния и полученные различия случайны).
    Анализ основан на сравнении межгрупповой и внутригрупповой дисперсий.
    """

    assert len(means) == len(stds), "len(means) != len(stds)"

    # size is int, np.array, pd.Series or list
    if type(size) == int:
        n = np.repeat(size, len(means))
    else:
        n = np.array(size)

    assert len(means) == len(n), "len(means) != len(n)"
    N = sum(n)

    # степени свободы
    vbg = len(means) - 1
    vwg = N - len(means)

    # оценка внутригрупповой и межгрупповой дисперсий
    MSbg = (n * (means - np.average(means, weights=n)) ** 2).sum() / vbg
    MSwg = ((n - 1) * np.power(stds, 2)).sum() / vwg

    F = MSbg / MSwg
    p = 1 - scipy.stats.f.cdf(F, dfn=vbg, dfd=vwg)

    # Определение критического уровня F
    F01 = scipy.stats.f.ppf(q=1 - 0.01, dfn=vbg, dfd=vwg)
    F05 = scipy.stats.f.ppf(q=1 - 0.05, dfn=vbg, dfd=vwg)

    if not silent:
        print(f"Межгрупповая дисперсия: MSbg = {MSbg:.3f}")
        print(f"Внутригрупповая дисперсия: MSwg = {MSwg:.3f}\n")
        print(f"F = MSbg / MSwg = {F:.3f}\n")
        print(f"P = {p:.3f}\n")
        print(f"Межгрупповое число степеней свободы: vbg = {vbg}")
        print(f"Внутригрупповое число степеней свободы: vwg = {vwg}\n")
        print(
            f"Критический значение для заданного числа степеней сввободы и уровней значимости 0.01 и 0.05: \n\t F01 = {F01:.3f} \n\t F05 = {F05:.3f}\n"
        )

    if F > F01:
        print(
            f"Различия статистически значимы. P = {p:.3f} < 0.01 (F = {F:.3f} > F01 = {F01:.3f})"
        )
    elif F < F05:
        print(
            f"Различия статистически не значимы. P = {p:.3f} > 0.05 (F = {F:.3f} < F05 = {F05:.3f})"
        )
    else:
        print(
            f"Пограничный случай, есть основания задуматься над наличием различий. 0.01 < P = {p:.3f} < 0.05 (F05 = {F05:.3f} < F = {F:.3f} < F01 = {F01:.3f})"
        )


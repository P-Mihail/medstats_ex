from typing import Callable, List, Optional, Union

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import interpolate
import scipy.stats
import scipy.special
from statsmodels.stats.libqsturng import psturng

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


# НЕСКОЛЬКО ГРУПП

# проверка гипотезы, что все средние равны


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


# А) Критерий Стьюдента для множественных сравнений
# поправки Бонферрони


def t_test_benf(
    size: Union[int, list, np.ndarray, pd.Series],
    means: Union[list, np.ndarray, pd.Series],
    stds: Union[list, np.ndarray, pd.Series],
    names: Optional[List[str]] = None,
    ctrl_group: Optional[None] = None,  # индекс контрольной группы
    silent: bool = True,
) -> None:
    """
    Критерий Стьюдента для множественных сравнений с применением поправки Бонферрони для компенсации эффекта множественных сравнений.
    Предполагается что предварительно уже была отвергнута нулевая гипотеза о равенстве всех средних (f_test).

    Если не задан индекс контрольной группы:
        Позволяет выделить группы отличные от остальных.
    иначе:
        Производится сравнения каждой группы только с контрольной.
    """

    assert len(means) == len(stds), "len(means) != len(stds)"

    # size is int, np.array, pd.Series or list
    if type(size) == int:
        n = np.repeat(size, len(means))
    else:
        n = np.array(size)
    assert len(means) == len(n), "len(means) != len(n)"

    assert not names or len(names) == len(n), "names not None and len(names) != len(n)"

    # Внутригрупповая оценка дисперсии
    Sw = np.average(np.power(stds, 2), weights=n)
    # Число степеней свободы
    v = sum(n) - len(n)

    if ctrl_group is None:
        k = scipy.special.comb(len(n), 2, exact=True)  # Число сравнений
        idxs = np.array(
            [[x[0], x[1]] for x in list(itertools.combinations(range(len(n)), 2))]
        ).T
    else:
        k = len(n) - 1
        idxs = np.array([[ctrl_group, i] for i in range(len(n)) if i != ctrl_group]).T

    t01 = scipy.stats.t.ppf(q=1 - 0.01 / 2 / k, df=v)  # two-sided
    t05 = scipy.stats.t.ppf(q=1 - 0.05 / 2 / k, df=v)

    t = (np.take(list(means), idxs[1]) - np.take(list(means), idxs[0])) / np.sqrt(  # type: ignore
        Sw * (1.0 / np.take(n, idxs[0]) + 1.0 / np.take(n, idxs[1]))  # type: ignore
    )

    if not silent:
        print(f"Внутригрупповая оценка дисперсии: {Sw:.3f}")
        print(f"Число степеней свободы: {v}")
        print(f"Число сравнений: {k}")
        print(
            f"Критический значение для заданного числа степеней сввободы и уровней значимости 0.01 и 0.05 с учетом поправки Бонферрони: t01 = {t01:.3f}, t05 = {t05:.3f}\n"
        )

    p = 2 * (scipy.stats.t.cdf(-np.abs(t), v))

    groups = [[], [], []]
    for i in np.argsort(-p):
        if p[i] > 0.05 / k:
            groups[0].append(i)
        elif p[i] > 0.01 / k:
            groups[1].append(i)
        else:
            groups[2].append(i)

    labels = [
        "Различия статистически не значимы:",
        "Пограничный случай, есть основания задуматься над наличием различий:",
        "Различия статистически значимы.",
    ]

    for label, group in zip(labels, groups):
        print(f"\n{label}")
        if len(group) == 0:
            print("\t--------------------------------------------")
        else:
            for i in group:
                if names is None:
                    print(
                        f"\tt('{idxs[0][i]}', '{idxs[1][i]}') = {t[i]:.3f} (p-value = {p[i]:.3f})"
                    )
                else:
                    print(
                        f"\tt('{names[idxs[0][i]]}', '{names[idxs[1][i]]}') = {t[i]:.3f} (p-value = {p[i]:.3f})"
                    )


# Б) Критерий Стьюдента для множественных сравнений
# Критерий Ньюмена-Кейлса + Критерий Тьюки
# для большого числа сравнений, когда поправки Бонферрони делает критерий Стьюдента излишне жестким


def t_test_nk(
    size: Union[int, list, np.ndarray, pd.Series],
    means: Union[list, np.ndarray, pd.Series],
    stds: Union[list, np.ndarray, pd.Series],
    names: Optional[List[str]] = None,
    tukey: bool = False,  # Критерий Тьюки
    silent: bool = True,
) -> None:
    """
    Критерий Стьюдента для множественных сравнений с применением критериев Ньюмена-Кейлса или Тьюки для компенсации эффекта множественных сравнений.
    Предполагается что предварительно уже была отвергнута нулевая гипотеза о равенстве всех средних (f_test).

    * Критерий Ньюмена-Кейлса был разработан как усовершенствование критерия Тьюки.
    """

    assert len(means) == len(stds), "len(means) != len(stds)"

    # size is int, np.array, pd.Series or list
    if type(size) == int:
        n = np.repeat(size, len(means))
    else:
        n = np.array(size)
    assert len(means) == len(n), "len(means) != len(n)"

    assert not names or len(names) == len(n), "names not None and len(names) != len(n)"

    # Внутригрупповая оценка дисперсии
    Sw = np.average(np.power(stds, 2), weights=n)
    # Число степеней свободы
    v = sum(n) - len(n)

    # t05 = {l: qsturng(1-.05, l, v) for l in range(2, len(means) + 1)}
    # t01 = {l: qsturng(1-.01, l, v) for l in range(2, len(means) + 1)}

    if not silent:
        print(f"Внутригрупповая оценка дисперсии: {Sw:.3f}")
        print(f"Число степеней свободы: {v}")

    means_rank = np.argsort(means)  # индексы упорядочивающие средние значения
    idxs = np.array(
        [
            [means_rank[i], means_rank[j]]
            for i in range(len(means) - 1, -1, -1)
            for j in range(i)
        ]
    ).T

    if tukey:
        l = len(means)  # КРИТЕРИИ ТЬЮКИ
        # (можно l задать просто скаляром len(means), но тогда надо бороться с pylance)
    else:
        l = [
            i - j + 1 for i in range(len(means) - 1, -1, -1) for j in range(i)
        ]  # КРИТЕРИЙ НЬЮМЕНА-КЕЙЛСА

    t = (np.take(list(means), idxs[0]) - np.take(list(means), idxs[1])) / np.sqrt(  # type: ignore
        Sw * (1.0 / np.take(n, idxs[0]) + 1.0 / np.take(n, idxs[1])) / 2.0  # type: ignore
    )
    p: np.ndarray = psturng(np.abs(t), l, v)  # type: ignore

    groups = [[], [], []]

    for i in np.argsort(t):  # отображать в каждой группе по нарастанию стат значимости
        if p[i] > 0.05:
            groups[0].append(i)
        elif p[i] > 0.01:
            groups[1].append(i)
        else:
            groups[2].append(i)
    labels = [
        "Различия статистически не значимы:",
        "Пограничный случай, есть основания задуматься над наличием различий:",
        "Различия статистически значимы.",
    ]

    for label, group in zip(labels, groups):
        print(f"\n{label}")
        if len(group) == 0:
            print("\t--------------------------------------------")
        else:
            for i in group:
                if names is None:
                    print(
                        f"\tt('{idxs[0][i]}', '{idxs[1][i]}') = {t[i]:.3f}, {'' if tukey else f'l = {l[i]},'} p-value = {p[i]:.3f}"
                    )
                else:
                    print(
                        f"\tt('{names[idxs[0][i]]}', '{names[idxs[1][i]]}') = {t[i]:.3f}, {'' if tukey else f'l = {l[i]},'} p-value = {p[i]:.3f}"
                    )


# В) Критерий Стьюдента для множественных сравнений
# Критерий Даннета
# *это вариант критерия Ньюмена–Кейлса для сравнения нескольких групп с одной контрольной


def _make_dunnetts_q_value(
    file: Union[str, Path]
) -> Callable[[int, float, int], float]:
    """
    Создает функцию, для получения критических значений для теста Даннета на основе таблицы из file, интерполируя промежуточные значения.
    """
    df = pd.read_csv(file, index_col=[1, 0, 2]).unstack(-1).droplevel(0, axis=1)  # type: ignore
    mem = {}
    for alpha in df.index.unique(0):
        df_ = df.loc[alpha]
        mem[alpha] = interpolate.interp2d(
            df_.columns, np.where(df_.index == np.inf, 1e12, df_.index), df_
        )

    def nested(v: int, alpha: float, l: int) -> float:
        assert alpha in mem, f"alpha should be in {mem.keys()}"
        assert 2 <= v, f"v should be >= 2"
        assert int(v) == v, f"v should be int"
        assert 3 <= l <= 21, f"l should be >= 3"
        assert int(l) == l, f"l should be int"

        return mem[alpha](l, v)[0]

    return nested


dunnetts_q_value = _make_dunnetts_q_value(
    Path(__file__).parent.resolve() / "Dunnetts_Table.csv"
)


def dunnetts_test(
    size: Union[int, list, np.ndarray, pd.Series],
    means: Union[list, np.ndarray, pd.Series],
    stds: Union[list, np.ndarray, pd.Series],
    names: Optional[List[str]] = None,
    ctrl_group: int = 0,  # индекс контрольной группы
    silent: bool = True,
) -> None:
    """
    Критерий Стьюдента для множественных сравнений с контрольной группой с применением критерия Даннета для компенсации эффекта множественных сравнений.
    Предполагается что предварительно уже была отвергнута нулевая гипотеза о равенстве всех средних (f_test).

    * Критерий Даннета это вариант критерия Ньюмена–Кейлса для сравнения нескольких групп с одной контрольной.
    """
    assert len(means) == len(stds), "len(means) != len(stds)"

    # n is int, np.array, pd.Series or list
    if type(size) == int:
        n = np.repeat(size, len(means))
    else:
        n = np.array(size)
    assert len(means) == len(n), "len(means) != len(n)"

    assert not names or len(names) == len(n), "names not None and len(names) != len(n)"

    # Внутригрупповая оценка дисперсии
    Sw = np.average(np.power(stds, 2), weights=n)
    # Число степеней свободы
    v = sum(n) - len(n)

    if not silent:
        print(f"Внутригрупповая оценка дисперсии: {Sw:.3f}")
        print(f"Число степеней свободы: {v}")

    means_rank = np.argsort(means)  # индексы упорядочивающие средние значения
    idxs = np.array([[ctrl_group, i] for i in range(len(n)) if i != ctrl_group]).T

    l = len(means)

    q = (np.take(list(means), idxs[0]) - np.take(list(means), idxs[1])) / np.sqrt(  # type: ignore
        Sw * (1.0 / np.take(n, idxs[0]) + 1.0 / np.take(n, idxs[1]))  # type: ignore
    )

    groups = [[], [], []]

    q05 = dunnetts_q_value(sum(n), 0.05, l)
    q01 = dunnetts_q_value(sum(n), 0.01, l)

    for i in np.argsort(q):  # отображать в каждой группе по нарастанию стат значимости
        if abs(q[i]) < q05:
            groups[0].append(i)
        elif abs(q[i]) < q01:
            groups[1].append(i)
        else:
            groups[2].append(i)
    labels = [
        f"Различия статистически не значимы, |q| < q05 = {q05:.3f}:",
        f"Пограничный случай, есть основания задуматься над наличием различий, q01 = {q01:.3f} < |q| < q05 = {q05:.3f}:",
        f"Различия статистически значимы, q01 = {q01:.3f} < |q|:",
    ]

    for label, group in zip(labels, groups):
        print(f"\n{label}")
        if len(group) == 0:
            print("\t--------------------------------------------")
        else:
            for i in group:
                if names is None:
                    print(f"\tq('{idxs[0][i]}', '{idxs[1][i]}') = {q[i]:.3f}")
                else:
                    print(
                        f"\tq('{names[idxs[0][i]]}', '{names[idxs[1][i]]}') = {q[i]:.3f}"
                    )


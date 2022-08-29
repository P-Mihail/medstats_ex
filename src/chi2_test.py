from typing import List, Union

import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact


def _is_expected_valid(expected_inp: Union[List[List[float]], np.ndarray], silent=True) -> bool:
    """
    Проверка валидности матрицы ожидаемых числел для проверки применимости критерия chi2.
    """
    expected = np.array(expected_inp)

    if expected.shape == (2, 2):
        if expected.min() < 5:
            if not silent:
                print(
                    "Не выполнено условие применимости для матрицы 2х2, в ожидаемой матрице существует элемент < 5")
            return False
    else:
        if expected.min() < 1:
            if not silent:
                print(
                    f"Не выполнено условие применимости для матрицы {expected.shape[0]}x{expected.shape[1]}, в ожидаемой матрице существует элемент < 1")
            return False
        if (expected < 5).mean() > 0.2:
            if not silent:
                print(
                    f"Не выполнено условие применимости для матрицы {expected.shape[0]}x{expected.shape[1]}, в ожидаемой матрице число элементов < 5 превышает 20%")
            return False

    return True


def chi2_test(df: pd.DataFrame, bonf=1, silent=True) -> None:
    """
    Проверка статистической значимости различий в качественных показателях групп.

    bonf - коэффициент поправки Бонферрони (число множественных сравнений)
    """
    chi2, p, dof, expected = chi2_contingency(df)

    if _is_expected_valid(expected, silent=silent):
        if not silent:
            print(f"chi^2 = {chi2:.3f}, v={dof}, P={p:.3f}")
    else:
        print("Критерий CHI2 не применим для этих данных.")
        if df.shape == (2, 2):
            print("Применение точного критерия Фишера.")
            _, p = fisher_exact(df, alternative='two-sided')
            if not silent:
                print(f"P={p:.3f}")
        else:
            return
    
    if p < 0.01 / bonf:
        print("Отличия статистически значимы.")
    elif p < 0.05 / bonf:
        print("Пограничный случай, есть основания задуматься над наличием различий.")
    else:
        print("Отличия статистически не значимы.")


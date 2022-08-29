import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def barplot_ct(df: pd.DataFrame, colname: str = "Исход", idxname: str = "Метод") -> None:
    """
    Представление таблицы сопряженности в виде barplot с долями исходов для данных методов.
    Вертикальной чертой указан 95% доверительный интервал оценки доли (удвоенная стандартная ошибка доли), 
    если можно выполнить такую оценку (т.е. np > 5 и n(1-p) >5)  
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ndf = df.div(df.sum(axis=1), axis=0).reset_index().melt("index").rename(
        columns={"value": "Доля", "index": idxname, "variable": colname})
    ndf["count"] = ndf[idxname].map(df.sum(1))
    ndf["CI"] = 2 * np.sqrt((ndf["Доля"] * (1 - ndf["Доля"])
                             ) / ndf["count"])  # ci = 2 * se, 95%
    ndf.loc[(ndf["count"] * ndf["Доля"]
             <= 5) | (ndf["count"] * (1-ndf["Доля"]) <= 5), "CI"] = 0  # не выполнено условие применимости

    ndf = ndf.sort_values([idxname, colname])
    sns.barplot(
        data=ndf,
        y="Доля",
        x=colname,
        hue=idxname)

    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=ndf["CI"], fmt="none", c="k")
    ax.set_ylim(0, 1)

    plt.show()

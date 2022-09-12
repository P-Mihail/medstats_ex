def Sp(n1, n2, S1, S2):
    """
    Объединенная оценка стандартного отклонения (pooled standard deviation).
    """
    return (((n1 - 1)*S1**2 + (n2 - 1)*S2**2) / (n1 + n2 - 2))**.5
def inequalities(x):
    """返回三条不等式的松弛量。"""
    x_0, x_1 = x
    inequality_1 = 1 - (x_0 + 2 * x_1)
    inequality_2 = 1 - (x_0 ** 2 + x_1)
    inequality_3 = 1 - (x_0 ** 2 - x_1)
    return inequality_1, inequality_2, inequality_3


def equality(x):
    """返回等式的带符号残差。"""
    x_0, x_1 = x
    equality_1 = 2 * x_0 + x_1 - 1
    return equality_1

def objective(x):
    """目标函数"""
    x_0, x_1 = x
    r = x_1 - x_0 ** 2
    return 100 * r ** 2 + (1 - x_0) ** 2

BOUNDS = ((0.0, 1.0), (-0.5, 2.0))
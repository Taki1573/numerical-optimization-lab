"""数值最优化课程的两个测试函数。输入为实数向量，梯度为同长度的列表。"""

from collections.abc import Iterable
from math import fsum, isfinite, pi, sin


def _vector(x: Iterable[float], min_dimension: int) -> tuple[float, ...]:
    """先复制并校验输入，调用目标与梯度均不会改动求解器的当前点。"""
    values = tuple(float(value) for value in x)
    if len(values) < min_dimension:
        raise ValueError(f"输入向量至少需要 {min_dimension} 维")
    if not all(isfinite(value) for value in values):
        raise ValueError("坐标必须是有限实数")
    return values


def rosenbrock(x: Iterable[float]) -> float:
    values = _vector(x, 2)
    return fsum(
        100 * (right - left**2) ** 2 + (1 - left) ** 2
        for left, right in zip(values, values[1:])
    )


def rosenbrock_grad(x: Iterable[float]) -> list[float]:
    values = _vector(x, 2)
    gradient = [0.0] * len(values)
    # 每个相邻项各影响两个坐标；内部坐标会同时收到左右两项的贡献。
    for i, (left, right) in enumerate(zip(values, values[1:])):
        residual = right - left**2
        gradient[i] += -400 * left * residual + 2 * (left - 1)
        gradient[i + 1] += 200 * residual
    return gradient


def rastrigin(x: Iterable[float]) -> float:
    values = _vector(x, 1)
    # 10(1 − cos(2πx)) = 20 sin²(πx)，在零点附近避免相近数相减。
    return fsum(value**2 + 20 * sin(pi * value) ** 2 for value in values)


def rastrigin_grad(x: Iterable[float]) -> list[float]:
    values = _vector(x, 1)
    return [2 * value + 20 * pi * sin(2 * pi * value) for value in values]

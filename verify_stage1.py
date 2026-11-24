"""验证十维函数值、梯度及其方向导数；独立运行以复现阶段一数据。"""

import argparse
import json
from collections.abc import Callable
from math import fsum, sqrt
from pathlib import Path

import benchmarks

ROOT = Path(__file__).resolve().parent
PROBLEMS = {
    "rosenbrock": (benchmarks.rosenbrock, benchmarks.rosenbrock_grad),
    "rastrigin": (benchmarks.rastrigin, benchmarks.rastrigin_grad),
}
# 误差判据使用绝对项与相对项，适用于小方向导数和大方向导数。
STEP = 1e-5
ATOL = 2e-7
RTOL = 2e-7


def directions(n: int) -> dict[str, list[float]]:
    result = {
        f"coordinate_{i}": [float(i == j) for j in range(n)]
        for i in range(n)
    }
    patterns = {
        "mixed_alternating": [(-1) ** j * (j + 1) for j in range(n)],
        "mixed_sloped": [2 * j - n + 0.5 for j in range(n)],
    }
    for name, vector in patterns.items():
        norm = sqrt(fsum(v * v for v in vector))
        result[name] = [v / norm for v in vector]
    return result


def derivative_check(
    fun: Callable, grad: list[float], x: list[float], direction: list[float]
) -> dict:
    plus = [value + STEP * v for value, v in zip(x, direction)]
    minus = [value - STEP * v for value, v in zip(x, direction)]
    finite_difference = (fun(plus) - fun(minus)) / (2 * STEP)
    analytic = fsum(g * v for g, v in zip(grad, direction))
    absolute_error = abs(analytic - finite_difference)
    allowed_error = ATOL + RTOL * max(abs(analytic), abs(finite_difference))
    return {
        "analytic": analytic,
        "central_difference": finite_difference,
        "absolute_error": absolute_error,
        "allowed_error": allowed_error,
        "passed": absolute_error <= allowed_error,
    }


def verify(config: dict) -> dict:
    n = config["dimension"]
    checks = []
    minimum_checks = []
    for name, (fun, grad) in PROBLEMS.items():
        minimum = [1.0 if name == "rosenbrock" else 0.0] * n
        value = fun(minimum)
        grad_inf = max(map(abs, grad(minimum)))
        minimum_checks.append({
            "function": name, "point": minimum, "value": value,
            "gradient_inf_norm": grad_inf, "passed": value == 0 and grad_inf == 0,
        })
        # 额外的非对称点补充重复初始模式，能发现索引或坐标交换错误。
        points = {**config["initial_points"][name],
                  "asymmetric": [round((j - 4) * 0.173 + 0.11, 3) for j in range(n)]}
        for point_id, point in points.items():
            gradient = grad(point)
            for label, v in directions(n).items():
                checks.append({
                    "function": name, "point_id": point_id, "point": point,
                    "direction_id": label, "direction": v,
                    **derivative_check(fun, gradient, point, v),
                })
    return {
        "dimension": n, "difference_step": STEP, "atol": ATOL, "rtol": RTOL,
        "minimum_checks": minimum_checks,
        "directional_checks": checks,
        "summary": {
            "minima_passed": sum(c["passed"] for c in minimum_checks),
            "minima_total": len(minimum_checks),
            "directions_passed": sum(c["passed"] for c in checks),
            "directions_total": len(checks),
            "max_absolute_error": max(c["absolute_error"] for c in checks),
            "passed": all(c["passed"] for c in minimum_checks + checks),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "stage1-verification.json")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs" / "experiments.json").read_text(encoding="utf-8"))
    report = verify(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if not report["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

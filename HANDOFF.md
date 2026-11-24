# A → B · 无约束实验交接

**交接状态**　A 的第一阶段实现与自动验证已备齐。共同实验配置为提案，B 的确认待记录。

## 调用入口

从仓库根目录运行以下 Python 代码即可对照初始函数值与梯度。后续向 SciPy 传递对应函数对象即可；`Nelder-Mead` 只传目标函数，`trust-exact` 所需 Hessian 由 B 准备并校验。

```python
import json
from pathlib import Path
from benchmarks import rosenbrock, rosenbrock_grad, rastrigin, rastrigin_grad

config = json.loads(Path("configs/experiments.json").read_text(encoding="utf-8"))
for name, fun, grad in [
    ("rosenbrock", rosenbrock, rosenbrock_grad),
    ("rastrigin", rastrigin, rastrigin_grad),
]:
    for start_id, x0 in config["initial_points"][name].items():
        print(name, start_id, fun(x0), max(map(abs, grad(x0))))
```

| 函数 | 起点 | 初始函数值 | 初始梯度无穷范数 |
|---|---|---:|---:|
| Rosenbrock | primary | 2057 | 792 |
| Rosenbrock | secondary | 9 | 2 |
| Rastrigin | primary | 158.063350 | 65.331853 |
| Rastrigin | secondary | 66.659041 | 61.277875 |

## 实验约定草案

1. **使用相同输入。** 两组无约束实验都使用十维。用 `primary` 完成 2 函数 × 3 步长 × 3 梯度，以及 2 函数 × 4 算法的基础对比。`secondary` 用于可选的起点敏感性实验，另列结果。每次运行从配置中复制新的起点。
2. **设置每次运行的预算。** 建议最多 20,000 次迭代、300,000 次目标函数调用，任一耗尽就结束。函数调用次数通过统一目标函数包装器计数，包含初始评估、试探步长与数值差分调用；保留最后一个完整迭代点和退出原因。
3. **记录实际使用的停止条件。** 自编法建议用工作梯度的无穷范数 ≤ 10⁻⁶；BFGS 使用无穷范数和对应容差；trust-exact 使用其原生梯度范数及配置中的容差；Nelder-Mead 使用坐标和函数值容差。各法都在结束后独立计算一次解析梯度无穷范数，将 ≤ 10⁻⁶ 记为通过统一驻点复核。复核开销单列。驻点仍可能是局部极小点，需同时报告最终函数值。
4. **比较每种方法的真实开销。** 记录目标调用 `nfev`、完整梯度调用 `ngev` 和 Hessian 调用 `nhev`，另记迭代数 `nit`。数值梯度的一次完整求值计入 `ngev`，其内部调用目标函数的次数同时计入 `nfev`。失败运行也要记录原因，包括达到预算、线搜索失败或出现非有限值。
5. **核对参数后再批量运行。** 固定步长建议值、Armijo 回溯参数、强 Wolfe 常数以及高低两档差分配置见 JSON。先对照解析梯度实测两档差分误差；需要改动时，在配置中注明改动并统一重跑受影响的组。四法对比中，适用方法共用解析梯度，最速下降与 BFGS 的强 Wolfe 参数保持一致。

结果字段以配置的 `result_fields` 为准。`native_success` 记录原生求解器或自编算法的成功判定，`stationary_postcheck` 记录统一驻点复核。配置及上述数值均为课程小组可协商的实验选择。

## 待确认

- [ ] B 独立运行 README 的两条命令，核对两个极小点及 72 个方向导数结果。
- [ ] B 用上述调用示例核对四个起点，确认或修订预算、停止条件和结果字段。
- [ ] A、B 将确认采用的配置版本和相关 GitHub 提交或 Issue 留在协作看板，再开展正式对比实验。

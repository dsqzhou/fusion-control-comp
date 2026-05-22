# 电源模型

本环境在策略电压指令与 HFM 实际输入之间加入电源响应模型，位于 `environment/power_supply.py`，由 `HFMSocketPredictor.step()` 自动调用。

## 信号链

每个仿真步（1 ms）按以下顺序处理 12 路电压：

```text
U_set[k]  策略给出的电压指令
   |
   v  ① 传输时延：取 U_set[k - d_i]（历史不足时用首步指令填充）
   |
   v  ② 速率限制：|u_r[k] - u_r[k-1]| <= max_change_i
   |
   v  ③ PSM 仿射：U_real[k] = slope_i * u_r[k] + intercept_i
   |
HFM predictor
```

## 参数

| 项 | 说明 |
|----|------|
| PSM slope / intercept | 12 路实测标定常数（替换原占位 K/b） |
| 时延 ch0–10 | 每 episode 在 2–5 ms 内随机 |
| 时延 ch11 (VS) | 每 episode 在 0–1 ms 内随机 |
| 速率限制 | 7 路模板经 12D 映射，每步最大变化量 |

选手仍只输出 12D 电压指令；实际进入 HFM 的电压经过上述模型，且评估时 delay 随机性不对选手公开。

## 示例

```bash
python examples/example_power_supply_step.py
```

![Power supply step response](power_supply_step_response.png)

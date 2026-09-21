# 电源模型

本环境在策略电压指令与 HFM 实际输入之间加入电源响应模型，位于 `environment/power_supply.py`，由 `HFMSocketPredictor.step()` 自动调用。

## 信号链

每个仿真步（1 ms）按以下顺序处理 12 路电压：

```text
U_set[k]  策略给出的电压指令（UOUT）
   |
   v  ① 幅值限幅：clip 到 ±um_i
   |
   v  ② 传输时延：取 U_set[k - d_i]（可配置；默认每步随机采样）
   |
   v  ③ uout_to_urec：相位角速率限制，|Δθ| ≤ 7.2°
   |
   v  ④ PSM：U_real = slope * UREC + intercept（系数来自 .mat；VS 旁路）
   |
HFM predictor
```

旧实现里的「线性 ΔV」是伏特空间限速 `|u[k]-u[k-1]| ≤ Δu_max`。现已替换为装置侧的相位限速，不再叠加一层伏特限速。

## 公式

```text
u_clip,i[k] = clip(U_set,i[k], -um_i, um_i)
u_d,i[k]    = u_clip,i[k - d_i]

θ*_i[k]     = arcsin(clip(u_d,i[k] / um_i, -1, 1))
θ_i[k]      = θ_i[k-1] + clip(θ*_i[k] - θ_i[k-1], -7.2°, 7.2°)
UREC,i[k]   = um_i * sin(θ_i[k])

U_real,i[k] = a_i * UREC,i[k] + b_i    (i = 0..10)
U_real,11[k]= UREC,11[k]               (VS 不走 PSM)
```

首步没有历史 UREC 时，只做幅值限幅，不从 0 爬升。

## 参数

| 项 | 说明 |
|----|------|
| `um_values` | `[1500, 231, 231, 173, 173, 173, 173, 348, 348, 348, 348, 80]`，动作空间同步 |
| `rate_deg` | 默认 `7.2` |
| 时延 ch0–10 | 不传 `delay_s` 时每步在 2–5 ms 内随机 |
| 时延 ch11 (VS) | 不传 `delay_s` 时每步在 0–1 ms 内随机 |
| PSM | `configs/psm/fitting_coefficients_v4.mat`；150 ms 算例用 `*_rampup.mat` |

配置入口：`predictor.power_supply`。`delay_s` 传 12 路秒数则固定延迟，传全 0 可关掉延迟。`shot_id` 会选用对应 mat。

## 示例

```bash
python examples/example_power_supply_step.py
```

![Power supply step response](power_supply_step_response.png)

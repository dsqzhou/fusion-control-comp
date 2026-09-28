# 21311 / 21316 训练算例

装置版本 **v58.6**。给合作训练用，不是复赛正式评测 shot。

训练时建议 `fge_init_config.z_target = -0.015`（略微下单零）。算例配置不要直接当最终目标抄走，150 ms 给出的是变轨程序。

## 电源

动作进入 HFM 前走：

```text
UOUT --clip(um)--> delay(可配置) --> uout_to_urec(7.2°) --> PSM(.mat, VS 旁路) --> HFM
```

| 算例 | PSM 文件 | 说明 |
| --- | --- | --- |
| `21311_150` / `21316_150` | `configs/psm/fitting_coefficients_v4_rampup.mat` | 爬升段 CS 正组 |
| 其余合作算例 | `configs/psm/fitting_coefficients_v4.mat` | CS 负组 |

v4 与 rampup 的斜率相同，只有 CS 截距不同（`+113` vs `-64`）。

满幅电压（动作上下界与 `uout_to_urec` 共用）：

```text
CS, PS1, PS2, PS3, PS4, PS5, PS6, PS7, PS8, PS9, PS10, VS
1500, 231, 231, 173, 173, 173, 173, 348, 348, 348, 348, 80
```

## shot_id

| shot_id | 起始 | LX | bp / q0 |
| --- | --- | --- | --- |
| `21311_150` | 150 ms | `ini_21311_150_v4.mat` | 0.15 / 2.64 |
| `21311_300` | 300 ms | `ini_21311_300_v4.mat` | 0.12 / 1.76 |
| `21311_timevar` | 300 ms | `ini_21311_300_v4.1.mat` | 不传，用 LX 剖面 |
| `21316_150` | 150 ms | `ini_21316_150_v4.mat` | 0.04 / 2.62 |
| `21316_300` | 300 ms | `ini_21316_300_v4.mat` | 0.06 / 1.80 |
| `21316_timevar` | 300 ms | `ini_21316_300_v4.1.mat` | 不传，用 LX 剖面 |
| `21743_200` | 200 ms | `ini_21743_200_v4.mat` | 0.012 / 2.47 |
| `21743_300` | 300 ms | `ini_21743_300_v4.mat` | 0.21 / 2.01 |
| `21743_400` | 400 ms | `ini_21743_400_v4.mat` | 0.28 / 1.84 |
| `21710_200` | 200 ms | `ini_21710_200_v4.mat` | 0.02 / 2.60 |
| `21710_300` | 300 ms | `ini_21710_300_v4.mat` | 0.08 / 2.21 |

L 一律 `ini_L_v4.1.mat`。150 ms 时变把对应 LX 的 `v4` 改成 `v4.1`，并去掉 bp/q0 即可。

## 150 ms 变轨

两炮形状/电流程序相同，差在 bp、q0、signeo：

| 量 | 程序 |
| --- | --- |
| Rmax | 1.13 (150 ms) → 1.26 (320 ms) |
| Rmin | 0.27 (150–200 ms) → 0.29 (300 ms) |
| Z | -0.015 |
| Ip | 320 kA (150) → 475 kA (270) → 500 kA (300) |
| kappa | 1.60 (150–200) → 1.85 (240–300) |

`HFMSimulator` 在 `reference.mode=trajectory` 时会按 `shot_id` 自动填这条轨迹。

## 300 ms

Rmax=1.26，Rmin=0.29，Z=-0.015，Ip=500 kA，kappa=1.85。

训练时可叠加扰动：21316 的 Rmax ≤ 4 cm、21311 的 Rmax ≤ 3 cm，Rmin ≤ 1.5 cm；波形主频率可取 3 Hz 和 30 Hz 附近。

## 平顶维持

不跟爬升程序，参考等于该算例 t=0。

| 档 | shot | 维持目标 |
| --- | --- | --- |
| 300 kA | `21311_150` / `21316_150` | Rmax 1.13，Rmin 0.27，Ip 320 kA，kappa 1.60 |
| 400 kA | `21743_200/300/400` | Rmax 1.24，Rmin 0.31，Ip 400 kA，kappa 1.85 |
| 400 kA | `21710_200` | Rmax 1.25，Rmin 0.29，Ip 400 kA，kappa 1.85 |
| 400 kA | `21710_300` | Rmax 1.27，Rmin 0.29，Ip 400 kA，kappa 1.85 |
| 500 kA | `21311_300` / `21316_300` | Rmax 1.26，Rmin 0.29，Ip 500 kA，kappa 1.85 |

平顶维持加载下面的 yaml。`case_21311_150.yaml` 和 `case_21316_150.yaml` 是爬升程序，Ip 会走到 500 kA，这个任务不要用。

| 用途 | yaml |
| --- | --- |
| 300 kA 训练 | `configs/case_21311_150_hold.yaml` |
| 300 kA 测试 | `configs/case_21316_150_hold.yaml` |
| 400 kA 训练 | `configs/case_21743_300.yaml` |
| 400 kA 测试 | `configs/case_21710_300.yaml` |
| 400 kA 不同 bp | `configs/case_21743_200.yaml`、`configs/case_21743_400.yaml` |
| 400 kA 加测 | `configs/case_21710_200.yaml` |
| 500 kA 训练 | `configs/case_21311_300.yaml` |
| 500 kA 测试 | `configs/case_21316_300.yaml` |

`*_150_hold.yaml` 的 `reference.mode` 是 `hold`，目标停在 t=0。21710 / 21743 和 `*_300.yaml` 的默认轨迹已经是各自的常值。时变把 LX 的 `v4` 换成 `v4.1`，并去掉 bp、q0。

## 入口

```bash
python examples/run_case.py --shot 21311_150 --config configs/case_21311_150_hold.yaml --dry-run
python examples/run_case.py --shot 21743_300 --dry-run
python examples/run_case.py --shot 21311_300 --dry-run
```

爬升轨迹仍是 `configs/case_21311_150.yaml` / `configs/case_21316_150.yaml`，和平顶维持分开。

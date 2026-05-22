# 复赛补充说明

本文件只补充复赛相关流程，原 `README.md` 继续保留作为基础环境、训练、提交说明。

## 复赛任务

复赛评估包含 3 个子任务：

| 任务 | 起始位形 | 目标位形 | 步数 | 服务 |
| --- | --- | --- | --- | --- |
| `F1` | `13705_500` 偏滤器 | `14795_400` 限制器 | 300 | policy1 |
| `F2a` | `13705_500` 偏滤器 | `13906_500` XPT | 500 | policy2 |
| `F2b` | `14795_400` 限制器 | `13906_500` XPT | 500 | policy2 |

`F2a` 和 `F2b` 的目标 XPT 位形一致，只有初始位形不同。

## 推理服务

复赛提交模板支持一个镜像内启动两个 HTTP 服务：

```text
submission/service1.py -> inference1.Policy -> 默认读取 submission/model/policy1.onnx
submission/service2.py -> inference2.Policy -> 默认读取 submission/model/policy2.onnx
```

评测脚本按任务路由：

```text
F1        -> service1
F2a/F2b   -> service2
```

如果两个任务使用同一个模型，也可以只放一个通用 `policy.onnx`，`inference1.py` / `inference2.py` 会在找不到 `policy1.onnx` 或 `policy2.onnx` 时回退到 `policy.onnx`。

## 评测输出

复赛推理结果固定写入：

```text
/saisresult/infer_result.json
```

每个任务的 trajectory 字段对齐复赛评分器：

```text
Ip
lX
nX
Icoil
psia
psib
lcfs_per_step
Xpt_main
Xpt_sec
strike
psiX_main
psiX_sec
```

`run_test.py` 会在每个子任务结束后打印分项得分，最后打印总分：

```text
[subtask-score] F1 ...
[subtask-score] F2a ...
[subtask-score] F2b ...
[total-score] ...
```

## 本地轻量检查

不启动 HFM 时，可以只检查 submission HTTP 层：

```bash
python submission/service1.py
python submission/service2.py
```

完整评估仍需要 HFM socket server 和 `/saisdata/11/inference/run_test.py`。

## 复赛 reference

`/data/dsq/saisdata_second/inference/task_references.yaml` 已改为 `F1/F2a/F2b`：

- `F1`: `13705_500 -> 14795_400`
- `F2a`: `13705_500 -> 13906_500`
- `F2b`: `14795_400 -> 13906_500`

策略收到的 `reference_*` 是训练/推理辅助信号；最终得分以 `target.json` 和 `infer_result.json` 计算。

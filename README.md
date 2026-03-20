# Fusion Control Competition Environment

本目录提供聚变智能控制赛题的本地训练与仿真测试环境。高保真仿真器（HFM）以 Docker 形式封装，Python 环境通过 socket 与仿真器通信，并对选手暴露 Gymnasium 风格接口。核心特点是：

- 观测为全量字典，包含当前步的完整物理量
- 动作为 12 维真实物理电压
- 每一步观测都带当前时刻 `reference`（目标值）
- 提供可选的 flatten 观测和 7 维动作示例（仅作训练参考，最终提交仍需转换为 12 维）
- 最终提交物为 Docker 镜像（包含 ONNX 推理服务），镜像中的 Dockerfile、start_infer.sh、run.sh 等不可修改

## 文档导航

建议按以下顺序阅读：

1. **赛题说明文档**：了解任务背景、评测规则和提交要求
2. **本文档（README.md）**：完成本地安装、仿真器启动、训练与自测流程
3. **docs/reference.md**：查看 observation、action、reference 和配置的详细字段说明
4. **submission/README.md**：了解提交镜像的构建、HTTP 接口、约束条件（不可改的 Dockerfile、启动脚本等）

## 项目目录结构

```text
fusion-control-comp/
├── configs/         # 默认配置
│   ├── env_default.yaml
│   └── shots.yaml
├── docs/
│   └── reference.md
├── environment/     # 环境实现
│   ├── hfm_simulator.py
│   ├── hfm_predictor.py
│   ├── shot_registry.py
│   └── ...
├── examples/        # 示例脚本
│   ├── run_random_policy.py
│   ├── custom_reference_reset.py
│   ├── train_sb3_ppo.py
│   └── ...
├── tools/           # 启停仿真器
│   ├── start_simulator.py
│   └── stop_simulator.py
├── test/            # 环境与 submission 测试
└── submission/      # 最终提交模板（详见 submission/README.md）
    ├── Dockerfile   # 正式提交用，勿改
    ├── start_infer.sh
    ├── run.sh
    ├── inference.py
    ├── service.py
    ├── requirements.txt
    └── model/
```

## 赛题简要说明

本赛题要求选手针对高保真 HFM 仿真环境设计控制策略。环境中关注的控制目标包括：

- 等离子体电流 `Ip`
- 等离子体位置 `R`、`Z`
- 最外闭合磁面接口 `lcfs_points`

任务形态包括两类：

- `hold`：维持初始状态不变，reference 由 reset 初态自动生成
- `trajectory`：跟踪人为指定的逐时刻 reference 轨迹

说明：

- 环境内部测试会覆盖不同 shot 和不同 reference 任务
- 选手训练时可以自行设计 reward
- 正式评测时只接收推理服务，不暴露评测代码

对 AI 选手更重要的直观理解是：

- `Ip` 决定电流跟踪质量
- `R`、`Z` 决定等离子体中心位置
- `lcfs_points` 对应最外闭合磁面边界点，用于描述整体位形轮廓
- 其余观测量主要作为辅助状态，帮助模型判断当前位置、形状变化趋势和控制响应

## 训练建议

为了提高策略对不同初始条件和 reference 任务的泛化能力，建议训练时不要只覆盖 `hold` 任务，也加入一定比例的 `trajectory` 任务。

推荐的基础做法是优先围绕 `Ip` 和 `R` 构造变轨迹训练：

- `Ip`：相对初始目标做 `±100 kA` 范围内的变化
- `R`：相对初始目标做 `±3 cm` 范围内的变化
- `Z`：基础阶段可先保持不变
- `lcfs_points`：基础阶段可先保持不变，后续再逐步增加边界跟踪训练

另外，建议在 `reset_params` 中加入一定随机扰动，用于提升模型对不同初始平衡和隐藏测试条件的鲁棒性。一个实用的起点是：

- `signeo`：相对初始值扰动 `±50%`
- `bp`：相对初始值扰动 `±10%`
- `q0`：相对初始值扰动 `±10%`

这些扰动会影响等离子体的初始位置、形状和后续动态响应。更详细的参数说明见 `docs/reference.md`。

## 仿真环境准备

### 1. 安装 Python 环境

```bash
conda create -n competition_env python==3.11
conda activate competition_env
cd fusion-control-comp/
pip install -e .
```

如果你需要运行 SB3 训练和 ONNX 导出示例：

```bash
pip install stable-baselines3 torch onnx
```
其余训练框架由选手自行选择安装

### 2. 准备 HFM Docker 镜像

请先拉取比赛提供的 环境 镜像，并打上本地标签：

```bash
docker pull crpi-q4qg69o2szruzmaa.cn-beijing.personal.cr.aliyuncs.com/enn_smart/hfm_server:latest
docker tag crpi-q4qg69o2szruzmaa.cn-beijing.personal.cr.aliyuncs.com/enn_smart/hfm_server:latest hfm-matlab-server:latest
```

### 3. 启动仿真器

单容器示例：

```bash
cd tools
python start_simulator.py -n 1 -y
```

关闭：

```bash
python stop_simulator.py
```

说明：

- 默认对外端口从 `2223` 开始
- 端口需与 `configs/env_default.yaml` 中 `predictor.port` 一致
- 多容器并行训练由选手自行配置

## 训练环境：HFMSimulator 与 HFMSocketPredictor

### `HFMSocketPredictor`（底层通信层）

`environment/hfm_predictor.py` 中的 `HFMSocketPredictor` 负责与 Docker 仿真器通过 socket 通信：

- 连接到 Docker HFM 仿真器（地址和端口由 `env_default.yaml` 指定）
- 对每个 `shot_id` 固定维护 `L_addr`、`LX_addr` 等地址映射
- reset 接口仅暴露 `signeo`、`bp`、`q0` 三个扰动参数
- 接收并执行 12 维真实电压动作

### `HFMSimulator`（训练接口）

`environment/hfm_simulator.py` 是选手训练直接使用的 Gymnasium 标准环境，对外暴露：

- `reset(seed, options)`：重置 episode，支持自定义 reset_params 和 reference 模式
- `step(action)`：执行 12 维真实电压动作，返回 observation、reward、terminated 等
- `close()`：清理资源

基本使用示例：

```python
import yaml
from environment import HFMSimulator

with open("configs/env_default.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

env = HFMSimulator(config)
obs, info = env.reset(seed=42)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

**说明**：
- `HFMSimulator` 在内部使用 `HFMSocketPredictor` 与 Docker 仿真器通信
- 选手只需调用 `HFMSimulator` 接口，无需直接使用 `HFMSocketPredictor`

## 关键配置

`configs/env_default.yaml` 中最常用的配置如下：

- `max_steps`：单个 episode 的最大步数，默认 `100`，当前场景最长支持 `499`
- `reference.mode`：默认 reference 模式，`hold` 或 `trajectory`
- `predictor.host`：Docker 仿真器地址，默认 `127.0.0.1`
- `predictor.port`：Docker 仿真器端口，默认 `2223`
- `predictor.timeout`：socket 超时时间
- `predictor.shot_id`：初始化所使用的平衡场景

说明：

- `shot_id` 决定底层仿真初始化所用的平衡配置
- `lcfs_points` / `reference_lcfs_points` 现在固定为 `32 x 2`，不再通过配置项暴露
- 更详细的字段和维度说明见 `docs/reference.md`

### `shot` 是什么

可以把 `shot` 理解为官方提供的预设环境模板。每个 `shot_id` 固定了一组底层平衡配置和默认初始参数，方便选手直接开始训练。

当前提供 4 组 `shot`：

- `13844_500`
- `13844_600`
- `15892_400`
- `15892_500`

建议先固定一个 `shot` 跑通环境就够了。跑通之后，再逐步加入 `reference` 变化和 `reset_params` 扰动来提升泛化能力。

## 环境输入输出说明

### observation

环境输出为全量字典，重点可分为三类：

- 核心控制目标：`Ip`、`R`、`Z`、`lcfs_points`
- 当前步目标：`reference_Ip`、`reference_R`、`reference_Z`、`reference_lcfs_points`
- 辅助状态量：`I_PF`、`Rmax`、`Rmin`、`aminor`、`deltal`、`deltau`、`kappa`、`Bm`、`Fx` 等

说明：

- `observation` 每一步都带当前 reference
- `lcfs_points` 表示最外闭合磁面边界点序列，每个点为一组 `(R, Z)` 坐标
- 底层 环境中边界相关量可表示为 `rB` / `zB` 等坐标数组；在本环境里统一封装为 `lcfs_points`
- 详细字段名、维度和作用关系见 `docs/reference.md`

### action

- 标准动作空间为 12 维
- 每个维度都是真实物理电压
- 最终提交阶段也必须返回 12 维真实电压
- 默认动作上下界见 `environment/hfm_simulator.py`

### reset / reference

`reset(options=...)` 中当前主要开放：

- `reset_params.signeo`
- `reset_params.bp`
- `reset_params.q0`
- `reference_mode`
- `reference`

示例：

```python
options = {
    "reset_params": {
        "signeo": 7.2e6,
        "bp": 0.20,
        "q0": 1.6,
    },
    "reference_mode": "trajectory",
    "reference": {
        "Ip": np.linspace(4.95e5, 5.05e5, 100),
        "R": np.linspace(0.79, 0.81, 100),
        "Z": np.zeros(100),
    },
}
```

说明：

- `hold` 模式下，reference 由 reset 初态自动生成
- `trajectory` 模式下，可显式传入 `Ip`、`R`、`Z`、`lcfs_points` 的逐步目标
- 推荐先从 `Ip` 和 `R` 的轨迹变化开始做训练，再逐步加入更复杂的位形目标
- `reset_params` 可用于对初始平衡做扰动增强，提升泛化能力

## Example 说明

### 基础示例

- `examples/run_random_policy.py`：随机动作跑通环境
- `examples/custom_reference_reset.py`：自定义 reset 参数与 trajectory reference
- `examples/example_reward.py`：starter reward 示例

### 训练相关示例

- `examples/use_flatten_observation.py`：把 dict observation 展平成向量
- `examples/use_7d_action.py`：使用 7 维动作训练示例
- `examples/train_sb3_ppo.py`：训练一个最小 PPO，并导出 ONNX 到 `submission/model/policy.onnx`

### 工具脚本

- `tools/smoke_test_environment.py`：不依赖仿真器的本地结构检查
- `test/check_submission.py`：构建 submission docker 并验证 API
- `test/test_submission.py`：用真实环境联调 submission 服务

## 7 维动作：经验降维参考

比赛的**正式接口仍为 12 维真实电压动作**。`environment/wrappers.py` 中提供了 `Action7DTo12DWrapper` 和 `action_7d_to_12d()`，这是一种**降维参考方案**，仅用于快速实验和早期调参。

**关键约束**：
- 7 维动作映射**仅作训练阶段参考**，不代表正式评测允许 7 维输出
- 如模型内部输出 7 维，**必须在 `submission/inference.py` 中自行映射回 12 维**后再提交
- **正式提交的镜像必须返回 12 维真实电压**

### 7 维到 12 维的映射逻辑（参考）

核心假设：中间 10 维线圈按 5 组对称对处理，每组共享同一电压。映射关系如下：

```
v[0]  → u[0]        # 第 1 维单独控制
v[1]  → u[1] = u[11] # 第 2 和第 12 维共享
v[2]  → u[2] = u[10]
v[3]  → u[3] = u[9]
v[4]  → u[4] = u[8]
v[5]  → u[5] = u[7]
v[6]  → u[6]        # 第 7 维单独控制
```

这种对称保持的做法通常更有利于维持等离子体整体稳定，但**仅供参考，不是比赛规则**。

## ONNX 导出与提交工作流

本地训练完成后，需要将模型导出为 ONNX 格式，并按 `submission/` 目录的要求打包为 Docker 镜像。

### 工作流概览

1. **本地训练**：使用 `HFMSimulator` 环境训练策略
2. **模型导出**：将训练好的模型导出为 ONNX（通常命名 `policy.onnx`）
3. **推理脚本编写**：在 `submission/inference.py` 中实现推理逻辑
4. **本地自测**：运行 `test/check_submission.py` 和 `test/test_submission.py` 验证
5. **镜像构建与提交**：按 `submission/README.md` 的要求构建镜像并推送到阿里云 ACR

### 推荐流程（示例）

```bash
# 1. 启动本地仿真器
cd tools
python start_simulator.py -n 1 -y

# 2. 训练模型（示例：使用 SB3 PPO）
cd ../examples
python train_sb3_ppo.py

# 3. 检查 submission 镜像（仅验证服务是否可用，不运行完整评测）
cd ../test
python check_submission.py

# 4. 使用真实仿真环境联调 submission 服务
python test_submission.py --launch-service docker --service-url http://127.0.0.1:18001

# 5. 如无问题，按 submission/README.md 构建正式镜像并提交
```

**注意**：
- `submission/model/policy.onnx` 需要由选手自己导出和放入
- `submission/inference.py` 中的预处理和推理逻辑可自由实现，但需符合指定的 Policy 接口
- `submission/Dockerfile`、`start_infer.sh`、`run.sh` 及容器内路径均**不可修改**，详见 `submission/README.md`

### Windows 已知问题：OpenMP DLL 冲突
在 Windows 本地联调时，若出现 `libomp.dll` 与 `libiomp5md.dll` 冲突（OpenMP runtime duplicate），可临时设置：
```bash
set KMP_DUPLICATE_LIB_OK=TRUE
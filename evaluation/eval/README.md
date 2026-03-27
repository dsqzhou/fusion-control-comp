# 自定义评测脚本说明

本指南将帮助您理解如何通过自定义 `evaluate.py` 文件来定义您的评测逻辑。您可以通过编写 Python 脚本实现复杂的打分机制，并控制评测结果的输出格式。

## 1. 下载示例文件

首先，请下载并解压我们的示例评测包 `eval.zip`。

解压后，您将看到以下两个个文件：

```
eval/
├── evaluate.py
└── py_entrance.sh
```

## 2. 文件说明与修改指南

### `evaluate.py` (可修改)

`evaluate.py` 是评测的**核心脚本**，主要职责包括：

- 接收评测系统传入的参数  
- 读取标准答案与选手输出  
- 执行自定义评测与打分逻辑  
- 按照约定格式输出评测结果（JSON，stdout）

#### 参数说明

- `arg1`：标准答案文件路径  
- `arg2`：选手提交结果文件路径 

**请注意：** 您可以完全修改此文件的内容，以适应您的评测需求。

### `requirements` (可修改)

在大赛系统后台的**评测模块**中填写，用于声明 `evaluate.py` 所需的第三方 Python 依赖。

**重要提示：**

- 评测系统的 Python 版本为 **3.11**。
- 请务必使用 `==` 精确指定包的版本号，例如：`requests==2.29.0`。
- 以下是评测运行环境已预安装的 Python 包列表，您无需在 `requirements` 中重复声明这些包，除非您需要特定版本：

```
Package                      Version
------------------------------------
certifi                      2022.12.7
charset-normalizer           3.1.0
cmake                        3.26.3
filelock                     3.12.0
fsspec                       2023.6.0
huggingface-hub              0.16.4
idna                         3.4
Jinja2                       3.1.2
joblib                       1.3.1
lit                          16.0.2
MarkupSafe                   2.1.2
mpmath                       1.3.0
networkx                     3.1
numpy                        1.24.3
packaging                    23.1
pandas                       2.0.3
Pillow                       9.5.0
pip                          23.1.2
protobuf                     4.23.4
python-dateutil              2.8.2
pytz                         2023.3
PyYAML                       6.0.1
regex                        2023.6.3
requests                     2.29.0
safetensors                  0.3.1
scikit-learn                 1.3.0
scipy                        1.11.1
sentencepiece                0.1.99
setuptools                   58.1.0
six                          1.16.0
sympy                        1.11.1
threadpoolctl                3.2.0
tokenizers                   0.13.3
torch                        2.0.0+cu118
torchaudio                   2.0.0+cu118
torchvision                  0.15.0+cu118
tqdm                         4.65.0
transformers                 4.31.0
triton                       2.0.0
typing_extensions            4.5.0
tzdata                       2023.3
urllib3                      1.26.15
```

### `py_entrance.sh` (不可修改)

这是一个固定的入口脚本，负责执行您的 `evaluate.py`。

**请注意：** 您**不能**修改此文件或其名称。

## 3. 评测结果输出规范

您的 `evaluate.py` 脚本必须将评测结果以 JSON 格式打印到标准输出（stdout）。输出格式根据评测成功与否有所不同。评分应尽量避免出现负分。

### 3.1 评测的输出

当评测成功完成并计算出分数时，请按照以下格式输出 JSON：

```json
{
  "score": 1.0,
  "scoreJson": {
    "score": 1.0,
  },
  "errorMsg": "",
  "success": true
}
```

```json
{
  "errorMsg": "user input is wrong, please check !",
  "score": 0,
  "scoreJson": {},
  "success": false
}
```

**字段说明：**

- `score` (Number, **必需**): 这是评测的主要分数。请勿删除或修改此字段名。
- `scoreJson` (Object, **必需**): 包含详细分数的 JSON 对象。
  - `score` (Number, **必需**): `scoreJson` 中的主要分数，通常与顶层的 `score` 字段保持一致。请注意保留此 `key`。
  - 其他 `key-value` 对 (例如 `"score1": 1.5`, `"score2": 2.0`): 您可以根据需要添加额外的子分数或评测指标。`value` 必须是数字，不能是数组或其他复杂类型。
  - 错误时请保持 `scoreJson` 为空对象 `{}`
- `errorMsg` (String, **必需**): 将透出给用户的错误信息。
- `success` (Boolean, **必需**): 表示评测是否成功。成功时为 `true`，错误时为 `false`。


## 4. 示例 `evaluate.py` 结构

以下是一个简单的 `evaluate.py` 示例，演示了如何读取输入并输出结果。您需要根据您的实际评测逻辑进行修改。

```python
# coding=utf-8
import sys
import json
import difflib

def main():
    # 初始化默认的失败返回结构（对应 data2）
    output = {
        "errorMsg": "user input is wrong, please check !",
        "score": 0,
        "scoreJson": {},
        "success": False
    }

    try:
        # 检查参数数量
        if len(sys.argv) < 3:
            output["errorMsg"] = "Arguments missing. Usage: python evaluator.py <std_file> <test_file>"
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return

        std_path = sys.argv[1]
        user_path = sys.argv[2]

        # 读取文件内容
        try:
            with open(std_path, 'r', encoding='utf-8') as f:
                std_content = f.read().strip()
            with open(user_path, 'r', encoding='utf-8') as f:
                user_content = f.read().strip()
        except FileNotFoundError:
            output["errorMsg"] = "File not found."
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return

        # 逻辑：如果测试文件内容是特定关键词，触发标准错误输出
        if user_content == "TRIGGER_FAILURE_CASE":
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return

        # 计算分数：使用文本相似度，结果保留6位小数
        matcher = difflib.SequenceMatcher(None, std_content, user_content)
        raw_score = matcher.ratio()
        final_score = round(raw_score, 6)

        # 构建成功返回结构（对应 data1）
        success_output = {
            "score": final_score,
            "scoreJson": {
                "score": final_score,
                # 模拟子分数：score1 为总分的 80%，score2 为总分的 20%
                "score1": round(final_score * 0.8, 6),
                "score2": round(final_score * 0.2, 6)
            },
            "errorMsg": "",
            "success": True
        }

        print(json.dumps(success_output, ensure_ascii=False, indent=2))

    except Exception as e:
        output["errorMsg"] = str(e)
        print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

```

---

## 5. 注意事项
- `evaluate.py` 是唯一需要实现评测逻辑的文件
- 所有评测结果必须严格符合 JSON 输出规范
- 任意异常场景都必须返回 `success = false`
- 推荐在脚本中进行充分的参数与数据校验，以保证评测稳定性
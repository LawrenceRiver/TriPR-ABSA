[English](README.md) | [简体中文](README.zh-CN.md)

# TextGT 语用残差适配器

这是 TextGT 的独立研究分支，用与模型无关的语用残差调整三分类属性级情感分析 logits。

[![CI](https://github.com/LawrenceRiver/TextGT/actions/workflows/ci.yml/badge.svg)](https://github.com/LawrenceRiver/TextGT/actions/workflows/ci.yml)
![Python 3.9 and 3.10](https://img.shields.io/badge/Python-3.9%20%7C%203.10-3776AB?logo=python&logoColor=white)
![PyTorch 1.12.1](https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C?logo=pytorch&logoColor=white)
[![MIT software license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Upstream TextGT](https://img.shields.io/badge/upstream-shuoyinn%2FTextGT-181717?logo=github)](https://github.com/shuoyinn/TextGT)
[![Base paper DOI](https://img.shields.io/badge/base%20paper-10.1609%2Faaai.v38i17.29911-2f6f9f)](https://doi.org/10.1609/aaai.v38i17.29911)

## 与上游项目的关系

本仓库基于 [shuoyinn/TextGT](https://github.com/shuoyinn/TextGT) 独立开发。
上游团队未参与维护，也未对本项目作出背书。原始实现和引用信息见
[NOTICE](NOTICE)。

## 架构

![TextGT 语用残差架构](assets/architecture.png)

适配器读取骨干模型的三个 logits 和一条解析后的样本，返回调整后的 logits，不修改骨干模型。完整接口与公式见 [docs/method.md](docs/method.md)。

## 方法

类别顺序固定为 `positive`、`negative`、`neutral`。

### 事实

事实模块识别属性附近的客观列举和描述。满足置信条件时，它会将正向预测推向中性。

### 比较

比较模块判断当前属性在明确比较中占优还是处于劣势，再按该方向调整正向和负向 logits。

### 强度

强度模块匹配经验证的训练集短语先验，同时处理局部否定、分句边界、距离和属性作用域。未提供先验时，该模块不调整 logits。

### 组合器

组合器按给定顺序运行模块。每个模块都使用上一个模块输出 logits 对应的概率。
可直接调用 `apply_pragmatic_residual`、`apply_batch` 和 `load_prior`。

## Restaurant 已报告结果

SemEval-2014 Restaurant 测试集主表是三个入选检查点的平均值。数值来自 [results/reported_metrics.json](results/reported_metrics.json)。

| 策略 | 准确率 | Macro-F1 |
| --- | ---: | ---: |
| 基线 | 0.8382 | 0.7480 |
| 事实 | 0.8406 | 0.7522 |
| 比较 | 0.8394 | 0.7499 |
| 强度 | 0.8388 | 0.7488 |
| 事实 + 比较 | 0.8418 | 0.7541 |
| 全部 | 0.8424 | 0.7547 |

这些结果来自 3 个入选检查点。发布准备期间没有重新运行完整实验，因此这里不作
最先进性能声明。Laptop、Twitter 和多骨干模型表格及其限制见
[docs/results.md](docs/results.md)。

## 安装

语用残差包支持 Python 3.9 和 3.10。

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

开发检查需要安装 `requirements-dev.txt`。运行完整上游模型训练时，还需要 `requirements.txt` 以及单独获取的数据集和模型资源。

本仓库不分发数据集。请按[上游数据准备说明](https://github.com/shuoyinn/TextGT#priliminaries)操作。上游代码和数据集使用了 [DualGCN](https://github.com/CCChenhao997/DualGCN-ABSA)、[ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) 和 [CDT_ABSA](https://github.com/Guangzidetiaoyue/CDT_ABSA) 的资源；[SSEGCN](https://github.com/zhangzheng1997/SSEGCN-ABSA) 也提供兼容的预处理数据。从原始文本处理数据需要 [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)。非 BERT 训练还需要 [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)，上游命令使用 `glove.840B.300d.zip`。

## 快速开始

下面的例子只启用比较模块，不需要短语先验。

```python
import torch

from pragmatic_residual import apply_pragmatic_residual

sample = {
    "text_list": ["I", "have", "had", "better", "food", "elsewhere", "."],
    "aspect": "food",
    "aspect_post": [4, 5],
}
logits = torch.tensor([1.2, 0.3, 0.1])

adjusted, details = apply_pragmatic_residual(
    logits,
    sample,
    modules=("comparison",),
    return_details=True,
)

print(adjusted)
print(details["actions"])
```

## 配置

残差接口的类别顺序固定为 `positive`、`negative`、`neutral`。`modules` 可传入
`("fact", "comparison", "intensity")` 的任意子集，组合器会按给定顺序运行。
`prior` 可以是通过校验的映射，也可以是 `scripts/build_phrase_prior.py`
生成的 JSON 路径。没有提供先验时，强度模块不调整 logits。

先验构建器默认离线运行。只有显式选择 `--provider deepseek` 进行远程短语标注时，
程序才会读取 `DEEPSEEK_API_KEY`。推理、测试、可视化和离线构建均不需要该密钥。

## 构建先验

构建器只读取训练集。默认使用离线提供方，不发起网络请求：

```bash
python scripts/build_phrase_prior.py \
  --train-file dataset/Restaurants_corenlp/train.json \
  --output artifacts/restaurant-train-prior.json
```

DeepSeek 标注是可选路径。它向已配置的 API 发送由训练集提取的短语候选及汇总计数，不发送完整数据行或标签。API 密钥从环境变量读取，不会写入先验文件：

```bash
export DEEPSEEK_API_KEY="your-key"
python scripts/build_phrase_prior.py \
  --provider deepseek \
  --train-file dataset/Restaurants_corenlp/train.json \
  --output artifacts/restaurant-train-prior.json
```

使用远程路径时，需遵守提供方的隐私政策和服务条款。

## 可复现性

[docs/reproducibility.md](docs/reproducibility.md) 列出了 CPU 检查、上游 GPU 基线命令、数据准备链接和离线先验命令。已报告指标保存在机器可读文件中，不从图片反推。

## 贡献者

软件与图片贡献者见 [AUTHORS.md](AUTHORS.md)。参与项目前请阅读
[CONTRIBUTING.md](CONTRIBUTING.md)。

## 引用

[CITATION.cff](CITATION.cff) 记录软件元数据和上游 TextGT 论文条目，GitHub
与常见引用工具可直接读取。

## 许可证

[MIT 许可证](LICENSE) 适用于上游软件和本仓库的软件修改。第三方数据集、
预训练模型和 GloVe 资源仍按各自条款使用。原创架构图和结果图的版权归相应
贡献者所有，除非另行说明。归属与许可边界见 [NOTICE](NOTICE)。

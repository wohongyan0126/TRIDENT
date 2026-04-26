# TRIDENT

面向多种因果语言模型的文本评估与下游分类实验代码：在基座与微调模型上计算一组可对比的统计与生成质量指标，并在 `method/` 中用传统机器学习做二分类等分析。

## 目录结构

| 目录 | 说明 |
|------|------|
| `gpt/` | GPT-2：训练集构建、微调、全量指标打分（`all_scores_gpt.py` 等） |
| `deepseek/` | DeepSeek：训练集、微调、指标打分脚本 |
| `gpt-neo-1.3B/` | GPT-Neo 1.3B：同上流程 |
| `Qwen2.5-1.5B/` | Qwen2.5-1.5B：微调相关脚本 |
| `method/` | 从各模型导出的 JSON 特征训练随机森林等分类器，以及 ROC/AUC、阈值与评估脚本（`train_model.py`、`eval.py`、`threshold_auc.py` 等） |

各子目录内通常带有 `config` 与 `utils.py`（或同类工具），用于路径、设备与数据列表等配置。

## 指标与数据流（概要）

- **打分脚本**（如 `all_scores_*.py`）：在样本上计算 ROUGE、BERTScore、困惑度、TTR、BLEU、METEOR、n-gram distinct、压缩长度及基于采样的分布类统计等，结果多写入 JSON（含 baseline / 微调对照字段，供 `method` 读取）。
- **method**：读取上述 JSON，拼接特征，训练/评估分类模型并导出报告与曲线。

## 环境依赖（参考）

Python 3，常见依赖包括：`torch`、`transformers`、`numpy`、`scipy`、`scikit-learn`、`pandas`、`rouge-score`、`bert-score`、`nltk` 等。具体版本请按本机 CUDA 与模型仓库要求自行固定。

## 使用说明

1. 按子目录 `config` 填写数据路径、模型名与输出目录。  
2. 按需运行各目录下的 `trainset_builder*` → `finetune*` / `*_fine.py` → `all_scores*.py`。  
3. 将生成的 JSON 汇总到 `method` 期望的目录结构后，运行 `method/train_model.py`、`eval.py` 等完成训练与评估。

（仓库内若含日志或大文件，上传前建议用 `.gitignore` 排除。）

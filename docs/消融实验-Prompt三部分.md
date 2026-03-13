# 消融实验：Prompt 三部分（数据集背景 / 任务指令 / 输入统计）

完整 Prompt 由三部分组成，可分别做消融以分析各自贡献：

1. **数据集背景（Dataset description）**：领域与数据说明（如铁路货运、零膨胀等）。
2. **任务指令（Task description）**：如「预测未来 pred_len 步，给定过去 seq_len 步」。
3. **输入统计特征（Input statistics）**：当前样本的 min、max、median、trend、top-5 lags。

## 三个消融实验

| 消融 | 去掉部分 | 保留部分 | 参数 | checkpoint 路径后缀 |
|------|----------|----------|------|--------------------|
| 1 | 数据集背景 | 任务指令 + 输入统计 | `--ablate_prompt_description` | `_ablate_prompt_desc` |
| 2 | 任务指令 | 数据集背景 + 输入统计 | `--ablate_prompt_task` | `_ablate_prompt_task` |
| 3 | 输入统计特征 | 数据集背景 + 任务指令 | `--ablate_prompt_stats` | `_ablate_prompt_stats` |

## 训练

```bash
# 消融 1：去掉数据集背景
bash scripts/Iron_ablation_prompt_description.sh

# 消融 2：去掉任务指令
bash scripts/Iron_ablation_prompt_task.sh

# 消融 3：去掉输入统计特征
bash scripts/Iron_ablation_prompt_stats.sh
```

## 评估

评估时需带上与训练一致的消融参数，才能找到对应 checkpoint：

```bash
bash scripts/Eval_Iron_ablation_prompt_description.sh
bash scripts/Eval_Iron_ablation_prompt_task.sh
bash scripts/Eval_Iron_ablation_prompt_stats.sh
```

## 论文表述建议

- **完整模型**：Prompt = 数据集背景 + 任务指令 + 输入统计（min/max/median/trend/lags）。
- **消融 1**：去掉数据集背景，仅保留任务指令与输入统计，用于衡量「领域/数据描述」的贡献。
- **消融 2**：去掉任务指令，仅保留数据集背景与输入统计，用于衡量「预测步数/任务形式」等指令的贡献。
- **消融 3**：去掉输入统计，仅保留数据集背景与任务指令（等价于原 `prompt_type=short`），用于衡量「样本级统计特征」的贡献。

将三者与完整模型、以及「无 Prompt」消融（`--ablate_prompt`）一起列表对比，即可系统分析 Prompt 各成分在铁路零膨胀数据上的作用。

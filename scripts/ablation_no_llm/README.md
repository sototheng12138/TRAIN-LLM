# 消融变体 3：去掉整头大象（No LLaMA）——纯 Transformer Encoder

## 目的

- **输入**：纯数值 Patch，无文本 Prompt。
- **底座**：轻量级、随机初始化的标准 Transformer Encoder（无重编程层、无 LLaMA-7B）。
- **输出**：保留辅助分类头（AIN）+ 连续回归头，与主实验一致。
- **损失**：完全保留 L_BCE + L_MSE（JointMaskedAux），含 RMGM 掩码。

用于对比「时序经 LLM 文本对齐」与「时序仅经普通 Transformer」在相同损失与评估设定下的表现。

## 脚本

| 脚本 | 说明 |
|------|------|
| `Iron_no_llm.sh` | 训练，`--model TimeLLM_TransformerOnly`、`--model_comment iron_no_llm`，损失与 Iron.sh 一致 |
| `Eval_Iron_no_llm.sh` | 评估，保留门控与零阈值 |

## 使用

```bash
# 训练（项目根目录）
bash scripts/ablation_no_llm/Iron_no_llm.sh

# 评估
bash scripts/ablation_no_llm/Eval_Iron_no_llm.sh
```

结果目录：`checkpoints/long_term_forecast_Iron_96_48_TimeLLM_TransformerOnly_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Iron_Ore_Transport_Exp_0-iron_no_llm/`

## 说明

本消融无需 `prompt_bank/iron_no_llm.txt`，模型不读取文本。

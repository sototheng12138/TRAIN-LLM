# 消融变体 2：TRAIN-LLM (w/o RMGM) —— 砍掉原始量纲掩码

## 学术意义

证明：若多任务学习时**不**对 0 值做 Mask（数值头在所有点上算 MSE），分类头与回归头会互相干扰，回归头会被 0 值带偏。

## 设定

- **训练**：`JointMaskedMSEAuxBCE` 中数值头 **mask 强制全 1**（即不掩码，全部算 MSE）；辅助头与 BCE 保留。
- **测试**：**保留门控机制**（`--output_aux_semantic`、`--aux_confidence_threshold 0.5`），与主实验一致。

## 预期

测试时门控仍会压制 0，但在「有货日」预测值会明显偏低，整体 RMSE 变差，从而证明 RMGM 掩码既逻辑自洽，也有效保护数值头精度。

## 脚本

| 脚本 | 说明 |
|------|------|
| `Iron_no_rmgm.sh` | 训练，与 Iron.sh 相同但增加 `--model_comment iron_no_rmgm`、`--ablate_no_rmgm` |
| `Eval_Iron_no_rmgm.sh` | 评估，加载 `..._0-iron_no_rmgm`，保留门控 |

## 使用

```bash
# 训练（项目根目录）
bash scripts/ablation_no_rmgm/Iron_no_rmgm.sh

# 评估
bash scripts/ablation_no_rmgm/Eval_Iron_no_rmgm.sh
```

结果目录：`checkpoints/..._0-iron_no_rmgm/eval_result.txt`

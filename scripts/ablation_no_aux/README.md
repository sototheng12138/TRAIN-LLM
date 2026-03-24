# 消融：砍掉辅助头（仅数值头预测）

## 一、单阶段消融

### 目的

- **训练**：不使用「是否有发运」辅助分类头与 `JointMaskedAux` 损失，仅用数值头 + **MaskedMSE** 训练（与完整模型数值头同口径：有货日才算 MSE，没货日不扣分）。
- **测试**：不做辅助头置信度门控，只由数值头做正常预测（与主实验 Eval_Iron.sh 中「辅助头 P(有发运)<0.5 则置零」对比）。

### 脚本

| 脚本 | 说明 |
|------|------|
| `Iron_no_aux.sh` | 训练，对应 `scripts/Iron.sh` 去掉 `--use_aux_loss`、`--loss JointMaskedAux`，改为 `--loss MaskedMSE`，`model_comment=iron_no_aux` |
| `Eval_Iron_no_aux.sh` | 评估，加载 `..._0-iron_no_aux` 下 checkpoint，不启用辅助头语义与门控 |

### 使用

1. 训练（在项目根目录执行）  
   `bash scripts/ablation_no_aux/Iron_no_aux.sh`

2. 评估  
   `bash scripts/ablation_no_aux/Eval_Iron_no_aux.sh`

结果写入：`checkpoints/..._0-iron_no_aux/eval_result.txt`

---

## 二、两阶段法消融（去掉 AIN 辅助头与对应评估截断）

与主实验两阶段法（`Iron_stage1.sh` → `Iron_stage2.sh`）结构一致，但**训练**去掉辅助头与联合损失、**评估**去掉辅助头门控与零阈值截断，用于对比「两阶段 + AIN + 门控」的贡献。

### 目的

- **Stage 1**：与 `Iron_stage1` 相同（Mask 全 1、MLP 回归头、ablate_no_rmgm），但不用 AIN 与 `JointMaskedAuxMAE`，仅数值头 + **MaskedMAE**，保存为 `iron_stage1_no_aux`。
- **Stage 2**：从 `iron_stage1_no_aux` 加载，开启 RMGM（不加 ablate_no_rmgm），仍仅 **MaskedMAE** 微调，保存为 `iron_stage2_no_aux`。
- **测试**：不启用辅助头语义与门控（无「P(有发运)<阈值→置零」），且 `--zero_threshold 0` 关闭零阈值，即**完整去掉辅助头与对应评估截断**。

### 脚本

| 脚本 | 说明 |
|------|------|
| `Iron_stage1_no_aux.sh` | 两阶段 Stage 1 无辅助头，`model_comment=iron_stage1_no_aux` |
| `Iron_stage2_no_aux.sh` | 两阶段 Stage 2 无辅助头，从 `...-iron_stage1_no_aux` 加载，`model_comment=iron_stage2_no_aux` |
| `Eval_Iron_stage2_no_aux.sh` | 评估 `iron_stage2_no_aux`，无辅助头门控、无零阈值（`--zero_threshold 0`） |

### 使用

1. Stage 1（在项目根目录执行）  
   `bash scripts/ablation_no_aux/Iron_stage1_no_aux.sh`

2. Stage 2（需先完成 Stage 1）  
   `bash scripts/ablation_no_aux/Iron_stage2_no_aux.sh`

3. 评估  
   `bash scripts/ablation_no_aux/Eval_Iron_stage2_no_aux.sh`

结果写入：`checkpoints/..._0-iron_stage2_no_aux/eval_result.txt`

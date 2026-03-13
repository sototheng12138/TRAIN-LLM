# 联合损失 JointMaskedMSEAuxBCE 说明

## 设计逻辑

- **输出 A（数值头）**：一组具体吨数，如 [150.5, 8000.2, -45.3, 30.1, ...]。训练时用**纯 Masked MSE**，在有货日（target>0）往死里拟合；**没货日不扣分**，数值头可以“放飞”（毛刺、负数都行）。
- **输出 B（辅助头）**：0～1 的置信度 C_aux，如 [0.05, 0.92, 0.11, 0.02, ...]。经 Sigmoid/Softmax + **BCE** 严格训练，清楚哪天是真有货（≈1）、哪天是杂音（≈0）。

训练时**两路必须分道扬镳**，不能合并、不能相乘：
- **output_num** → 只进 **Masked MSE**，用真实发运量做 mask：没货日预测多少都不扣分，有货日必须拟合吨数。
- **output_aux** → 只进 **BCE**，用 0/1 标签考核，把没货日概率压向 0、有货日推向 1。

推理时用 **C_aux 裁决**：C_aux < 阈值（如 0.5）则整窗置 0，抹掉数值头在没货日的杂音。

## 是什么

- **数值头**：**纯 Masked MSE**，只在 target>0 的位置算 `(pred - target)^2` 并求均；**绝对不加**对 target=0 的 loss_zero。
- **辅助头**：**BCE**，P(有发运)=softmax(logits)[1]，标签 0/1。
- **总损失**：`L = L_num + λ * L_aux`，其中 `L_num` 仅为上述 Masked MSE。

## 和现有方案的区别

| 项目 | ZeroInflated + CrossEntropy(aux) | JointMaskedAux |
|------|-----------------------------------|----------------|
| 数值头对「真实=0」 | 加大权重，逼模型学会预测 0 | 不扣分、不约束（mask 掉） |
| 零的决策 | 数值头也要学 0 | 完全交给辅助头；推理时用 C_aux 置零 |
| 辅助头 | CrossEntropy(logits, label) | BCE(softmax(logits)[1], label) |

## 何时用

- **JointMaskedAux**：要的就是「数值头只管有货日的量，没货日可放飞；辅助头学 0/1，推理时用 C_aux 裁决置零」。
- **ZeroInflated + 辅助 CE**：希望数值头也在真实为 0 时尽量预测 0 时使用。

## 使用方式

训练时加上 `--loss JointMaskedAux`，且必须同时开启 `--use_aux_loss`：

```bash
--loss JointMaskedAux \
--use_aux_loss \
--aux_loss_weight 0.2
```

验证/测试阶段用 MSE 算 loss；推理时用 `--aux_confidence_threshold 0.5` 等做整窗置零。

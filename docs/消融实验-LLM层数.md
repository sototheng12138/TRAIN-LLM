# 消融实验：LLaMA 层数（32 层 → 8 层）

## 目的

对比「完整 32 层 LLaMA」与「仅用 8 层 LLaMA」在相同数据与任务上的表现，用于分析语言模型深度对时序预测的贡献及计算/显存权衡。

## 设置

- **完整模型**：`--llm_layers 32`（与 Iron.sh 一致），checkpoint 路径为 `...-iron/`。
- **消融**：`--llm_layers 8`，checkpoint 路径为 `...-iron_ablate_llm8/`。

模型在加载 LLaMA 时通过 `config.num_hidden_layers = llm_layers` 控制层数，其余结构（Patch、重编程、Prompt、Head）不变。

## 训练

```bash
bash scripts/Iron_ablation_llm8.sh
```

## 评估

```bash
bash scripts/Eval_Iron_ablation_llm8.sh
```

评估时必须传入 `--llm_layers 8`，且其他参数与训练一致，才能正确找到 `...-iron_ablate_llm8` 下的 checkpoint。

## 论文表述建议

- **完整**：使用 32 层 LLaMA 作为时序表示的后端。
- **消融**：将 LLaMA 层数减至 8 层，其余配置不变，用于衡量「LLM 深度」对预测精度与训练/推理成本的影响；8 层通常显存更低、速度更快，可对比精度下降幅度。

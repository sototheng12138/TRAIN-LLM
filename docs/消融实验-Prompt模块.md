# 消融实验：Prompt 模块（无 Prompt，仅重编程作为 LLM 输入）

## 目的

验证「Prompt + 重编程 patch」一起送入 LLM 时，**文本 Prompt**（数据说明、任务说明、输入统计等）是否必要。去掉 Prompt 后，仅将重编程后的时间序列 patch 表示作为 LLM 的输入，不进行任何语言侧的条件化。

## 完整 Time-LLM 流程（保留 Prompt）

1. 根据输入序列构造 prompt 文本（数据描述、任务说明、min/max/median/trend/lags 等）。
2. 对 prompt 做 tokenize 并取 embedding，得到 `prompt_embeddings`。
3. Patch Embedding + 重编程层得到 `enc_out`。
4. `llama_enc_out = cat([prompt_embeddings, enc_out], dim=1)`，送入 LLM。

## 消融设置（去掉 Prompt）

- **去掉**：不构造 prompt、不调用 tokenizer、不生成 `prompt_embeddings`。
- **保留**：Patch Embedding、重编程层（文本原型交叉注意力）不变。
- **输入**：`llama_enc_out = enc_out`，即仅将重编程后的 patch 序列作为 LLM 的输入，序列长度为 patch 数，无任何前置 prompt token。

即：**只保留“重编程”作为输入，直接进 LLM，不做任务/数据的语言描述。**

## 使用方式

### 训练（消融模型）

```bash
bash scripts/Iron_ablation_prompt.sh
```

- 与 `Iron.sh` 仅多 `--ablate_prompt`，其余超参与数据一致。
- checkpoint 保存在：`checkpoints/<setting>-iron_ablate_prompt/`。

### 评估

```bash
bash scripts/Eval_Iron_ablation_prompt.sh
```

- 必须带 `--ablate_prompt`，且其他参数与训练一致，才能正确找到并加载 `...-iron_ablate_prompt` 下的 checkpoint。

## 参数说明

| 参数 | 说明 |
|------|------|
| `--ablate_prompt` | 启用消融：不建 prompt，仅用重编程后的 patch 作为 LLM 输入。 |

## 论文表述建议

- **完整模型**：时间序列经重编程后，与描述任务与输入统计的 prompt embedding 拼接，再送入 LLM。
- **消融**：去掉 prompt，仅将重编程后的 patch 序列作为 LLM 输入，不提供任务或数据的语言描述。
- 对比两者在 Vali/Test 上的指标，即可量化「文本 Prompt」对预测性能的贡献。

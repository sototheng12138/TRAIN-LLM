# 实验设置与基准对比说明（论文用）

本文档统一描述所有模型的**数据、划分、任务与评估指标**，保证结果可直接对比，便于论文撰写。

---

## 1. 数据集与任务

| 项目 | 说明 |
|------|------|
| **数据集** | 铁路货运量日度数据（2023–2025），文件：`2023_2025_Iron_data.csv` |
| **变量** | 4 个通道：铁矿石、铁矿砂、铁矿粉、铁精矿粉（多变量预测多变量，M） |
| **任务** | 长期预测（long-term forecast）：给定过去 `seq_len` 天，预测未来 `pred_len` 天 |
| **数据特性** | 零膨胀、存在大量不规则零值（缺失或停运），采用 Zero-Inflated 损失（主模型及部分基线） |

---

## 2. 数据划分与预处理

所有模型采用**相同**划分与预处理，确保可比。

| 项目 | 设置 |
|------|------|
| **划分比例** | Train : Validation : Test = **7 : 1 : 2**（按时间顺序，无打乱） |
| **训练集** | 仅使用前 70% 进行训练/拟合，**不使用验证集参与训练** |
| **测试集** | 最后 20% 用于评估，所有模型在同一测试集上报告指标 |
| **标准化** | 使用 **StandardScaler**，**仅在训练集上 fit**，再对 train/val/test 做 transform，避免未来信息泄露 |
| **逆变换** | 评估时可用 `inverse_transform` 将预测与真值还原到原始量纲（万吨/吨），用于报告原始尺度 MAE/RMSE |

---

## 3. 默认预测设定（主实验）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **seq_len** | 96 | 输入序列长度（历史 96 天） |
| **label_len** | 48 | 解码器起始 token 长度（与 seq_len 同框架时多为 48） |
| **pred_len** | 48 | 预测长度（未来 48 天） |

消融实验另设：**seq_len** ∈ {48, 72, 120, 144}，**pred_len** ∈ {24, 48, 72}，对应脚本与 checkpoint 单独命名（如 `Iron_48_48`、`Iron_72_48`、`iron_pl24`、`iron_pl72`）。

---

## 4. 评估指标（统一口径）

所有模型在**同一测试集**上计算以下指标，保证直接可比。

### 4.1 标准化空间（推荐用于跨模型对比）

- **定义**：在标准化后的数据空间内计算 MAE/RMSE；或等价地，在原始空间计算 MAE/RMSE 后**除以各通道训练集标准差**。
- **公式**（单通道）：  
  \(\text{MAE}_{\text{std}} = \text{MAE}_{\text{orig}} / \sigma_{\text{train}}\)，  
  \(\text{RMSE}_{\text{std}} = \text{RMSE}_{\text{orig}} / \sigma_{\text{train}}\)。
- **报告**：  
  - **Test 整体**：对所有通道、所有测试窗口取平均后的标准化 MAE、RMSE。  
  - **各指标**：铁矿石、铁矿砂、铁矿粉、铁精矿粉各自的标准化 MAE、RMSE。
- **用途**：消除量纲差异，便于与 Time-LLM、DLinear、Autoformer 及传统基线（XGBoost、ARIMA、LSTM、Prophet）直接对比。

### 4.2 逆变换到原始量纲

- **定义**：将模型输出与真实值经数据集 `inverse_transform` 还原为原始单位后，再计算 MAE/RMSE。
- **报告**：Test 整体及各指标的 MAE、RMSE（万吨/吨）；以及各指标 **MAE / 训练集 std**（与标准化空间各通道 MAE 一致）。

### 4.3 实现对应关系

- 深度学习模型（Time-LLM、DLinear、Autoformer）：模型在标准化空间输出，评估脚本先算标准化空间 MAE/RMSE，再对 pred/true 做 `inverse_transform` 后算原始量纲 MAE/RMSE 及 MAE/std。
- 传统基线（XGBoost、ARIMA、LSTM、Prophet）：在原始空间或标准化空间得到预测后，用**同一**训练集 std 做除法得到标准化 MAE/RMSE，与上述定义一致。

---

## 5. 各模型基准设置摘要

### 5.1 主模型（Time-LLM）

| 项目 | 设置 |
|------|------|
| 脚本 | `scripts/Iron.sh`（训练）、`scripts/Eval_Iron.sh`（评估） |
| 训练数据 | 前 70%，滑动窗口 (seq_len=96, pred_len=48)，逐通道采样（非 multivariate） |
| 损失 | Zero-Inflated Loss |
| 正则 | dropout=0.15, weight_decay=1e-5 |
| 其他 | RevIN（实例归一化）、patch 嵌入、LLM 对齐等，见正文 |

### 5.2 深度时序基线

| 模型 | 训练数据 | 说明 |
|------|----------|------|
| **DLinear** | 前 70%，seq=96, pred=48，逐通道，与主模型一致 | 与主模型**完全同一数据与评估**，可比性最高 |
| **Autoformer** | 前 70%，seq=96, pred=48，**multivariate**（每样本 4 通道） | 输入形式与主模型不同，对比时需在文中说明 |

### 5.3 传统/统计与机器学习基线

| 模型 | 训练数据 | 预测与评估 |
|------|----------|------------|
| **XGBoost** | 仅前 70% 构造 (X, y) 窗口，seq=96 → pred=48 | 多输出回归，在所有测试窗口上评估，标准化 MAE/RMSE = 原始 / train_std |
| **ARIMA** | 仅前 70% 拟合 | 滚动预测：对每个测试窗口用「训练集 + 该窗口前」拟合，预测 48 步，再在所有窗口上平均 MAE/RMSE，除以 train_std |
| **LSTM** | 仅前 70% 构造 (X, y)，seq=96 → pred=48 | 全连接输出 48 步，在所有测试窗口上评估，标准化 MAE/RMSE = 原始 / train_std |
| **Prophet** | 仅前 70% 拟合 | 滚动预测：对每个测试窗口用「训练集 + 该窗口前」拟合，预测 48 步，再平均并除以 train_std |

实现与运行：`run_baseline_xgb_arima.py`（包含 XGBoost、ARIMA、LSTM、Prophet，统一 7:1:2、仅 70% 训练）。**公平对比**：默认使用全部测试窗口（与主模型 test set 样本数一致）；加 `--quick` 时 ARIMA/Prophet 仅用前 30 窗以省时，此时与主模型非同一批窗口，仅作快速调试。

---

## 6. 公平对比要点（论文表述建议）

1. **数据**：所有模型使用同一数据集、同一 7:1:2 划分，测试集为最后 20%。
2. **训练**：基线（DLinear、XGBoost、ARIMA、LSTM、Prophet）均**仅使用前 70% 数据**进行训练或拟合，与主模型一致。
3. **任务**：默认输入长度 96、预测长度 48；消融中另行说明 seq_len / pred_len 变化。
4. **指标**：以**标准化空间**的 Test 整体 MAE、RMSE 作为主要对比指标；可补充逆变换后的 MAE/RMSE 及各通道 MAE/std。
5. **复现**：训练与评估脚本、数据路径、随机种子（若固定）在附录或代码仓库中给出，便于复现。

---

## 7. 结果文件位置（便于制表）

- **主模型与深度基线**：  
  `checkpoints/<setting>-<model_comment>/eval_result.txt`  
  其中含【不逆变换，标准化空间】与【逆变换到原始量纲】两段。
- **传统基线**：  
  运行 `python run_baseline_xgb_arima.py`（脚本位置：项目根目录）后，结果会写入 **`baseline_eval_result.txt`**（与主模型 eval_result.txt 格式对齐），终端也会打印各模型整体标准化 MAE/RMSE。公平对比请**不要**加 `--quick`（默认使用全部测试窗口）。  
  **若未看到 Prophet**：多为未安装 `prophet`（`pip install prophet`）；脚本会在开头打印「未安装 prophet，Prophet 已跳过」。

---

## 8. 符号与术语表（可选，用于论文）

| 符号/术语 | 含义 |
|-----------|------|
| \(T\), \(L\) | 输入长度（seq_len）、预测长度（pred_len），默认 \(T=96,\,L=48\) |
| MAE / RMSE | 平均绝对误差 / 均方根误差 |
| 标准化 MAE/RMSE | 原始 MAE/RMSE 除以对应通道训练集标准差，或直接在标准化空间计算 |
| Zero-Inflated | 针对零膨胀序列的损失函数 |
| RevIN | 可逆实例归一化（主模型默认使用；消融中有关闭版本） |

---

## 9. 消融实验设计

### 9.1 已完成的消融

| 消融项 | 说明 | 脚本/comment 示例 |
|--------|------|-------------------|
| **Prompt** | 完整 vs 简短（description+task only） | `prompt_type=short` |
| **Loss** | Zero-Inflated vs MASE | `loss=MASE` |
| **seq_len** | 输入长度 | 48, 72, 96, 120, 144（iron_48, iron_72, iron_144, iron_seq120） |
| **pred_len** | 预测长度 | 24, 48, 72（iron_pl24, iron_pl96 等） |
| **RevIN** | 是否使用可逆实例归一化 | `no_revin=True` |

### 9.2 Patch 消融（建议方案）

Patch 将输入序列切成重叠/不重叠的片段，再映射为 token。代码中相关参数：

- **patch_len**：每个 patch 的时间步数（默认 16）
- **stride**：滑动步长（默认 8，即 50% 重叠）
- **patch_nums**：`(seq_len - patch_len) / stride + 2`（seq_len=96 时默认 12）
- **use_multiscale_patch**：多尺度 patch（多组 (patch_len, stride) 融合）vs 单尺度

**推荐做的两类消融：**

1. **单尺度 patch_len（保持 stride = patch_len/2）**

   | patch_len | stride | patch_nums (seq=96) | 含义 |
   |-----------|--------|----------------------|------|
   | 8 | 4 | 24 | 更细粒度，token 数多 |
   | **16** | **8** | **12** | 默认（基线） |
   | 24 | 12 | 8 | 更粗粒度 |
   | 32 | 16 | 6 | 最粗，token 数少 |

   训练/评估脚本：在 `Iron.sh` 基础上增加 `--patch_len 8 --stride 4`（或 24/12、32/16），并设独立 `model_id`、`model_comment`（如 `Iron_96_48_patch8`、`iron_patch8`），便于和默认 patch_len=16 对比。

2. **多尺度 vs 单尺度**

   - 单尺度：当前默认，`use_multiscale_patch=False`（不传即可）。
   - 多尺度：`--use_multiscale_patch`，内部为多组 (8,4)、(16,8)、(32,16) 融合。
   - 对比「单尺度 patch_len=16」与「多尺度」即可说明多尺度 patch 的增益。

**实现注意：**  
- `run_main.py` 已支持 `--patch_len`、`--stride`、`--use_multiscale_patch`；评估时 `run_eval.py` 需传入与训练一致的 `patch_len`、`stride`（及是否多尺度），否则 head 维度不匹配。

### 9.3 未来工作 / 局限与展望（可写在论文末尾）

以下内容**不必**在正文中做完整实验，可在「未来工作」「局限性」或「展望」中简要提及，作为后续可做方向：

| 方向 | 简要说明 | 论文中可写表述示例 |
|------|----------|---------------------|
| **LLM 层数** | 探索不同 `llm_layers` 对性能与效率的权衡 | “未来可系统比较不同 LLM 深度对时序表示的影响。” |
| **d_model / d_ff** | 时序适配器容量（如 32/128 vs 64/256） | “扩大时序编码维度以提升长序列建模是值得探索的方向。” |
| **Dropout / 正则** | 不同正则强度在零膨胀数据上的表现 | “针对零膨胀数据的正则与损失设计可进一步优化。” |
| **时间编码与 Freq** | timeF / fixed、h vs d 等 | “时间特征编码方式与数据频率的匹配可做进一步分析。” |
| **多数据集 / 多领域** | 在更多行业、更长序列上验证 | “在更多领域与更长预测 horizon 上的泛化性有待验证。” |

正文消融建议**只保留**：Prompt、Loss、seq_len、pred_len、RevIN，以及你计划做的 **Patch**（单尺度 patch_len 或 多尺度 vs 单尺度）。其余放到未来展望即可，避免实验过多、主线不清。

---

*文档版本与代码库一致，实验设置以各脚本及 `run_eval.py`、`run_baseline_xgb_arima.py` 为准。*

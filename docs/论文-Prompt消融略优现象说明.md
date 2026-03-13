# Prompt 消融略优现象：论文中如何表述

## 现象（基于你当前的评估结果）

- **完整模型**（Prompt + 重编程）：Test MAE = 0.7747，RMSE = 1.2371（标准化空间）
- **消融（无 Prompt，仅重编程）**：Test MAE = 0.7568，RMSE = 1.2149（标准化空间）

即：去掉文本 Prompt 后，在测试集上略优（MAE/RMSE 均略有下降）。**这一发现值得在论文中正面、诚实地说清楚，而不是回避。**

---

## 可能原因（小数据 + 间歇性）

1. **数据规模小**  
   训练样本有限时，长 Prompt（数据描述、任务说明、min/max/median/trend/lags 等）会占用大量 token 和注意力。模型容易对“如何读 prompt”过拟合，而不是对时间序列模式过拟合，从而在测试集上略逊。

2. **零膨胀 / 间歇性**  
   铁路货运存在大量无发运日（0 值）。Prompt 里的统计量（如均值、趋势）更多反映“整体水平”，对“何时为 0、何时非 0”的细粒度模式帮助有限。重编程后的 patch 直接编码了局部时序与稀疏结构，在无 prompt 时，LLM 的注意力更集中在这些 patch 上，反而有利于间歇性序列的预测。

3. **Prompt 作为“干扰”**  
   预训练 LLM 更习惯处理语言。前置长 prompt 可能把部分注意力吸引到语义/统计描述上，相对削弱了对时间序列 patch 的利用。去掉 prompt 后，输入全是重编程得到的时序表示，信号更集中，在小数据场景下有时会略好。

4. **任务与数据匹配**  
   你的任务本质是“多变量、零膨胀、铁路货运量”的回归，时序结构本身已经通过 patch + 重编程注入 LLM。在某些设定下，额外的语言描述未必带来边际收益，甚至带来额外方差（尤其是数据少时）。

---

## 论文中如何写（建议）

### 1. 消融实验部分（客观陈述 + 简短解释）

**中文示例：**

> 表 X 中，去掉文本 Prompt、仅以重编程后的 patch 作为 LLM 输入的消融模型（w/o Prompt）在测试集上略优于完整模型（MAE 0.7568 vs 0.7747，RMSE 1.2149 vs 1.2371）。我们将其归因于：（1）本数据集规模有限，长 Prompt 易引入过拟合或分散对时序表示的利用；（2）铁路货运量具有零膨胀与间歇性，重编程 patch 已编码局部时序与稀疏结构，在此设定下额外语言描述未必带来边际增益，甚至可能稀释模型对 patch 的注意力。该结果表明，在**小样本、零膨胀/间歇性**场景中，时间序列的“重编程表示”本身已具有较强的可预测性，而文本 Prompt 的收益可能随数据规模与任务特性而变化，值得在更大规模或不同领域数据上进一步验证。

**English (for paper):**

> In Table X, the variant without the text prompt (w/o Prompt), where only the reprogrammed patch embeddings are fed to the LLM, slightly outperforms the full model on the test set (MAE 0.7568 vs 0.7747, RMSE 1.2149 vs 1.2371). We attribute this to: (1) the limited size of the dataset, where a long prompt may encourage overfitting or dilute the use of the time-series representation; (2) the zero-inflated and intermittent nature of railway freight volumes, for which the reprogrammed patches already encode local temporal and sparse structure, so that additional linguistic conditioning may not add marginal benefit and can shift attention away from the patches. This suggests that in **small-sample, zero-inflated/intermittent** settings, the reprogrammed time-series representation alone can be highly predictive, and the benefit of the text prompt may depend on data scale and task characteristics—worthy of further study on larger or different domains.

### 2. 讨论 / 局限性（可选）

- 明确说明：当前实验是在**单一、小规模、零膨胀**铁路数据集上的结果；在更大规模或更连续的数据上，Prompt 仍可能带来增益。
- 强调：消融的目的是**理解各模块的作用**，发现“在本设定下无 Prompt 略优”本身就是一个有价值的结果，有助于后续在类似场景中做模型简化或部署选择。

### 3. 避免的写法

- 不要只报完整模型指标、不报无 Prompt 消融，或把无 Prompt 结果藏在附录而不解释。
- 不要写成“我们的模型在消融下变差了”的消极语气；应写成“消融表明，在本任务与数据特性下，去掉 Prompt 略优，可能原因是……”

---

## 小结

- **现象**：无 Prompt 的消融在本数据上略优于完整模型（MAE/RMSE 略低）。
- **原因**：小数据 + 零膨胀/间歇性 + Prompt 可能分散注意力或引入过拟合。
- **论文态度**：如实报告消融结果，用 1–2 段解释原因并限定在“小样本、零膨胀/间歇性”场景，同时指出在更大规模或不同任务上 Prompt 仍可能有益。这样既诚实又有信息量，审稿人更容易接受。

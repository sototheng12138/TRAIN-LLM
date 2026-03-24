#!/bin/bash
# =============================================================================
# 推理阶段：用 z 执行裁决（Aux head confidence → 是否整窗置 0）
# 与训练脚本 Iron.sh 配合，构成完整的「z 的生命周期」框架。
# =============================================================================
#
# 【z 的生命周期】
#
# 1. 训练阶段（由 Iron.sh 完成）：z 用来“挨打”并“进化”
#    - 输入训练集 → 算出 z → Softmax 得到置信度 C
#    - 有伪标签 c_disp（1=有货，0=没货），BCE Loss 惩罚错误：没货时 z 应小，有货时 z 应大
#    - 反向传播更新 W、b，让模型学会区分“发运前兆”与“空闲前兆”
#    → 论文必须强调：清零动作在推理阶段执行，但判断能力（z）是在训练阶段用真实业务数据严格训练得到的
#
# 2. 推理阶段（本脚本）：z 用来“执行裁决”
#    - 输入未见过的历史数据 → 算出 z（无标签，仅凭训练好的权重）
#    - C = P(有发运)；若 C < 0.5，判定数值头的小值/毛刺为“幻觉”，对该窗口执行强制清零
#    - 输出：数值预测 + 系统决策说明（调度员可直接采用）
#
# 3. 使用顺序
#    先训练： bash scripts/Iron.sh
#    再裁决： bash scripts/Eval_Iron_AuxDecision.sh
# =============================================================================
# 默认 CPU；有 GPU 可去掉 --device cpu 并设置 export CUDA_VISIBLE_DEVICES=2
# export CUDA_VISIBLE_DEVICES=2

python run_eval.py \
  --device cpu \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path 2023_2025_Iron_data.csv \
  --model_id Iron_96_48 \
  --model TimeLLM \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --d_model 32 \
  --d_ff 128 \
  --factor 3 \
  --des 'Iron_Ore_Transport_Exp' \
  --model_comment iron \
  --llm_layers 32 \
  --itr 1 \
  --output_aux_semantic \
  --aux_confidence_threshold 0.5 \
  --zero_threshold -1

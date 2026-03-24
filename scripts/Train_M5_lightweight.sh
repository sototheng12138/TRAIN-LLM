#!/bin/bash
# 轻量级领域微调 (Lightweight Domain Fine-tuning)：冻结 LLM 底座，仅用目标域（M5）完整训练集
# 对重编程层与预测头进行参数更新。模型内 LLM 已默认冻结，本脚本加载主模型后在 M5 上微调。
# 与 Few-shot 区别：使用 100% M5 训练集、较多 epoch（如 15），仍只更新 reprogramming + head。
# 使用 m5_10ch.csv（10 组），主模型预测头与通道数无关，直接 enc_in=10 微调即可。若无 m5_10ch.csv 则先运行 convert_m5_for_iron.py。

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}

model_name=TimeLLM
train_epochs=15
learning_rate=0.00001
llama_layers=32
percent=100

master_port=29531
num_process=2
batch_size=4
d_model=32
d_ff=128

comment='iron_stage2_linear_m5_lightweight'
MAIN_CKPT_DIR="./checkpoints/long_term_forecast_Iron_96_48_TimeLLM_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Iron_Ore_Transport_Exp_0-iron_stage2_linear"

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
DATA_10CH="$ROOT/dataset/m5_10ch.csv"
if [ ! -f "$DATA_10CH" ]; then
  echo "生成 m5_10ch.csv ..."
  python "$SCRIPT_DIR/convert_m5_for_iron.py"
fi
if [ ! -f "$MAIN_CKPT_DIR/checkpoint" ]; then
  echo "错误: 主模型 checkpoint 不存在: $MAIN_CKPT_DIR/checkpoint"
  exit 1
fi

echo "========== 轻量级领域微调：主模型 -> M5（100% 训练集，仅重编程层+预测头更新）=========="
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path m5_10ch.csv \
  --model_id Iron_96_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --factor 3 \
  --enc_in 10 \
  --dec_in 10 \
  --c_out 10 \
  --des 'Iron_Ore_Transport_Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --loss JointMaskedAuxMAE \
  --use_aux_loss \
  --aux_loss_weight 0.1 \
  --percent $percent \
  --load_ckpt_dir "$MAIN_CKPT_DIR"

echo ""
echo "轻量级领域微调完成。评估: bash scripts/Eval_M5_lightweight.sh"

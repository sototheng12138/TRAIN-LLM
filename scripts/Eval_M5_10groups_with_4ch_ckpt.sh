#!/bin/bash
# 用同一 4 通道主模型 checkpoint 得到 10 组泛化结果：分 3 次 4 通道 eval（子集 0-3、4-7、8-9），再合并。
# 说明：模型输出维度固定为 4，不能一次前向得到 10 维，所以用「分片 eval + 合并」在不改结构、不重训的前提下覆盖 10 组。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
CKPT_DIR="$ROOT/checkpoints/long_term_forecast_Iron_96_48_TimeLLM_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Iron_Ore_Transport_Exp_0-iron_stage2_linear"

if [ ! -f "$ROOT/dataset/m5_4ch_0_3.csv" ]; then
  echo "生成 4ch 子集 (0_3, 4_7, 8_9) ..."
  python "$SCRIPT_DIR/convert_m5_for_iron.py"
fi

run_one() {
  local data_path=$1
  local save_as=$2
  echo "========== Eval: $data_path =========="
  python run_eval.py \
    --device cpu \
    --task_name long_term_forecast \
    --root_path ./dataset/ \
    --data_path "$data_path" \
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
    --model_comment iron_stage2_linear \
    --llm_layers 32 \
    --itr 1 \
    --output_aux_semantic \
    --aux_confidence_threshold 0.5 \
    --zero_threshold -1
  [ -f "$CKPT_DIR/eval_result.txt" ] && cp "$CKPT_DIR/eval_result.txt" "$CKPT_DIR/$save_as"
}

run_one "m5_4ch_0_3.csv" "eval_m5_0_3.txt"
run_one "m5_4ch_4_7.csv" "eval_m5_4_7.txt"
run_one "m5_4ch_8_9.csv" "eval_m5_8_9.txt"

echo ""
python "$SCRIPT_DIR/merge_eval_10groups.py"
echo ""
echo "10 组结果已合并到: $CKPT_DIR/eval_result_m5_10groups_4ch_ckpt.txt"

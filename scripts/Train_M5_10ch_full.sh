#!/bin/bash
# M5 十通道从零训练一键脚本：Stage 1 → Stage 2，在 10 组数据上训练，得到可评估 10 组泛化结果的 checkpoint。
# 需先有 m5_10ch.csv（运行 convert_m5_for_iron.py 或 Eval_M5_generalization.sh 时会生成）。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
if [ ! -f "$ROOT/dataset/m5_10ch.csv" ]; then
  echo "生成 m5_4ch.csv / m5_10ch.csv ..."
  python "$SCRIPT_DIR/convert_m5_for_iron.py"
fi

echo "========== Stage 1: M5 10ch =========="
bash "$SCRIPT_DIR/Train_M5_10ch_stage1.sh"
echo ""
echo "========== Stage 2: M5 10ch =========="
bash "$SCRIPT_DIR/Train_M5_10ch_stage2.sh"
echo ""
echo "完成。10 组泛化评估: bash scripts/Eval_M5_10ch.sh"

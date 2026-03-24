#!/bin/bash
# 两阶段训练（多元 + 线性头）一键脚本：先 Stage 1 再 Stage 2，用于与主实验 iron_stage2_linear 对比。
# 等价于依次运行 Iron_stage1_multivariate_linear.sh、Iron_stage2_multivariate_linear.sh。
# 主实验对应：Iron_stage1_linear.sh → Iron_stage2_linear.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}

echo "=============================================="
echo "  Multivariate 两阶段训练（线性头）"
echo "  Stage 1 → Stage 2"
echo "=============================================="
echo ""

echo "========== Stage 1: iron_stage1_multivariate_linear =========="
bash "$SCRIPT_DIR/Iron_stage1_multivariate_linear.sh"
echo ""

echo "========== Stage 2: iron_stage2_multivariate_linear =========="
bash "$SCRIPT_DIR/Iron_stage2_multivariate_linear.sh"
echo ""

echo "=============================================="
echo "  两阶段训练完成"
echo "  评估: bash scripts/Eval_Iron_stage2_linear_vs_multivariate.sh"
echo "=============================================="

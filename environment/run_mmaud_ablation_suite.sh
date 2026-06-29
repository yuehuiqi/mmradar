#!/usr/bin/env bash
# =============================================================================
# Script 1: MMAUD PFA-NET 改进系列消融实验 (一键运行)
# 消融设计:
#   A1  PFA (baseline)          -- 已完成，复用 mmaud_all14_v1 结果
#   A2  DG-PFA w/o Graph        -- 去除动态图，保留辅助注意力
#   A3  DG-PFA w/o AuxAttn      -- 去除辅助注意力，保留动态图
#   A4  DG-PFA (full)           -- 完整模型
#
# 用法:
#   chmod +x run_mmaud_ablation_suite.sh
#   nohup bash run_mmaud_ablation_suite.sh > /mnt/e/Scholar/logs/mmaud_ablation_suite.log 2>&1 &
#   disown
# =============================================================================

set -euo pipefail

# ─── 硬件配置（根据显存和CPU核心调整）────────────────────────────────────────
BATCH_SIZE=4
WORKERS=2
EPOCHS=80
EVAL_INTERVAL=5

# ─── 路径 ─────────────────────────────────────────────────────────────────────
PFANET_DIR="/mnt/e/Scholar/mmradarDetect/PFA-NET"
PYTHON="/home/yuehui/miniforge3/envs/PFANet/bin/python"
TOOLS="$PFANET_DIR/tools"
LOG_DIR="/mnt/e/Scholar/logs/mmaud_ablation"
SUMMARY_SCRIPT="/mnt/e/Scholar/mmradarDetect/environment/scripts/generate_ablation_summary.py"
OUT_MD="/mnt/e/Scholar/mmradarDetect/environment/ablation_summary_mmaud.md"
EXTRA_TAG="mmaud_abl_v1"

mkdir -p "$LOG_DIR"
cd "$PFANET_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ─── 辅助：找到最优 checkpoint ────────────────────────────────────────────────
find_best_ckpt() {
    local model_dir="$1"
    local tag="$2"
    local ckpt_dir="output/mmradar_models/$model_dir/$tag/ckpt"
    if [ ! -d "$ckpt_dir" ]; then
        echo ""
        return
    fi
    # 取数值最大的 epoch checkpoint（假设最后一个是最高轮次）
    local best
    best=$(ls "$ckpt_dir"/checkpoint_epoch_*.pth 2>/dev/null | sort -V | tail -1)
    echo "$best"
}

# ─── 辅助：用 test.py 评估并生成 result.pkl ───────────────────────────────────
run_eval() {
    local cfg="$1"
    local ckpt="$2"
    local logfile="$3"
    log "  Evaluating $cfg with ckpt $(basename $ckpt)"
    $PYTHON tools/test.py \
        --cfg_file "$cfg" \
        --ckpt "$ckpt" \
        --batch_size $BATCH_SIZE \
        --workers $WORKERS \
        >> "$logfile" 2>&1
}

# ─── 辅助：训练单个模型 ───────────────────────────────────────────────────────
run_train() {
    local model_name="$1"
    local cfg_rel="$2"
    local logfile="$LOG_DIR/${model_name}.log"
    log "─────────────────────────────────────────"
    log "Training: $model_name (tag=$EXTRA_TAG)"
    log "Log: $logfile"
    $PYTHON tools/train.py \
        --cfg_file "$cfg_rel" \
        --extra_tag "$EXTRA_TAG" \
        --batch_size $BATCH_SIZE \
        --workers $WORKERS \
        --epochs $EPOCHS \
        --eval_interval $EVAL_INTERVAL \
        --fix_random_seed \
        >> "$logfile" 2>&1
    log "Training done: $model_name"
}

# =============================================================================
log "===== MMAUD DG-PFA 消融实验启动 ====="
log "BATCH_SIZE=$BATCH_SIZE  WORKERS=$WORKERS  EPOCHS=$EPOCHS"
log "Extra tag: $EXTRA_TAG"
log ""

# ─── A1: PFA baseline（复用已有结果，只做 eval 补全）────────────────────────
log "[A1] PFA baseline — 使用已有 mmaud_all14_v1 结果，跳过训练"
PFA_CKPT=$(find_best_ckpt "pfanet_mmaud_full" "mmaud_all14_v1")
if [ -n "$PFA_CKPT" ]; then
    log "[A1] Found ckpt: $PFA_CKPT"
    # 检查 result.pkl 是否已存在
    PFA_PKL=$(ls output/mmradar_models/pfanet_mmaud_full/mmaud_all14_v1/periodic_eval/*/result.pkl 2>/dev/null | sort -V | tail -1 || true)
    if [ -z "$PFA_PKL" ]; then
        log "[A1] result.pkl 不存在，运行 test.py 生成"
        run_eval "tools/cfgs/mmradar_models/pfanet_mmaud_full.yaml" "$PFA_CKPT" "$LOG_DIR/pfanet_mmaud_eval.log"
    else
        log "[A1] result.pkl 已存在: $PFA_PKL"
    fi
else
    log "[A1] 警告: 未找到 PFA mmaud_all14_v1 checkpoint，请先确认 mmaud_all14_v1 训练已完成"
fi

# ─── A2: DG-PFA w/o Graph ─────────────────────────────────────────────────────
log ""
log "[A2] DG-PFA w/o Graph (NUM_LAYERS=0, AUX_ATTENTION=True)"
run_train "dg_pfanet_mmaud_ablation_nograph" \
    "tools/cfgs/mmradar_models/dg_pfanet_mmaud_ablation_nograph.yaml"

# ─── A3: DG-PFA w/o AuxAttn ───────────────────────────────────────────────────
log ""
log "[A3] DG-PFA w/o AuxAttn (NUM_LAYERS=2, AUX_ATTENTION=False)"
run_train "dg_pfanet_mmaud_ablation_noaux" \
    "tools/cfgs/mmradar_models/dg_pfanet_mmaud_ablation_noaux.yaml"

# ─── A4: DG-PFA full ──────────────────────────────────────────────────────────
log ""
log "[A4] DG-PFA full (NUM_LAYERS=2, AUX_ATTENTION=True)"
run_train "dg_pfanet_mmaud_full" \
    "tools/cfgs/mmradar_models/dg_pfanet_mmaud_full.yaml"

# ─── 汇总 ──────────────────────────────────────────────────────────────────────
log ""
log "===== 生成汇总 MD ====="
$PYTHON "$SUMMARY_SCRIPT" \
    --output_root "$PFANET_DIR/output/mmradar_models" \
    --dataset MMAUD \
    --out_md "$OUT_MD"

log "===== 消融实验全部完成 ====="
log "汇总文件: $OUT_MD"

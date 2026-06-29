#!/usr/bin/env bash
# =============================================================================
# Script 2: AIQII PFA-NET 改进系列消融实验 (一键运行)
# 消融设计:
#   A1  PFA (baseline)          -- AIQII 从头训练
#   A2  DG-PFA w/o Graph        -- 去除动态图，保留辅助注意力
#   A3  DG-PFA w/o AuxAttn      -- 去除辅助注意力，保留动态图
#   A4  DG-PFA (full)           -- 完整模型
#
# 用法:
#   chmod +x run_aiqii_ablation_suite.sh
#   nohup bash run_aiqii_ablation_suite.sh > /mnt/e/Scholar/logs/aiqii_ablation_suite.log 2>&1 &
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
LOG_DIR="/mnt/e/Scholar/logs/aiqii_ablation"
SUMMARY_SCRIPT="/mnt/e/Scholar/mmradarDetect/environment/scripts/generate_ablation_summary.py"
OUT_MD="/mnt/e/Scholar/mmradarDetect/environment/ablation_summary_aiqii.md"
EXTRA_TAG="aiqii_abl_v1"

mkdir -p "$LOG_DIR"
cd "$PFANET_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

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
log "===== AIQII DG-PFA 消融实验启动 ====="
log "BATCH_SIZE=$BATCH_SIZE  WORKERS=$WORKERS  EPOCHS=$EPOCHS"
log "Extra tag: $EXTRA_TAG"
log ""

# ─── A1: PFA baseline ─────────────────────────────────────────────────────────
log "[A1] PFA baseline (AIQII 从头训练)"
run_train "pfanet_aiqii_full" \
    "tools/cfgs/mmradar_models/pfanet_aiqii_full.yaml"

# ─── A2: DG-PFA w/o Graph ─────────────────────────────────────────────────────
log ""
log "[A2] DG-PFA w/o Graph"
run_train "dg_pfanet_aiqii_ablation_nograph" \
    "tools/cfgs/mmradar_models/dg_pfanet_aiqii_ablation_nograph.yaml"

# ─── A3: DG-PFA w/o AuxAttn ───────────────────────────────────────────────────
log ""
log "[A3] DG-PFA w/o AuxAttn"
run_train "dg_pfanet_aiqii_ablation_noaux" \
    "tools/cfgs/mmradar_models/dg_pfanet_aiqii_ablation_noaux.yaml"

# ─── A4: DG-PFA full ──────────────────────────────────────────────────────────
log ""
log "[A4] DG-PFA full"
run_train "dg_pfanet_aiqii_full" \
    "tools/cfgs/mmradar_models/dg_pfanet_aiqii_full.yaml"

# ─── 汇总 ──────────────────────────────────────────────────────────────────────
log ""
log "===== 生成汇总 MD ====="
$PYTHON "$SUMMARY_SCRIPT" \
    --output_root "$PFANET_DIR/output/mmradar_models" \
    --dataset AIQII \
    --out_md "$OUT_MD"

log "===== AIQII 消融实验全部完成 ====="
log "汇总文件: $OUT_MD"

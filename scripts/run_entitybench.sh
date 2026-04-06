#!/bin/bash

# 配置基础路径
CKPT_DIR="/root/autodl-tmp/models/Wan-Move-14B-480P"
DATA_ROOT="./data/EntityBench"
SAVE_ROOT="./results/entitybench"
MAX_RETRIES=3

mkdir -p $SAVE_ROOT

# 1. 显存管理优化
# max_split_size_mb:128 有助于减少显存碎片
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" 
export MASTER_PORT=29501

# 遍历测试用例
for case_dir in $(find "$DATA_ROOT" -maxdepth 1 -type d -name "case_*" | sort); do
    case_name=$(basename "$case_dir")
    SAVE_FILE="$SAVE_ROOT/${case_name}_gen.mp4"

    # 跳过已生成的
    if [ -f "$SAVE_FILE" ] && [ -s "$SAVE_FILE" ]; then
        echo "⏭️  $case_name already exists, skipping..."
        continue
    fi

    prompt_content=$(cat "$case_dir/prompt.txt")
    echo "🚀 [Start] Processing $case_name with 4 GPUs..."

    attempt=1
    success=false

    while [ $attempt -le $MAX_RETRIES ]; do
        # 2. 关键调整：
        # --nproc_per_node 4: 调用全部 4 张显卡
        # --ulysses_size 4: 将序列并行度提升到 4，极大降低单卡显存峰值
        # --offload_model True: 依然保留，确保万无一失
        torchrun --nproc_per_node=4 --master_port=$MASTER_PORT generate.py \
          --task wan-move-i2v \
          --size 480*832 \
          --ckpt_dir "$CKPT_DIR" \
          --ulysses_size 4 \
          --dit_fsdp \
          --t5_cpu \
          --dtype bf16 \
          --sample_steps 40 \
          --offload_model True \
          --image "$case_dir/image.jpg" \
          --track "$case_dir/tracks.npy" \
          --track_visibility "$case_dir/visibility.npy" \
          --prompt "$prompt_content" \
          --save_file "$SAVE_FILE"

        if [ $? -eq 0 ]; then
            echo "✅ $case_name finished successfully."
            success=true
            break
        else
            echo "❌ OOM or Error at attempt $attempt. Retrying..."
            rm -f "$SAVE_FILE"
            # 彻底清理残留进程，释放 4 张卡的显存
            pkill -9 python
            sleep 5 
            ((attempt++))
        fi
    done
done
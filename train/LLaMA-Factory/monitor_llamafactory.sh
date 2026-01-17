#!/bin/bash

echo "=== 🔍 LLaMA-Factory 训练监控 ==="
echo ""

OUTPUT_DIR="train_output/test_edit_thinker_qwen2_5vl_sft_lr_-4"

while true; do
    clear
    echo "=== 📊 训练状态监控 ($(date '+%Y-%m-%d %H:%M:%S')) ==="
    echo ""
    
    # 检查进程
    PROC_COUNT=$(ps aux | grep "llamafactory-cli" | grep -v grep | wc -l)
    if [ $PROC_COUNT -gt 0 ]; then
        echo "✅ 训练进程运行中"
        ps aux | grep "llamafactory-cli" | grep -v grep | awk '{print "   PID: "$2", CPU: "$3"%, MEM: "$4"%, RSS: "$6" KB"}'
    else
        echo "❌ 训练进程未运行"
    fi
    
    echo ""
    echo "=== 📁 输出目录状态 ==="
    if [ -d "$OUTPUT_DIR" ]; then
        echo "✅ 输出目录已创建"
        ls -lth "$OUTPUT_DIR" | head -10
        
        echo ""
        echo "=== 📝 最新训练日志 (最后10行) ==="
        if [ -f "$OUTPUT_DIR/trainer_log.jsonl" ]; then
            tail -10 "$OUTPUT_DIR/trainer_log.jsonl" | jq -r '. | "Step: \(.current_steps), Loss: \(.loss), LR: \(.learning_rate)"' 2>/dev/null || tail -10 "$OUTPUT_DIR/trainer_log.jsonl"
        else
            echo "等待日志文件生成..."
        fi
        
        echo ""
        echo "=== 💾 Checkpoint 信息 ==="
        ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | tail -5 || echo "暂无checkpoint"
    else
        echo "⏳ 等待输出目录创建（模型加载中）..."
    fi
    
    echo ""
    echo "=== 🎮 GPU 状态 ==="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -8
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "按 Ctrl+C 退出监控"
    echo ""
    
    sleep 10
done

#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

# =============================================================================
# ContraBERT 漏洞检测深度优化训练脚本 V3
# 整合SOTA技术，专门针对代码漏洞检测任务优化
# =============================================================================

echo "🚀 ContraBERT Enhanced Training V3"
echo "🎯 深度优化版本，集成多种SOTA技术"
echo "============================================"

# 基础配置
Pretrain_dir="../saved_models/pretrain_models/"
Model_type="ContraBERT_C"  # ContraBERT_G
DATA_DIR="./dataset"
OUTPUT_DIR="./results_enhanced_v3"
timestamp=$(date +%Y%m%d_%H%M%S)

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/conservative
mkdir -p $OUTPUT_DIR/aggressive
mkdir -p $OUTPUT_DIR/lightweight
mkdir -p $OUTPUT_DIR/baseline
mkdir -p $OUTPUT_DIR/ensemble

echo "📁 输出目录: $OUTPUT_DIR"
echo "🤖 模型类型: $Model_type"
echo "⏰ 开始时间: $(date)"

# =============================================================================
# 策略1: 保守优化 (推荐首选)
# 使用经过验证的优化技术，稳定提升性能
# =============================================================================

echo ""
echo "🔵 策略1: 保守优化策略"
echo "- Focal Loss处理类别不平衡"
echo "- 多头注意力池化"
echo "- 加权采样"
echo "- 早停机制"
echo "- 梯度裁剪优化"

python vulnerability_detection_v3.py \
    --output_dir="$OUTPUT_DIR/conservative" \
    --model_type=roberta \
    --model_name_or_path=${Pretrain_dir}/$Model_type \
    --tokenizer_name=microsoft/codebert-base \
    --train_data_file="$DATA_DIR/train.jsonl" \
    --eval_data_file="$DATA_DIR/valid.jsonl" \
    --test_data_file="$DATA_DIR/test.jsonl" \
    --epoch=6 \
    --block_size=512 \
    --train_batch_size=24 \
    --eval_batch_size=48 \
    --learning_rate=1e-5 \
    --max_grad_norm=0.5 \
    --weight_decay=0.01 \
    --evaluate_during_training \
    --pooling_strategy=attention \
    --dropout_rate=0.2 \
    --use_focal_loss \
    --focal_alpha=2.0 \
    --focal_gamma=2.0 \
    --use_weighted_sampling \
    --use_early_stopping \
    --patience=3 \
    --seed=42 \
    --do_train \
    --do_eval \
    --do_test 2>&1| tee $OUTPUT_DIR/conservative/train.log

# =============================================================================
# 策略2: 激进优化 (追求最高性能)
# 使用更多高级技术，可能提供更大提升但需要更多调优
# =============================================================================

echo ""
echo "🔴 策略2: 激进优化策略"
echo "- Focal Loss + Label Smoothing"
echo "- 更深的分类器网络"
echo "- 差异化学习率"
echo "- 多项式学习率衰减"

python vulnerability_detection_v3.py \
    --output_dir="$OUTPUT_DIR/aggressive" \
    --model_type=roberta \
    --model_name_or_path=${Pretrain_dir}/$Model_type \
    --tokenizer_name=microsoft/codebert-base \
    --train_data_file="$DATA_DIR/train.jsonl" \
    --eval_data_file="$DATA_DIR/valid.jsonl" \
    --test_data_file="$DATA_DIR/test.jsonl" \
    --epoch=8 \
    --block_size=512 \
    --train_batch_size=16 \
    --eval_batch_size=32 \
    --learning_rate=2e-5 \
    --max_grad_norm=1.0 \
    --weight_decay=0.02 \
    --evaluate_during_training \
    --pooling_strategy=attention \
    --dropout_rate=0.3 \
    --use_focal_loss \
    --focal_alpha=2.5 \
    --focal_gamma=2.0 \
    --use_label_smoothing \
    --label_smoothing=0.1 \
    --use_weighted_sampling \
    --use_early_stopping \
    --patience=4 \
    --seed=123 \
    --do_train \
    --do_eval \
    --do_test 2>&1| tee $OUTPUT_DIR/aggressive/train.log

# =============================================================================
# 策略3: 轻量级优化 (快速训练)
# 针对计算资源有限的情况，平衡性能和效率
# =============================================================================

echo ""
echo "🟡 策略3: 轻量级优化策略"
echo "- CLS token池化(最快)"
echo "- 较小批次大小"
echo "- 更少训练轮数"
echo "- 基础优化技术"

python vulnerability_detection_v3.py \
    --output_dir="$OUTPUT_DIR/lightweight" \
    --model_type=roberta \
    --model_name_or_path=${Pretrain_dir}/$Model_type \
    --tokenizer_name=microsoft/codebert-base \
    --train_data_file="$DATA_DIR/train.jsonl" \
    --eval_data_file="$DATA_DIR/valid.jsonl" \
    --test_data_file="$DATA_DIR/test.jsonl" \
    --epoch=4 \
    --block_size=400 \
    --train_batch_size=32 \
    --eval_batch_size=64 \
    --learning_rate=2e-5 \
    --max_grad_norm=1.0 \
    --weight_decay=0.01 \
    --evaluate_during_training \
    --pooling_strategy=cls \
    --dropout_rate=0.1 \
    --use_weighted_sampling \
    --use_early_stopping \
    --patience=2 \
    --seed=456 \
    --do_train \
    --do_eval \
    --do_test 2>&1| tee $OUTPUT_DIR/lightweight/train.log

# =============================================================================
# 策略4: 原始基线对比 (验证改进效果)
# 使用原始代码进行训练，作为性能基准
# =============================================================================

echo ""
echo "⚫ 策略4: 原始基线对比"
echo "- 使用原始训练代码"
echo "- 相同训练参数"
echo "- 用于验证改进效果"

python vulnerability_detection.py \
    --output_dir="$OUTPUT_DIR/baseline" \
    --model_type=roberta \
    --model_name_or_path=${Pretrain_dir}/$Model_type \
    --tokenizer_name=microsoft/codebert-base \
    --train_data_file="$DATA_DIR/train.jsonl" \
    --eval_data_file="$DATA_DIR/valid.jsonl" \
    --test_data_file="$DATA_DIR/test.jsonl" \
    --epoch=6 \
    --block_size=512 \
    --train_batch_size=24 \
    --eval_batch_size=48 \
    --learning_rate=1e-5 \
    --max_grad_norm=0.5 \
    --weight_decay=0.01 \
    --evaluate_during_training \
    --seed=42 \
    --do_train \
    --do_eval \
    --do_test 2>&1| tee $OUTPUT_DIR/baseline/train.log

# =============================================================================
# 集成学习策略 (可选)
# 训练多个模型进行集成预测
# =============================================================================

echo ""
echo "🟢 策略5: 集成学习 (可选)"
echo "- 训练3个不同配置的模型"
echo "- 使用投票或平均进行预测"

for i in {1..3}; do
    seed=$((42 + i * 111))
    lr_multiplier=$(echo "scale=1; 1.0 + $i * 0.2" | bc)
    lr=$(echo "scale=6; 1e-5 * $lr_multiplier" | bc)
    
    echo "训练集成模型 $i/3, seed=$seed, lr=$lr"
    
    python vulnerability_detection_v3.py \
        --output_dir="$OUTPUT_DIR/ensemble/model_$i" \
        --model_type=roberta \
        --model_name_or_path=${Pretrain_dir}/$Model_type \
        --tokenizer_name=microsoft/codebert-base \
        --train_data_file="$DATA_DIR/train.jsonl" \
        --eval_data_file="$DATA_DIR/valid.jsonl" \
        --test_data_file="$DATA_DIR/test.jsonl" \
        --epoch=6 \
        --block_size=512 \
        --train_batch_size=24 \
        --eval_batch_size=48 \
        --learning_rate=$lr \
        --max_grad_norm=0.5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --pooling_strategy=attention \
        --dropout_rate=$((20 + i * 5))e-2 \
        --use_focal_loss \
        --focal_alpha=2.0 \
        --focal_gamma=2.0 \
        --use_weighted_sampling \
        --use_early_stopping \
        --patience=3 \
        --seed=$seed \
        --do_train \
        --do_eval \
        --do_test 2>&1| tee $OUTPUT_DIR/ensemble/model_$i/train.log
done

# =============================================================================
# 结果汇总和分析
# =============================================================================

echo ""
echo "📊 训练完成，正在生成结果分析..."

# 创建结果分析脚本
cat > $OUTPUT_DIR/analyze_results.py << 'EOF'
#!/usr/bin/env python3
"""
结果分析脚本
分析不同策略的性能表现
"""

import json
import os
import glob
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd

def load_metrics(result_dir):
    """加载测试结果"""
    metrics_file = os.path.join(result_dir, 'test_metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def analyze_results(base_dir):
    """分析所有策略的结果"""
    strategies = ['conservative', 'aggressive', 'lightweight', 'baseline']
    results = []
    
    for strategy in strategies:
        strategy_dir = os.path.join(base_dir, strategy)
        if os.path.exists(strategy_dir):
            metrics = load_metrics(strategy_dir)
            if metrics:
                results.append({
                    'Strategy': strategy,
                    'Accuracy': metrics.get('Accuracy', 0),
                    'F1_Score': metrics.get('F1_Score', 0),
                    'Precision': metrics.get('Precision', 0),
                    'Recall': metrics.get('Recall', 0),
                    'PRC_AUC': metrics.get('PRC_AUC', 0)
                })
    
    if not results:
        print("❌ 没有找到有效的测试结果")
        return
        
    # 创建结果表格
    df = pd.DataFrame(results)
    print("📈 各策略性能对比:")
    print("=" * 80)
    print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # 找到最佳策略
    best_strategy = df.loc[df['Accuracy'].idxmax()]
    print(f"\n🏆 最佳策略: {best_strategy['Strategy']}")
    print(f"   准确率: {best_strategy['Accuracy']:.4f}")
    print(f"   F1分数: {best_strategy['F1_Score']:.4f}")
    
    # 与baseline对比
    baseline_metrics = df[df['Strategy'] == 'baseline']
    if not baseline_metrics.empty:
        baseline_acc = baseline_metrics['Accuracy'].iloc[0]
        baseline_f1 = baseline_metrics['F1_Score'].iloc[0]
        
        print(f"\n📊 相比Baseline的改进:")
        for _, row in df.iterrows():
            if row['Strategy'] != 'baseline':
                acc_improvement = (row['Accuracy'] - baseline_acc) * 100
                f1_improvement = (row['F1_Score'] - baseline_f1) * 100
                print(f"   {row['Strategy']:12s}: 准确率 {acc_improvement:+.2f}%, F1 {f1_improvement:+.2f}%")
    
    # 保存结果
    df.to_csv(os.path.join(base_dir, 'comparison_results.csv'), index=False)
    print(f"\n💾 结果已保存到: {os.path.join(base_dir, 'comparison_results.csv')}")

if __name__ == "__main__":
    import sys
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    analyze_results(base_dir)
EOF

# 运行分析脚本
echo "🔍 分析结果中..."
python $OUTPUT_DIR/analyze_results.py $OUTPUT_DIR

# 创建集成预测脚本（如果训练了集成模型）
if [ -d "$OUTPUT_DIR/ensemble" ]; then
    cat > $OUTPUT_DIR/ensemble_predict.py << 'EOF'
#!/usr/bin/env python3
"""
集成模型预测脚本
"""

import torch
import numpy as np
import json
import glob
import os
from torch.utils.data import DataLoader, SequentialSampler
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

# 需要导入自定义模块
import sys
sys.path.append('.')
from vulnerability_detection_v3 import TextDataset, EnhancedVulModel, evaluate_predictions_enhanced

class Args:
    def __init__(self):
        self.block_size = 512
        self.pooling_strategy = 'attention'
        self.dropout_rate = 0.2
        self.use_focal_loss = True
        self.focal_alpha = 2.0
        self.focal_gamma = 2.0
        self.use_label_smoothing = False

def ensemble_predict(ensemble_dir, test_file, pretrain_model, tokenizer_name="microsoft/codebert-base"):
    """集成模型预测"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    
    # 加载测试数据
    args = Args()
    test_dataset = TextDataset(tokenizer, args, test_file)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=32
    )
    
    # 找到所有模型
    model_dirs = glob.glob(os.path.join(ensemble_dir, "model_*"))
    all_predictions = []
    
    for model_dir in model_dirs:
        print(f"加载模型: {model_dir}")
        
        # 加载模型
        config = RobertaConfig.from_pretrained(pretrain_model)
        config.num_labels = 1
        
        encoder = RobertaForSequenceClassification.from_pretrained(pretrain_model, config=config)
        model = EnhancedVulModel(encoder, config, tokenizer, args)
        
        # 加载最佳权重
        model_file = os.path.join(model_dir, 'checkpoint-best-acc', 'model.bin')
        if os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.to(device)
            model.eval()
            
            predictions = []
            with torch.no_grad():
                for batch in test_dataloader:
                    inputs = batch[0].to(device)
                    probs = model(inputs)
                    predictions.append(probs.cpu().numpy())
            
            all_predictions.append(np.concatenate(predictions, axis=0))
    
    if not all_predictions:
        print("❌ 没有找到有效的模型文件")
        return
    
    # 集成预测 - 平均概率
    ensemble_probs = np.mean(all_predictions, axis=0)
    
    # 评估
    labels = [example.label for example in test_dataset.examples]
    metrics = evaluate_predictions_enhanced(np.array(labels), ensemble_probs)
    
    print("\n🎯 集成模型测试结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # 保存结果
    with open(os.path.join(ensemble_dir, 'ensemble_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    ensemble_dir = sys.argv[1] if len(sys.argv) > 1 else "./ensemble"
    test_file = sys.argv[2] if len(sys.argv) > 2 else "./dataset/test.jsonl"
    pretrain_model = sys.argv[3] if len(sys.argv) > 3 else "../saved_models/pretrain_models/ContraBERT_C"
    
    ensemble_predict(ensemble_dir, test_file, pretrain_model)
EOF
    
    echo "🎯 运行集成模型预测..."
    python $OUTPUT_DIR/ensemble_predict.py $OUTPUT_DIR/ensemble $DATA_DIR/test.jsonl ${Pretrain_dir}/$Model_type
fi

echo ""
echo "🎉 =============================================="
echo "✅ 训练脚本执行完成!"
echo "⏰ 结束时间: $(date)"
echo "📁 结果目录: $OUTPUT_DIR"
echo "📊 查看结果: cat $OUTPUT_DIR/comparison_results.csv"
echo "==============================================="

echo ""
echo "📋 主要优化技术说明:"
echo "1. 🎯 Focal Loss: 解决类别不平衡问题"
echo "2. 🧠 多头注意力池化: 更好的序列表示"
echo "3. ⚖️ 加权采样: 平衡训练数据"
echo "4. 🛑 早停机制: 防止过拟合"
echo "5. 📐 梯度裁剪: 稳定训练过程"
echo "6. 🎛️ 标签平滑: 提升泛化能力"
echo "7. 🔄 多项式学习率衰减: 更好的收敛"
echo "8. 🏗️ 多层分类器: 增强分类能力"
echo "9. 🎲 集成学习: 多模型投票提升性能"
echo "10. 📈 增强评估指标: 更全面的性能分析" 
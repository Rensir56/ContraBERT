#!/bin/bash

# ContraBERT 漏洞检测优化训练脚本
# 使用多种先进技术提升模型性能

set -e  # 遇到错误时停止执行

export HF_ENDPOINT=https://hf-mirror.com

# =============================================================================
# 配置参数
# =============================================================================

# 基础路径配置
PRETRAIN_DIR="../saved_models/pretrain_models/"
MODEL_TYPE="ContraBERT_C"  # 或 ContraBERT_G
DATA_DIR="./dataset"
OUTPUT_BASE="../saved_models/finetune_models/vulnerability_detection_enhanced"

# 创建输出目录
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_TYPE}_enhanced_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${OUTPUT_DIR}

echo "🚀 开始ContraBERT增强训练"
echo "📁 输出目录: ${OUTPUT_DIR}"
echo "🤖 模型类型: ${MODEL_TYPE}"

# =============================================================================
# 策略1: 基础增强训练 (推荐用于大多数情况)
# =============================================================================

echo "📊 策略1: 基础增强训练"

python vulnerability_detection_enhanced.py \
    --train_data_file=${DATA_DIR}/train.jsonl \
    --eval_data_file=${DATA_DIR}/valid.jsonl \
    --test_data_file=${DATA_DIR}/test.jsonl \
    --output_dir=${OUTPUT_DIR}/basic_enhanced \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=${PRETRAIN_DIR}/${MODEL_TYPE} \
    --num_train_epochs=6 \
    --block_size=512 \
    --train_batch_size=32 \
    --eval_batch_size=64 \
    --learning_rate=1e-5 \
    --max_grad_norm=1.0 \
    --seed=42 \
    --dropout_rate=0.3 \
    --pooling_strategy=attention \
    --use_focal_loss \
    --focal_alpha=2.0 \
    --focal_gamma=2.0 \
    --use_early_stopping \
    --patience=3 \
    --use_gradual_unfreezing \
    --use_weighted_sampling \
    2>&1 | tee ${OUTPUT_DIR}/basic_enhanced/train.log

# =============================================================================
# 策略2: 数据增强 + 对比学习 (追求最佳性能)
# =============================================================================

echo "🔥 策略2: 数据增强 + 对比学习"

python vulnerability_detection_enhanced.py \
    --train_data_file=${DATA_DIR}/train.jsonl \
    --eval_data_file=${DATA_DIR}/valid.jsonl \
    --test_data_file=${DATA_DIR}/test.jsonl \
    --output_dir=${OUTPUT_DIR}/contrastive_augmented \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=${PRETRAIN_DIR}/${MODEL_TYPE} \
    --num_train_epochs=6 \
    --block_size=512 \
    --train_batch_size=32 \
    --eval_batch_size=64 \
    --learning_rate=8e-6 \
    --max_grad_norm=1.0 \
    --seed=42 \
    --use_data_augmentation \
    --augmentation_ratio=0.3 \
    --dropout_rate=0.2 \
    --pooling_strategy=attention \
    --use_focal_loss \
    --focal_alpha=3.0 \
    --focal_gamma=2.5 \
    --label_smoothing=0.1 \
    --use_contrastive \
    --contrastive_temperature=0.07 \
    --contrastive_weight=0.2 \
    --use_early_stopping \
    --patience=4 \
    --use_gradual_unfreezing \
    --use_weighted_sampling \
    2>&1 | tee ${OUTPUT_DIR}/contrastive_augmented/train.log

# =============================================================================
# 策略3: 轻量化快速训练 (适合资源受限环境)
# =============================================================================

echo "⚡ 策略3: 轻量化快速训练"

python vulnerability_detection_enhanced.py \
    --train_data_file=${DATA_DIR}/train.jsonl \
    --eval_data_file=${DATA_DIR}/valid.jsonl \
    --test_data_file=${DATA_DIR}/test.jsonl \
    --output_dir=${OUTPUT_DIR}/lightweight \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=${PRETRAIN_DIR}/${MODEL_TYPE} \
    --num_train_epochs=4 \
    --block_size=256 \
    --train_batch_size=48 \
    --eval_batch_size=96 \
    --learning_rate=2e-5 \
    --max_grad_norm=1.0 \
    --seed=42 \
    --dropout_rate=0.1 \
    --pooling_strategy=cls \
    --use_early_stopping \
    --patience=2 \
    --use_weighted_sampling \
    2>&1 | tee ${OUTPUT_DIR}/lightweight/train.log

# =============================================================================
# 策略4: 集成学习 (训练多个模型进行投票)
# =============================================================================

echo "🎯 策略4: 集成学习"

# 训练3个不同配置的模型
for i in {1..3}; do
    seed=$((42 + i * 100))
    lr=$(python -c "print(f'{1e-5 + i * 5e-6:.2e}')")
    
    echo "训练集成模型 ${i}/3, seed=${seed}, lr=${lr}"
    
    python vulnerability_detection_enhanced.py \
        --train_data_file=${DATA_DIR}/train.jsonl \
        --eval_data_file=${DATA_DIR}/valid.jsonl \
        --test_data_file=${DATA_DIR}/test.jsonl \
        --output_dir=${OUTPUT_DIR}/ensemble/model_${i} \
        --model_type=roberta \
        --tokenizer_name=microsoft/codebert-base \
        --model_name_or_path=${PRETRAIN_DIR}/${MODEL_TYPE} \
        --num_train_epochs=6 \
        --block_size=512 \
        --train_batch_size=32 \
        --eval_batch_size=64 \
        --learning_rate=${lr} \
        --max_grad_norm=1.0 \
        --seed=${seed} \
        --dropout_rate=0.2 \
        --pooling_strategy=attention \
        --use_focal_loss \
        --focal_alpha=2.0 \
        --focal_gamma=2.0 \
        --use_early_stopping \
        --patience=3 \
        --use_gradual_unfreezing \
        --use_weighted_sampling \
        2>&1 | tee ${OUTPUT_DIR}/ensemble/model_${i}/train.log
done

# =============================================================================
# 结果分析
# =============================================================================

echo "📊 训练完成，正在分析结果..."

# 创建结果汇总脚本
cat > ${OUTPUT_DIR}/analyze_results.py << 'EOF'
#!/usr/bin/env python3
import json
import os
import glob
from tabulate import tabulate

def analyze_results(base_dir):
    results = []
    
    # 查找所有test_results.json文件
    result_files = glob.glob(f"{base_dir}/**/test_results.json", recursive=True)
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
            
            strategy = os.path.dirname(file_path).split('/')[-1]
            
            results.append([
                strategy,
                f"{metrics['Acc']:.4f}",
                f"{metrics['Pos_f1']:.4f}",
                f"{metrics['PRC_AUC']:.4f}",
                f"{metrics['Pos_pre']:.4f}",
                f"{metrics['Pos_rec']:.4f}"
            ])
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    if results:
        headers = ["策略", "准确率", "F1分数", "AUC", "精确率", "召回率"]
        print("\n🏆 各策略性能对比:")
        print(tabulate(results, headers=headers, tablefmt="grid"))
        
        # 找出最佳策略
        best_acc = max(results, key=lambda x: float(x[1]))
        best_f1 = max(results, key=lambda x: float(x[2]))
        best_auc = max(results, key=lambda x: float(x[3]))
        
        print(f"\n🥇 最佳准确率: {best_acc[0]} ({best_acc[1]})")
        print(f"🥇 最佳F1分数: {best_f1[0]} ({best_f1[2]})")
        print(f"🥇 最佳AUC: {best_auc[0]} ({best_auc[3]})")
    else:
        print("未找到测试结果文件")

if __name__ == "__main__":
    import sys
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    analyze_results(base_dir)
EOF

# 运行结果分析
python ${OUTPUT_DIR}/analyze_results.py ${OUTPUT_DIR}

# =============================================================================
# 模型融合 (可选)
# =============================================================================

echo "🔄 模型融合预测"

cat > ${OUTPUT_DIR}/ensemble_predict.py << 'EOF'
#!/usr/bin/env python3
import torch
import numpy as np
import json
import glob
from torch.utils.data import DataLoader, SequentialSampler
from vulnerability_detection import TextDataset, evaluate_predictions
from vulmodel_enhanced import EnhancedVulModel
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

def ensemble_predict(model_dirs, test_file, tokenizer_name):
    # 加载tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    
    # 加载测试数据
    class Args:
        block_size = 512
    
    test_dataset = TextDataset(tokenizer, Args(), test_file)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=32
    )
    
    all_predictions = []
    
    for model_dir in model_dirs:
        print(f"加载模型: {model_dir}")
        
        # 加载模型
        config = RobertaConfig.from_pretrained("microsoft/codebert-base")
        config.num_labels = 1
        encoder = RobertaForSequenceClassification.from_pretrained(
            "microsoft/codebert-base", config=config
        )
        model = EnhancedVulModel(encoder, config, tokenizer, Args())
        
        # 查找最佳模型文件
        model_files = glob.glob(f"{model_dir}/best_model_*.bin")
        if model_files:
            model.load_state_dict(torch.load(model_files[0], map_location='cpu'))
            model.eval()
            
            predictions = []
            with torch.no_grad():
                for batch in test_dataloader:
                    inputs = batch[0]
                    probs = model(inputs)
                    predictions.append(probs.numpy())
            
            all_predictions.append(np.concatenate(predictions, axis=0))
    
    if all_predictions:
        # 平均预测
        ensemble_preds = np.mean(all_predictions, axis=0)
        
        # 评估
        test_labels = np.array([example.label for example in test_dataset.examples])
        metrics = evaluate_predictions(test_labels, ensemble_preds)
        
        print("🎯 集成模型结果:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics
    else:
        print("未找到有效的模型文件")
        return None

if __name__ == "__main__":
    model_dirs = glob.glob("ensemble/model_*")
    if model_dirs:
        ensemble_predict(model_dirs, "../../data/finetune_data/c_vulnerability/devign/test.jsonl", "microsoft/codebert-base")
    else:
        print("未找到集成模型目录")
EOF

# 如果有集成模型，运行融合预测
if [ -d "${OUTPUT_DIR}/ensemble" ]; then
    cd ${OUTPUT_DIR}
    python ensemble_predict.py
    cd -
fi

echo "✅ 所有训练策略执行完成！"
echo "📊 详细结果请查看: ${OUTPUT_DIR}"
echo ""
echo "💡 使用建议:"
echo "   - 策略1: 适合大多数情况，平衡性能和效率"
echo "   - 策略2: 追求最佳性能，需要更多计算资源"
echo "   - 策略3: 快速验证，适合原型开发"
echo "   - 策略4: 最高性能，通过集成多个模型" 
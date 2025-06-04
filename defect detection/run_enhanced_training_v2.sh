#!/bin/bash

# 增强版训练脚本 - 基于原始代码的最小化改进
# 保持原有训练逻辑，仅添加必要的优化

echo "===================="
echo "ContraBERT Enhanced Training V2"
echo "基于原始代码的最小化增强版本"
echo "===================="

# 基础配置
MODEL_PATH="~/ContraBERT/saved_models/pretrain_models/ContraBERT_hybrid/microsoft/graphcodebert-base/checkpoint-100000"
DATA_DIR="~/ContraBERT/datasets/finetune_datasets/vulnerability_detection_jsonl"
OUTPUT_DIR="./results_enhanced_v2"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "策略1: 基础增强 (保守优化)"
echo "- 加权采样处理类别不平衡"
echo "- 余弦学习率调度"
echo "- 早停机制"

python vulnerability_detection_v2.py \
    --output_dir="$OUTPUT_DIR/strategy1" \
    --model_type=roberta \
    --model_name_or_path="$MODEL_PATH" \
    --tokenizer_name="$MODEL_PATH" \
    --train_data_file="$DATA_DIR/train.jsonl" \
    --eval_data_file="$DATA_DIR/val.jsonl" \
    --test_data_file="$DATA_DIR/test.jsonl" \
    --block_size=512 \
    --train_batch_size=16 \
    --eval_batch_size=32 \
    --learning_rate=1e-5 \
    --num_train_epochs=6 \
    --epoch=6 \
    --weight_decay=0.01 \
    --evaluate_during_training \
    --use_weighted_sampling \
    --use_early_stopping \
    --patience=3 \
    --use_cosine_schedule \
    --do_train \
    --do_eval \
    --do_test

echo ""
echo "策略2: 层次化学习率"
echo "- 分类器层使用2倍学习率"
echo "- 其他所有优化保持一致"

python vulnerability_detection_v2.py \
    --output_dir="$OUTPUT_DIR/strategy2" \
    --model_type=roberta \
    --model_name_or_path="$MODEL_PATH" \
    --tokenizer_name="$MODEL_PATH" \
    --train_data_file="$DATA_DIR/train.jsonl" \
    --eval_data_file="$DATA_DIR/val.jsonl" \
    --test_data_file="$DATA_DIR/test.jsonl" \
    --block_size=512 \
    --train_batch_size=16 \
    --eval_batch_size=32 \
    --learning_rate=1e-5 \
    --num_train_epochs=6 \
    --epoch=6 \
    --weight_decay=0.01 \
    --evaluate_during_training \
    --use_weighted_sampling \
    --use_early_stopping \
    --patience=3 \
    --use_cosine_schedule \
    --use_layer_lr \
    --do_train \
    --do_eval \
    --do_test

echo ""
echo "策略3: 原始基线 (仅使用原始代码)"
echo "- 不使用任何增强功能"
echo "- 用于对比验证"

python vulnerability_detection_v2.py \
    --output_dir="$OUTPUT_DIR/baseline" \
    --model_type=roberta \
    --model_name_or_path="$MODEL_PATH" \
    --tokenizer_name="$MODEL_PATH" \
    --train_data_file="$DATA_DIR/train.jsonl" \
    --eval_data_file="$DATA_DIR/val.jsonl" \
    --test_data_file="$DATA_DIR/test.jsonl" \
    --block_size=512 \
    --train_batch_size=16 \
    --eval_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=5 \
    --epoch=5 \
    --weight_decay=0.0 \
    --evaluate_during_training \
    --do_train \
    --do_eval \
    --do_test

echo ""
echo "===================="
echo "训练脚本生成完成!"
echo "请根据需要修改路径和参数"
echo "===================="

echo ""
echo "主要改进说明:"
echo "1. 加权采样: 自动处理类别不平衡问题"
echo "2. 余弦调度: 更平滑的学习率衰减"
echo "3. 早停机制: 防止过拟合"
echo "4. 层次学习率: 分类器层使用更高学习率"
echo "5. 保持原始训练逻辑不变" 
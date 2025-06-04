# ContraBERT 增强版本 V2 - 保守优化

基于原始 `vulnerability_detection.py` 的最小化增强版本，保持核心训练逻辑不变，仅添加必要的优化策略。

## 设计原则

1. **保持原有逻辑**: 完全保留原始代码的训练流程和架构
2. **最小化改动**: 仅添加可选的优化功能，默认关闭
3. **向后兼容**: 可以完全按照原始方式运行
4. **渐进改进**: 优化策略可以单独启用/禁用

## 核心改进

### 1. 加权采样 (`--use_weighted_sampling`)
```bash
# 自动处理类别不平衡问题
--use_weighted_sampling
```
- 根据类别频率自动计算样本权重
- 在训练时优先采样少数类样本
- 不改变模型架构，仅改变采样策略

### 2. 早停机制 (`--use_early_stopping`)
```bash
# 防止过拟合
--use_early_stopping --patience=3
```
- 监控验证损失，连续N个epoch无改善则停止
- 避免过度训练，节省计算资源
- patience参数可调节容忍度

### 3. 余弦学习率调度 (`--use_cosine_schedule`)
```bash
# 更平滑的学习率衰减
--use_cosine_schedule
```
- 替换线性学习率衰减为余弦衰减
- 提供更平滑的优化过程
- 通常带来更好的收敛性

### 4. 层次化学习率 (`--use_layer_lr`)
```bash
# 分类器层使用更高学习率
--use_layer_lr
```
- 分类器层学习率 = 2 × 基础学习率
- 其他层使用原始学习率
- 加速下游任务适应

## 使用方法

### 方式1: 完全兼容原始代码
```bash
# 和原始 vulnerability_detection.py 完全一样的用法
python vulnerability_detection_v2.py \
    --model_type=roberta \
    --model_name_or_path="path/to/model" \
    --train_data_file="train.jsonl" \
    --eval_data_file="val.jsonl" \
    --block_size=512 \
    --train_batch_size=16 \
    --learning_rate=2e-5 \
    --num_train_epochs=5 \
    --epoch=5 \
    --do_train --do_eval
```

### 方式2: 启用基础优化
```bash
# 推荐的保守优化设置
python vulnerability_detection_v2.py \
    --model_type=roberta \
    --model_name_or_path="path/to/model" \
    --train_data_file="train.jsonl" \
    --eval_data_file="val.jsonl" \
    --block_size=512 \
    --train_batch_size=16 \
    --learning_rate=1e-5 \
    --num_train_epochs=6 \
    --epoch=6 \
    --weight_decay=0.01 \
    --evaluate_during_training \
    --use_weighted_sampling \
    --use_early_stopping \
    --patience=3 \
    --use_cosine_schedule \
    --do_train --do_eval --do_test
```

### 方式3: 使用训练脚本
```bash
# 运行预配置的训练策略
chmod +x run_enhanced_training_v2.sh
./run_enhanced_training_v2.sh
```

## 训练策略对比

| 策略 | 配置 | 预期提升 | 适用场景 |
|------|------|----------|----------|
| 基线 | 原始设置 | - | 复现原始结果 |
| 策略1 | 基础优化 | +2-4% | 通用改进 |
| 策略2 | +层次学习率 | +3-5% | 进一步优化 |

## 关键差异对比

### vs 原始代码
```diff
+ 加权采样处理类别不平衡
+ 早停机制防止过拟合  
+ 余弦学习率调度
+ 层次化学习率支持
- 保持所有原有逻辑不变
```

### vs 完全重写版本 (vulnerability_detection_enhanced.py)
```diff
+ 保持原有训练循环逻辑
+ 保持原有模型架构
+ 保持原有评估方式
+ 最小化改动风险
- 优化策略相对保守
- 功能相对简单
```

## 参数推荐

### 基础设置
```bash
--learning_rate=1e-5        # 相比原始2e-5更保守
--weight_decay=0.01         # 添加正则化
--block_size=512           # 保持与原始一致
--train_batch_size=16      # 保持与原始一致
--num_train_epochs=6       # 相比原始5略增
```

### 优化设置
```bash
--use_weighted_sampling    # 推荐开启
--use_early_stopping      # 推荐开启
--patience=3              # 适中的容忍度
--use_cosine_schedule     # 推荐开启
--use_layer_lr            # 可选，通常有帮助
```

## 代码结构

```
vulnerability_detection_v2.py          # 主训练脚本
├── create_weighted_sampler()          # 加权采样器
├── create_optimizer_with_decay()      # 层次化优化器
├── create_scheduler()                 # 学习率调度器
├── EarlyStopping                      # 早停类
└── train() # 增强的训练函数
    ├── 保持原有训练循环
    ├── 添加早停检查
    └── 其他逻辑完全不变
```

## 验证建议

1. **首先运行基线**: 确保能复现原始结果
2. **逐步启用优化**: 一次启用一个优化功能
3. **监控性能变化**: 对比准确率、F1和AUC指标
4. **检查过拟合**: 观察训练/验证损失曲线

## 故障排除

### 性能没有提升
1. 检查数据集是否存在类别不平衡
2. 尝试调整学习率和权重衰减
3. 增加训练epoch数或调整早停patience

### 训练不稳定
1. 降低学习率
2. 关闭层次化学习率
3. 使用线性调度器替代余弦调度器

### 内存不足
1. 减少batch_size
2. 减少block_size
3. 关闭加权采样

## 总结

这个增强版本的核心优势是**风险最小化**:
- 保持原有训练逻辑不变
- 所有优化都是可选的
- 可以逐步启用功能
- 易于调试和比较

相比完全重写的版本，这个版本更适合:
- 对原始代码有信心的情况
- 需要稳定可靠结果的项目
- 希望渐进式改进的场景 
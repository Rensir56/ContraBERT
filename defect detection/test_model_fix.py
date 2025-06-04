#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from vulmodel_enhanced import EnhancedVulModel, VulModelWithContrastive

def test_model():
    """测试修复后的模型是否能正常运行"""
    
    # 模拟参数
    class Args:
        dropout_rate = 0.3
        num_attention_heads = 8
        attention_dropout = 0.1
        pooling_strategy = 'attention'
        use_focal_loss = True
        focal_alpha = 2.0
        focal_gamma = 2.0
        use_contrastive = False
    
    args = Args()
    
    # 初始化tokenizer和配置
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    config = RobertaConfig.from_pretrained('microsoft/codebert-base')
    config.num_labels = 1
    
    # 初始化编码器
    encoder = RobertaForSequenceClassification.from_pretrained(
        'microsoft/codebert-base',
        config=config
    )
    
    # 测试EnhancedVulModel
    print("🧪 测试 EnhancedVulModel...")
    model = EnhancedVulModel(encoder, config, tokenizer, args)
    
    # 创建测试输入
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    # 测试前向传播
    try:
        loss, probs = model(input_ids, labels)
        print(f"✅ EnhancedVulModel 测试通过!")
        print(f"   Loss shape: {loss.shape}")
        print(f"   Probs shape: {probs.shape}")
        print(f"   Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ EnhancedVulModel 测试失败: {e}")
        return False
    
    # 测试不带标签的推理
    try:
        probs = model(input_ids)
        print(f"✅ 推理模式测试通过!")
        print(f"   Probs shape: {probs.shape}")
    except Exception as e:
        print(f"❌ 推理模式测试失败: {e}")
        return False
    
    # 测试对比学习模型
    print("\n🧪 测试 VulModelWithContrastive...")
    args.use_contrastive = True
    contrastive_model = VulModelWithContrastive(encoder, config, tokenizer, args)
    
    try:
        contrastive_model.train()  # 设置为训练模式以启用对比学习
        loss, probs = contrastive_model(input_ids, labels)
        print(f"✅ VulModelWithContrastive 测试通过!")
        print(f"   Loss shape: {loss.shape}")
        print(f"   Probs shape: {probs.shape}")
        print(f"   Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ VulModelWithContrastive 测试失败: {e}")
        return False
    
    print("\n🎉 所有测试都通过了!")
    return True

if __name__ == "__main__":
    test_model() 
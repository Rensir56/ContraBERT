import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CosineAnnealingWarmRestarts(_LRScheduler):
    """带热重启的余弦退火学习率调度器"""
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_0) * self.T_mult + self.T_0
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        """保存最佳权重"""
        self.best_weights = model.state_dict().copy()

class GradualUnfreezing:
    """渐进式解冻策略"""
    def __init__(self, model, unfreeze_schedule: List[int]):
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule
        self.current_epoch = 0
        
        # 首先冻结所有预训练层
        self._freeze_all_pretrained()
    
    def _freeze_all_pretrained(self):
        """冻结所有预训练层"""
        if hasattr(self.model, 'encoder'):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
    
    def step(self, epoch):
        """根据epoch解冻相应层"""
        self.current_epoch = epoch
        
        if epoch in self.unfreeze_schedule:
            layers_to_unfreeze = self.unfreeze_schedule.index(epoch) + 1
            self._unfreeze_layers(layers_to_unfreeze)
    
    def _unfreeze_layers(self, num_layers):
        """解冻指定数量的层"""
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'encoder'):
            encoder_layers = self.model.encoder.encoder.layer
            total_layers = len(encoder_layers)
            
            # 从最后几层开始解冻
            for i in range(max(0, total_layers - num_layers), total_layers):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True
                    
            logger.info(f"Unfrozen {num_layers} layers at epoch {self.current_epoch}")

class FocalLossWithLabelSmoothing(nn.Module):
    """结合标签平滑的Focal Loss"""
    def __init__(self, alpha=1, gamma=2, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        # 标签平滑
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # 计算focal loss
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets_smooth, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class MixUp:
    """MixUp数据增强策略"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam

class AdversarialTraining:
    """对抗训练"""
    def __init__(self, model, epsilon=0.01, alpha=0.01, num_steps=1):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
    
    def generate_adversarial_examples(self, inputs, labels):
        """生成对抗样本"""
        # 获取词嵌入
        embeddings = self.model.encoder.embeddings.word_embeddings(inputs)
        
        # 初始化扰动
        delta = torch.zeros_like(embeddings).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True
        
        for _ in range(self.num_steps):
            # 前向传播
            perturbed_embeddings = embeddings + delta
            loss, _ = self.model(inputs_embeds=perturbed_embeddings, labels=labels)
            
            # 反向传播
            loss.backward()
            
            # 更新扰动
            delta.data = delta.data + self.alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            delta.grad.zero_()
        
        return embeddings + delta.detach()

class KnowledgeDistillation:
    """知识蒸馏"""
    def __init__(self, teacher_model, temperature=4.0, alpha=0.7):
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # 设置教师模型为评估模式
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def compute_loss(self, student_logits, teacher_logits, true_labels):
        """计算蒸馏损失"""
        # 软标签损失
        soft_loss = nn.functional.kl_div(
            nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            nn.functional.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 硬标签损失
        hard_loss = nn.functional.cross_entropy(student_logits, true_labels)
        
        # 组合损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return total_loss

class PolyLoss(nn.Module):
    """PolyLoss - 更好的分类损失函数"""
    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, outputs, targets):
        ce_loss = nn.functional.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        poly_loss = ce_loss + self.epsilon * (1 - pt)
        return poly_loss.mean()

class WeightedSampler:
    """类别权重采样器"""
    def __init__(self, labels):
        self.labels = labels
        self.class_counts = np.bincount(labels)
        self.num_samples = len(labels)
        
        # 计算类别权重
        self.class_weights = 1.0 / self.class_counts
        self.sample_weights = self.class_weights[labels]
    
    def get_sampler(self):
        """返回PyTorch的WeightedRandomSampler"""
        from torch.utils.data import WeightedRandomSampler
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=self.num_samples,
            replacement=True
        )

class CurriculumLearning:
    """课程学习策略"""
    def __init__(self, initial_difficulty=0.5, pace_function='linear'):
        self.initial_difficulty = initial_difficulty
        self.pace_function = pace_function
        self.current_epoch = 0
    
    def get_difficulty_threshold(self, epoch, total_epochs):
        """根据训练进度返回难度阈值"""
        progress = epoch / total_epochs
        
        if self.pace_function == 'linear':
            return self.initial_difficulty + progress * (1.0 - self.initial_difficulty)
        elif self.pace_function == 'quadratic':
            return self.initial_difficulty + (progress ** 2) * (1.0 - self.initial_difficulty)
        elif self.pace_function == 'exponential':
            return 1.0 - (1.0 - self.initial_difficulty) * np.exp(-5 * progress)
        else:
            return 1.0  # 默认使用所有数据
    
    def filter_samples(self, dataset, difficulty_scores, threshold):
        """根据难度阈值过滤样本"""
        easy_indices = np.where(difficulty_scores <= threshold)[0]
        return [dataset[i] for i in easy_indices]

def get_optimizer_with_different_lr(model, base_lr=2e-5, classifier_lr=1e-4):
    """为不同层设置不同学习率"""
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'classifier' not in n and p.requires_grad],
            'lr': base_lr
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'classifier' in n and p.requires_grad],
            'lr': classifier_lr
        }
    ]
    
    return torch.optim.AdamW(param_groups, weight_decay=0.01)

def create_advanced_scheduler(optimizer, num_training_steps, warmup_ratio=0.1):
    """创建高级学习率调度器"""
    from transformers import get_cosine_schedule_with_warmup
    
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    ) 
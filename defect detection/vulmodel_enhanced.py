import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, Dropout, LayerNorm
import math

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 计算Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.out(context), attention_weights

class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EnhancedVulModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(EnhancedVulModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        
        # 增强的架构组件
        self.dropout = Dropout(getattr(args, 'dropout_rate', 0.3))
        self.layer_norm = LayerNorm(config.hidden_size)
        
        # 多头注意力层
        self.attention = MultiHeadAttention(
            config.hidden_size, 
            num_heads=getattr(args, 'num_attention_heads', 8),
            dropout=getattr(args, 'attention_dropout', 0.1)
        )
        
        # 多层分类器
        classifier_layers = []
        hidden_sizes = [config.hidden_size, config.hidden_size // 2, config.hidden_size // 4]
        
        for i in range(len(hidden_sizes) - 1):
            classifier_layers.extend([
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_sizes[i + 1])
            ])
        
        classifier_layers.append(nn.Linear(hidden_sizes[-1], config.num_labels))
        self.classifier = nn.Sequential(*classifier_layers)
        
        # 池化策略
        self.pooling_strategy = getattr(args, 'pooling_strategy', 'attention')
        if self.pooling_strategy == 'attention':
            self.attention_pooling = nn.Sequential(
                nn.Linear(config.hidden_size, 1),
                nn.Tanh()
            )
        
        # 损失函数选择
        self.use_focal_loss = getattr(args, 'use_focal_loss', False)
        if self.use_focal_loss:
            self.loss_fn = FocalLoss(alpha=getattr(args, 'focal_alpha', 1), 
                                   gamma=getattr(args, 'focal_gamma', 2))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def pool_hidden_states(self, hidden_states, attention_mask):
        """多种池化策略"""
        if self.pooling_strategy == 'cls':
            return hidden_states[:, 0]  # [CLS] token
        
        elif self.pooling_strategy == 'mean':
            # 平均池化（考虑mask）
            attention_mask = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
            sum_mask = torch.sum(attention_mask, dim=1)
            return sum_embeddings / sum_mask
        
        elif self.pooling_strategy == 'max':
            # 最大池化
            attention_mask = attention_mask.unsqueeze(-1)
            hidden_states = hidden_states.masked_fill(~attention_mask.bool(), -1e9)
            return torch.max(hidden_states, dim=1)[0]
        
        elif self.pooling_strategy == 'attention':
            # 注意力池化
            attention_weights = self.attention_pooling(hidden_states).squeeze(-1)
            attention_mask = attention_mask.float()
            attention_weights = attention_weights.masked_fill(~attention_mask.bool(), -1e9)
            attention_weights = F.softmax(attention_weights, dim=1).unsqueeze(-1)
            return torch.sum(hidden_states * attention_weights, dim=1)
        
        else:
            return hidden_states[:, 0]  # 默认使用[CLS]
    
    def forward(self, input_ids=None, labels=None):
        # 获取基础编码器输出
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        
        # 获取隐藏状态
        if hasattr(encoder_outputs, 'last_hidden_state'):
            hidden_states = encoder_outputs.last_hidden_state
        else:
            hidden_states = encoder_outputs[0]
        
        # 应用层归一化和dropout
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # 多头注意力增强
        enhanced_states, attention_weights = self.attention(hidden_states, attention_mask)
        
        # 残差连接
        hidden_states = hidden_states + enhanced_states
        
        # 池化获取句子表示
        sentence_embedding = self.pool_hidden_states(hidden_states, attention_mask)
        
        # 分类
        logits = self.classifier(sentence_embedding)
        
        # 计算概率
        if self.config.num_labels == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=-1)
        
        if labels is not None:
            if self.config.num_labels == 1:
                labels = labels.float().view(-1, 1)
                loss = self.loss_fn(logits, labels)
            else:
                loss = F.cross_entropy(logits, labels)
            return loss, probs
        else:
            return probs

class ContrastiveLoss(nn.Module):
    """对比学习损失"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        # 计算相似度矩阵
        embeddings = F.normalize(embeddings, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 创建标签掩码
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        
        pos_sim = torch.sum(exp_sim * mask, dim=1, keepdim=True)
        loss = -torch.log(pos_sim / sum_exp_sim)
        
        return loss.mean()

class VulModelWithContrastive(EnhancedVulModel):
    """带对比学习的漏洞检测模型"""
    def __init__(self, encoder, config, tokenizer, args):
        super().__init__(encoder, config, tokenizer, args)
        
        # 对比学习组件
        self.use_contrastive = getattr(args, 'use_contrastive', False)
        if self.use_contrastive:
            self.contrastive_loss = ContrastiveLoss(
                temperature=getattr(args, 'contrastive_temperature', 0.1)
            )
            self.contrastive_weight = getattr(args, 'contrastive_weight', 0.1)
    
    def forward(self, input_ids=None, labels=None):
        # 获取基础输出
        if labels is not None:
            loss, probs = super().forward(input_ids, labels)
            
            # 添加对比学习损失
            if self.use_contrastive and self.training:
                # 获取句子嵌入
                attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
                encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
                hidden_states = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') else encoder_outputs[0]
                sentence_embeddings = self.pool_hidden_states(hidden_states, attention_mask)
                
                # 计算对比损失
                contrastive_loss = self.contrastive_loss(sentence_embeddings, labels)
                total_loss = loss + self.contrastive_weight * contrastive_loss
                
                return total_loss, probs
            else:
                return loss, probs
        else:
            return super().forward(input_ids, labels) 
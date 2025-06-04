import random
import re
import ast
import json
from typing import List, Dict, Any

class CodeAugmentation:
    """代码数据增强工具类"""
    
    def __init__(self):
        # 常见变量名替换
        self.var_replacements = {
            'i': ['idx', 'index', 'counter'],
            'j': ['jdx', 'j_idx', 'inner_i'],
            'k': ['k_idx', 'key', 'cnt'],
            'temp': ['tmp', 'temporary', 'temp_val'],
            'data': ['info', 'content', 'buffer'],
            'size': ['len', 'length', 'sz'],
            'count': ['cnt', 'num', 'counter']
        }
        
        # 注释模板
        self.comment_templates = [
            "// {comment}",
            "/* {comment} */",
            "// TODO: {comment}",
            "// FIXME: {comment}"
        ]
    
    def variable_renaming(self, code: str) -> str:
        """变量重命名增强"""
        for old_var, new_vars in self.var_replacements.items():
            if old_var in code:
                new_var = random.choice(new_vars)
                # 使用正则表达式替换完整的变量名
                pattern = r'\b' + re.escape(old_var) + r'\b'
                code = re.sub(pattern, new_var, code)
        return code
    
    def add_random_comments(self, code: str) -> str:
        """添加随机注释"""
        lines = code.split('\n')
        comments = [
            "loop iteration",
            "variable initialization", 
            "condition check",
            "memory allocation",
            "bounds checking"
        ]
        
        # 随机在某些行添加注释
        for i in range(len(lines)):
            if random.random() < 0.1:  # 10%概率添加注释
                comment = random.choice(comments)
                template = random.choice(self.comment_templates)
                lines[i] = lines[i] + " " + template.format(comment=comment)
        
        return '\n'.join(lines)
    
    def code_reordering(self, code: str) -> str:
        """代码重排序（仅限独立语句）"""
        lines = code.split('\n')
        # 简单的变量声明重排序
        var_declarations = []
        other_lines = []
        
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('int ') or stripped.startswith('char ') or 
                stripped.startswith('float ') or stripped.startswith('double ')) and '=' in stripped:
                var_declarations.append(line)
            else:
                other_lines.append(line)
        
        # 随机打乱变量声明顺序
        random.shuffle(var_declarations)
        
        # 重新组合
        result_lines = []
        var_idx = 0
        for line in lines:
            if line in var_declarations:
                if var_idx < len(var_declarations):
                    result_lines.append(var_declarations[var_idx])
                    var_idx += 1
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def augment_code(self, code: str, num_augmentations: int = 3) -> List[str]:
        """生成多个增强版本的代码"""
        augmented_codes = [code]  # 包含原始代码
        
        for _ in range(num_augmentations):
            aug_code = code
            
            # 随机应用增强技术
            if random.random() < 0.7:
                aug_code = self.variable_renaming(aug_code)
            
            if random.random() < 0.3:
                aug_code = self.add_random_comments(aug_code)
            
            if random.random() < 0.2:
                aug_code = self.code_reordering(aug_code)
            
            augmented_codes.append(aug_code)
        
        return augmented_codes

def create_augmented_dataset(input_file: str, output_file: str, augmentation_ratio: float = 0.3):
    """创建增强数据集"""
    augmenter = CodeAugmentation()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    
    augmented_data = []
    
    for item in data:
        augmented_data.append(item)  # 添加原始数据
        
        # 只对正样本（漏洞代码）进行增强，平衡数据集
        if item['target'] == 1 and random.random() < augmentation_ratio:
            if 'func' in item:  # devign格式
                original_code = item['func']
                augmented_codes = augmenter.augment_code(original_code, num_augmentations=2)
                
                for i, aug_code in enumerate(augmented_codes[1:], 1):  # 跳过原始代码
                    new_item = item.copy()
                    new_item['func'] = aug_code
                    new_item['idx'] = f"{item['idx']}_aug_{i}"
                    augmented_data.append(new_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"原始数据: {len(data)}, 增强后数据: {len(augmented_data)}")

if __name__ == "__main__":
    # 使用示例
    create_augmented_dataset(
        "data/finetune_data/c_vulnerability/devign/train.jsonl",
        "data/finetune_data/c_vulnerability/devign/train_augmented.jsonl",
        augmentation_ratio=0.3
    ) 
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoModel
import base_llm.config as config


class LoRALinear(nn.Module):
    """标准 LoRA：包装原始 Linear，添加旁路"""
    def __init__(self, original_linear, r=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # LoRA 参数
        self.lora_B = nn.Parameter(
            torch.randn(self.in_features, r, 
                       device=original_linear.weight.device,
                       dtype=original_linear.weight.dtype) * 0.01
        )
        self.lora_A = nn.Parameter(
            torch.zeros(r, self.out_features,
                       device=original_linear.weight.device,
                       dtype=original_linear.weight.dtype)
        )
        self.scaling = alpha / r

        # 冻结原始权重
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        return self.original(x) + (x @ self.lora_B @ self.lora_A) * self.scaling


def replace_bert_attention_with_lora(model, r=8, alpha=16, target_modules=['query', 'value']):
    """替换 BERT 所有层的 Q/V 为 LoRALinear"""
    for layer in model.encoder.layer:
        if 'query' in target_modules:
            layer.attention.self.query = LoRALinear(
                layer.attention.self.query,
                r=r,
                alpha=alpha
            )
        if 'value' in target_modules:
            layer.attention.self.value = LoRALinear(
                layer.attention.self.value,
                r=r,
                alpha=alpha
            )
    return model


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # 加载 BERT
        self.base = AutoModel.from_pretrained(
            config.MODEL_PATH,
            local_files_only=True
        )

        # ===== 新增：梯度检查点 =====
        for parameter in self.base.parameters():
            parameter.requires_grad = False


        # 替换 Q/V 为 LoRA
        self.base = replace_bert_attention_with_lora(
            self.base,
            r=config.LORA_R,
            alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES
        )

        hidden = self.base.config.hidden_size
        num_labels = 3 if config.DATASET_NAME == "mnli" else 2
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

    def get_trainable_params(self):
        """返回所有 LoRA 参数（字典）+ 分类器"""
        lora_params = {}
        for name, param in self.base.named_parameters():
            if 'lora_' in name:
                lora_params[name] = param.data.clone()
        return {
            'lora_params': lora_params,
            'classifier_weight': self.classifier.weight.data.clone(),
            'classifier_bias': self.classifier.bias.data.clone(),
        }

    def set_trainable_params(self, params):
        for name, param in self.base.named_parameters():
            if 'lora_' in name and name in params['lora_params']:
                source = params['lora_params'][name]
                param.data.copy_(source.to(device=param.device, dtype=param.dtype))

        weight = params['classifier_weight']
        bias = params['classifier_bias']
        self.classifier.weight.data.copy_(
            weight.to(
                device=self.classifier.weight.device,
                dtype=self.classifier.weight.dtype,
            )
        )
        self.classifier.bias.data.copy_(
            bias.to(
                device=self.classifier.bias.device,
                dtype=self.classifier.bias.dtype,
            )
        )

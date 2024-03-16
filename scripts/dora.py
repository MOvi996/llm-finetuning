from lora import LoRALayer
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from transformers import AutoModelForCausalLM

class AutoModelwithDoRA(nn.Module):
    def __init__(self, model_name, rank, alpha):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.rank = rank
        self.alpha = alpha
        self.modify_model_for_dora()

    def modify_model_for_dora(self):
        for param in self.model.parameters():
            param.requires_grad = False

        assign_dora = partial(LinearWithDoRA, rank=self.rank, alpha=self.alpha)

        for layer in self.model.model.layers:
            layer.self_attn.q_proj = assign_dora(layer.self_attn.q_proj)
            layer.self_attn.v_proj = assign_dora(layer.self_attn.v_proj)
            
        return self.model
    
    def forward(self, **kwargs):
        return self.model(**kwargs)




# Code inspired by https://github.com/catid/dora/blob/main/dora.py
class LinearWithDoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
        
        self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True))

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        combined_weight = self.linear.weight + self.lora.alpha*lora.T
        column_norm = combined_weight.norm(p=2, dim=0, keepdim=True)
        V = combined_weight / column_norm
        new_weight = self.m * V
        return F.linear(x, new_weight, self.linear.bias)
import torch
from functools import partial
from transformers import AutoModelForCausalLM

# code inspired from https://lightning.ai/lightning-ai/studios/code-lora-from-scratch?tab=files&layout=column&path=cloudspaces%2F01hm9hypqc6y1hrapb5prmtz0h&y=5&x=0

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev, requires_grad=True)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim), requires_grad=True)
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
class AutoModelwithLoRA(torch.nn.Module):
    def __init__(self, model_name, rank, alpha):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.rank = rank
        self.alpha = alpha
        self.modify_model_for_lora()

    def modify_model_for_lora(self):
        for param in self.model.parameters():
            param.requires_grad = False

        assign_lora = partial(LinearWithLoRA, rank=self.rank, alpha=self.alpha)

        for layer in self.model.model.layers:
            layer.self_attn.q_proj = assign_lora(layer.self_attn.q_proj)
            layer.self_attn.v_proj = assign_lora(layer.self_attn.v_proj)
            
        return self.model
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    


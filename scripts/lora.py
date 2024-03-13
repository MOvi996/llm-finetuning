import torch
from functools import partial
from transformers import AutoModelForCausalLM

MODEL_NAME = "facebook/xglm-564M"

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
    


def modify_model_for_lora(model, rank=8, alpha=16):

    # default hyperparameter choices
    lora_query = True
    lora_key = False
    lora_value = True
    lora_out = False
    lora_mlp = False
    lora_head = False
    
    for param in model.parameters():
        param.requires_grad = False
    
    assign_lora = partial(LinearWithLoRA, rank=rank, alpha=alpha)

    for layer in model.model.layers:
        if lora_query:
            layer.self_attn.q_proj = assign_lora(layer.self_attn.q_proj)
        if lora_key:
            model.layers.self_attn.k_proj = assign_lora(layer.self_attn.k_proj)
        if lora_value:
            layer.self_attn.v_proj = assign_lora(layer.self_attn.v_proj)
        if lora_out:
            layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
        if lora_mlp:
            layer.fc1 = assign_lora(layer.fc1)
            layer.fc2 = assign_lora(layer.fc2)
    if lora_head:
        model.lm_head = assign_lora(model.lm_head)

    return model


if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    print(model)
    model = modify_model_for_lora(model)
    print(model)
    

    

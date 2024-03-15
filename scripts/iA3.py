import torch
import torch.nn as nn
from functools import partial
from transformers import AutoModelForCausalLM, XGLMConfig
from transformers.models.xglm.modeling_xglm import XGLMForCausalLM, XGLMAttention
from functools import partial
from typing import Optional, Tuple
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer



MODEL_NAME = "facebook/xglm-564M"
CONFIG = XGLMConfig.from_pretrained(MODEL_NAME)

class XGLMwithiA3(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.modify_model_for_iA3()

    def modify_model_for_iA3(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for layer in self.model.model.layers:
            layer.self_attn = iA3Attention(layer.self_attn)
            layer.fc1 = iA3Linear(layer.fc1)

        for layer in self.model.model.layers:
            layer.self_attn.l_k.requires_grad = True
            layer.self_attn.l_v.requires_grad = True
            layer.fc1.l_ff.requires_grad = True

        return self.model

    def forward(self, **kwargs):
        return self.model(**kwargs)

class iA3Attention(XGLMAttention):
    def __init__(self, layer_attn):
        super().__init__(layer_attn.embed_dim, layer_attn.num_heads, layer_attn.dropout, layer_attn.is_decoder)
        self.l_k = nn.Parameter(torch.ones((1, 1, layer_attn.embed_dim)), requires_grad=True)
        self.l_v = nn.Parameter(torch.ones((1, 1, layer_attn.embed_dim)), requires_grad=True)

    # overwrite the forward method to multiply the iA3 attention parameters l_k and l_v to key and value
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0] * self.l_k
            value_states = past_key_value[1] * self.l_v
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states) * self.l_k, -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states) * self.l_v, -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states) * self.l_k, -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states) * self.l_v, -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states) * self.l_k, -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states) * self.l_v, -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

    
class iA3Linear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.l_ff = nn.Parameter(torch.ones((1, 1, linear.out_features)))
    
    def forward(self, x):
        return self.linear(x) * self.l_ff
    

# Modify the model to include iA3 attention and feedforward layers    
def modify_model_for_iA3(model):

    for layer in model.model.layers:
        layer.self_attn = iA3Attention(layer.self_attn)
        layer.fc1 = iA3Linear(layer.fc1)

    # freeze all params except the iA3 attention and feedforward layers
    for param in model.parameters():
        param.requires_grad = False

    for layer in model.model.layers:
        layer.self_attn.l_k.requires_grad = True
        layer.self_attn.l_v.requires_grad = True
        layer.fc1.l_ff.requires_grad = True
    
    return model


if __name__ == "__main__":
    print("Loading iA3")
    #     model = AutoModelForC ausalLM.from_pretrained(MODEL_NAME)
    #     model = modify_model_for_iA3(model)

    #     # load flores dataset
    #     dataset = load_dataset("facebook/flores", "eng_Latn")

    # # Tokenize the dataset

    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # def tokenize_function(examples):
    #     return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=64, return_tensors="pt")

    # tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=list(dataset["devtest"].features.keys()))
    # tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], format_kwargs={'dtype': torch.long})
    # def add_labels(example):
    #     return {**example, "labels": example["input_ids"]}

    # tokenized_dataset = tokenized_dataset.map(add_labels)


    # # Prepare the data for training



    # train_dataset = tokenized_dataset["devtest"]
    # train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)


    # # Evaluate the model
    # model.()
    # model.to('cuda')
    # eval_data = next(iter(train_dataloader))
    # eval_data = {k: v.to('cuda') for k, v in eval_data.items()}
    # output = model(**eval_data)
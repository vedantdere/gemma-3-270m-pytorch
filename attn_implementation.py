from typing import Optional
import torch
from torch import nn


def repeat_kv(hidden_states,n_rep):
    batch,num_key_value_heads,seq_len,head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    hidden_states = hidden_states[:,:,None,:,:].expand(batch,num_key_value_heads,n_rep,seq_len,head_dim)
    return hidden_states.reshape(batch,num_key_value_heads*n_rep,seq_len,head_dim)


def eager_paged_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs
):
    key_states = key
    value_states = value

    attn_weights = torch.matmul(query,key_states.transpose(2,3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:,:,:,:key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights,value_states)

    attn_output = attn_output.transpose(1,2).contiguous()
    return attn_output,attn_weights
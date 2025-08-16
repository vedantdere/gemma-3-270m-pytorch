import torch
from torch import nn


class Gemma3TextScalableWordEmbedding(nn.Embedding):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx,
                 embed_scale=1.0):
        super().__init__(num_embeddings,embedding_dim,padding_idx)

        self.register_buffer('embed_scale', torch.tensor(embed_scale))

    def forward(self,input_ids):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)
    
class Gemma3RMSNorm(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config
        self.hidden_size=config.hidden_size
        self.intermediate_size=config.intermediate_size
        self.gate_proj=nn.Linear*self.hidden_size,self.intermediate_size,bias=False
        self.up_proj=nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
        self.down_proj=nn.Linear(self.intermediate_size,self.hidden_size,bias=False)
        self.act_fn = nn.GELU(approximate='tanh')

    def forward(self,x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
class Gemma3RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__()

        self.eps=eps
        self.weight=nn.Parameter(torch.zeros(dim))

    def _norm(self,x):
        return x * torch.rsqrt(x.pow(x).mean(-1,keepdim=True) + self.eps)
    
    def forward(self,x):
        output=self._norm(x)
        output=output * (1.0 + self.weight.float())
        return output.type_as(x)
    

class Gemma3RotaryEmbedding(nn.Module):
    def __init__(self,config,device=None):
        super().__init__()

        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        self.max_seq_len_cached=config.max_position_embeddings
        self.original_max_seq_len=config.max_position_embeddings

        self.config=config
        self.rope_init_fn=ROPE_INIT_FUNCTIONS(self.rope_type)

        inv_freq,self.attention_scaling=self.rope_init_fn(self.config,device)

        self.register_buffer("inv_freq", inv_freq)
        self.original_inv_freq=self.inv_freq

    def forward(self,x,position_ids):
        inv_freq_expanded = self.inv_freq[None,:,None].float().expand(position_ids.shape[0],-1,1).to(x.device)
        position_ids_expanded = position_ids[:,None,:].float()

        device_type=x.device.type if isinstance(x.device.type,str) and x.device.type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type,enable=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1,2)
            emb=torch.cat((freqs,freqs),dim=-1)
            cos=emb.cos() * self.attention_scaling
            sin=emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotart_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    cos=cos.unsqueeze(unsqueeze_dim)
    sin=sin.unsqueeze(unsqueeze_dim)

    q_embed=(q*cos) + (rotate_half(q) * sin)
    k_embed=(k*cos) + (rotate_half(k) * sin)
    return q_embed,k_embed


class Gemma3Attention(nn.Module):
    def __init__(self,config,layer_idx):
        super().__init__()

        self.is_sliding=config.layer_type[layer_idx]=="sliding_attention"

        self.config=config
        self.layer_idx=layer_idx

        self.head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups=config.num_attention_heads//config.num_key_value_heads if hasattr(config, "num_key_value_heads") else 1

        self.scaling=config.query_pre_attn_scalar**-0.5

        self.is_causal=True

        self.q_proj = nn.Linear(
            config.hidden_size,config.num_attention_heads*self.head_dim,bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size,config.num_attention_heads*self.head_dim,bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size,config.num_attention_heads*self.head_dim,bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads*self.head_dim,config.hidden_size,bias=config.attention_bias
        )

        self.attn_logit_softcapping=self.config.attn_logit_softcapping
        self.sliding_window=config.sliding_window if self.is_sliding else None

        self.q_norm=Gemma3RMSNorm(dim=config.head_dim,eps=config.rms_norm_eps)
        self.k_norm=Gemma3RMSNorm(dim=config.head_dim,eps=config.rms_norm_eps)

    def forward(self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                cache_posotion=None,
                **kwargs):
        
        input_shape=hidden_states.shape[:-1]
        hidden_shape=(*input_shape,-1,self.head_dim)

        query_states=self.q_proj(hidden_states).view(hidden_states).transpose(1,2)
        key_states=self.k_proj(hidden_states).view(hidden_shape).transpose(1,2)
        value_states=self.v_proj(hidden_states).view(hidden_shape).transpose(1,2)

        query_states=self.q_norm(query_states)
        key_states=self.k_norm(key_states)

        cos,sin=position_embeddings

        query_states,key_states=apply_rotart_pos_emb(query_states,key_states,cos,sin,position_ids=cache_posotion)

        attention_inference = eager_paged_attention_forward

        attn_output,attn_weights=attention_inference(
            self,
            query_states,
            key_states,
            value_states,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs
        )

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    



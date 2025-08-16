import torch
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from attn_implementation import eager_paged_attention_forward

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
    
class Gemma3MLP(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config
        self.hidden_size=config.hidden_size
        self.intermediate_size=config.intermediate_size
        self.gate_proj=nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
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

        self.is_sliding=config.layer_types[layer_idx]=="sliding_attention"

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
    


class Gemma3DecoderLayer(nn.Module):
    def __init__(self,
                 config,
                 layer_idx):
        super().__init__()

        self.config = config
        self.hidden_size=config.hidden_size
        self.layer_idx=layer_idx
        self.attention_type=config.layer_types[layer_idx]

        self.self_attn=Gemma3Attention(config,layer_idx)
        self.mlp=Gemma3MLP(config)

        self.input_layernorm=Gemma3RMSNorm(self.hidden_size,eps=config.rms_norm_eps)

        self.post_attention_layernorm=Gemma3RMSNorm(self.hidden_size,eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm=Gemma3RMSNorm(self.hidden_size,eps=config.rms_norm_eps)
        self.post_feedforward_layernorm=Gemma3RMSNorm(self.hidden_size,eps=config.rms_norm_eps) 

    def forward(self,
                hidden_states,
                posotion_embeddings_global,
                posotion_embeddings_local,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                output_attention=False,
                use_cache=False,
                cache_position=None,
                **kwargs):
        
        residual=hidden_states

        hidden_states=self.input_layernorm(hidden_states)

        if self.self_attn.is_sliding:
            position_embeddings=posotion_embeddings_local
        else:
            position_embeddings=posotion_embeddings_global 
        
        hidden_states,self_attn_weights=self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
            position_ids,
            past_key_values,
            output_attention,
            use_cache,
            cache_position,
            **kwargs
        )

        hidden_states=self.post_attention_layernorm(hidden_states)

        hidden_states = residual + hidden_states

        residual=hidden_states

        hidden_states=self.pre_feedforward_layernorm(hidden_states)
        hidden_states=self.mlp(hidden_states)
        hidden_states=self.post_feedforward_layernorm(hidden_states)
        hidden_states=residual+hidden_states

        output=(hidden_states,)

        if output_attention:
            output += (self_attn_weights)
        return output
    
class Gemma3TextModel(nn.Module):
    def __init__(self,
                 config):
        super().__init__()

        self.padding_idx=config.pad_token_id
        self.vocab_size=config.vocab_size
        self.config = config

        self.embed_tokens=Gemma3TextScalableWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            self.config.hidden_size**-0.5
        )

        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config,layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb=Gemma3RotaryEmbedding(config)

        self.gradient_checkpointing=False

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                **kwargs):
        
        output_attentions=output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states=(
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)

        hidden_states = input_embeds

        position_embeds_global=self.rotary_emb(hidden_states,position_ids)
        position_embeds_local=self.rotary_emb_local(hidden_states,position_ids)


        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": input_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        all_hidden_states = () if output_hidden_states else None
        all_self_attns=() if output_attentions else None

        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeds_global,
                position_embeddings_local=position_embeds_local,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states=layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states=layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        hidden_states=self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return BaseModelOutputWithPast(
            last_hiddden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns
        )
    

class Gemma3Model(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.model = Gemma3TextModel(config)
        self.vocab_size=config.vocab_size
        self.lm_head=nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                input_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                cache_position=None,
                logits_to_keep=0,
                **kwargs):
        
        outputs = self.model(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            input_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            cache_position,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state
        slice_indices=slice(-logits_to_keep,None) if logits_to_keep > 0 else slice(None,None)
        logits = self.lm_head(hidden_states[:,slice_indices,:])

        return logits
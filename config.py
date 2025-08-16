class PretrainedConfig:
    model_type=""
    base_config_key=""
    sub_configs={}
    has_no_defaults_at_init=None
    attribute_map={}
    base_model_tp_plan=None
    base_model_pp_plan=None
    
    def __init__(self,
                output_hidden_state=False,
                output_attentions=False,
                return_dict=True,
                torchscript=False,
                torch_dtype=False,
                purned_heads=None,
                tie_word_embeddings=True,
                chunk_size_feed_forward=0,
                is_encoder_decoder=False,
                is_decoder=False,
                cross_attention_hidden_size=None,
                and_cross_attention=False,
                tie_encoder_decoder=False,
                tokenizer_class=None,
                prefix=None,
                bos_token_id=None,
                pad_token_id=None,
                eos_token_id=None,
                sep_token_id=None,
                decoder_start_token_id=None
                ):
        self.return_dict=return_dict
        self.output_hidden_states=output_hidden_state
        self.torchscript=torchscript
        self.torch_dtype=torch_dtype
        self._output_attention-output_attentions

        self.pruned_heads=purned_heads if purned_heads is not None else {}
        self.tie_word_embeddings=tie_word_embeddings
        self.chunk_size_feed_forward=chunk_size_feed_forward
        self.is_encoder_decoder=is_encoder_decoder
        self.is_decoder=is_decoder
        self.cross_attention_hidden_size=cross_attention_hidden_size
        self.tie_encoder_decoder=tie_encoder_decoder


class GEMMA3_270M(PretrainedConfig):
    def __init__(self):
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.attn_logit_softcapping = None
        self.bos_token_id = 2
        self.eos_token_id = 1
        self.final_logit_softcapping = None
        self.head_dim = 256
        self.hidden_activation = "silu"
        self.hidden_size = 256
        self.num_hidden_layers = 18
        self.intermediate_size = 2048
        self.layer_types = [
            "sliding_attention" if bool((i+1)%2) else "full_attention" for i in range(self.num_hidden_layers)
        ]

        self.max_position_embeddings = 32768
        self.model_type = "gemma3_270m"
        self.num_attention_heads = 4
        self.num_key_value_heads = 1
        self.pad_token_id = 0
        self.query_pre_attn_scalar = 256
        self.rms_norm_eps = 1e-6
        self.rope_local_base_freq = 10000.0
        self.rope_scaling={
            "rope_type":"yarn",
            "factor":32.0,
            "beta_fast":32.0,
            "beta_slow":1.0,
            "truncate":False
        }
        self.rope_theta = 1000000.0
        self.sliding_window = 512
        self.use_cache = False
        self.output_hidden_states = None
        self.vocab_size = 262144
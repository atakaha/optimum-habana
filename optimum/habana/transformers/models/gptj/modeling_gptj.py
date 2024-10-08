from typing import Optional, Tuple, Union

import habana_frameworks.torch.core as htcore
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gptj.configuration_gptj import GPTJConfig
from transformers.models.gptj.modeling_gptj import (
    GPTJMLP,
    GPTJAttention,
    GPTJBlock,
    GPTJForCausalLM,
    GPTJModel,
    apply_rotary_pos_emb,
    create_sinusoidal_positions,
    logger,
)

'''
# Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.
    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """

    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        print(f"--> sequence_length {sequence_length}, target_length {target_length}, causal_mask {causal_mask.shape}")
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask
'''

class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.matmul(*args, **kwargs)


class KVCache(torch.nn.Module):
    def __init__(self):
        super(KVCache, self).__init__()
        self.cache = None
        self.inp_seq_len = -1

    def allocate(self, inp_seq_len, dtype, device, shape):
        if self.cache is None or self.cache.shape != shape:
            self.inp_seq_len = inp_seq_len
            self.cache = torch.zeros(shape, dtype=dtype, device=device)
        else:
            assert (
                self.inp_seq_len == inp_seq_len
            ), f"inp_seq_len must be the same. self.inp_seq_len:{self.inp_seq_len} inp_seq_len:{inp_seq_len}"
            self.cache.fill_(0)

    def update(self, prev, cur, dim, idx, inp_seq_len):
        orig_cur = cur
        if prev.shape == cur.shape:
            prev.copy_(cur)
            return orig_cur
        if cur.shape[2] > 1 and cur.shape[2] <= prev.shape[2]:
            # Initialize
            prev[:, :, :inp_seq_len, :].copy_(cur)
            return orig_cur
        assert cur.shape[2] == 1, f"Cannot update kv-cache. Unsupported shapes. prev:{prev.shape} cur:{cur.shape}"
        if idx is not None:
            prev.index_copy_(dim, idx - 1, cur)
            return prev
        else:
            return torch.cat((prev, cur), dim=dim)

    def get_shape(self):
        if self.cache is None:
            return None
        return self.cache.shape

    def forward(self, cur, dim, idx):
        return self.update(self.cache, cur, dim, idx, self.inp_seq_len)


class GaudiGPTJAttention(GPTJAttention):
    def __init__(self, config: GPTJConfig, layer_idx=None):
        super().__init__(config, layer_idx)
        self.config = config

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.matmul_qk = Matmul()
        self.matmul_av = Matmul()
        # self.k_cache = KVCache()
        # self.v_cache = KVCache()
        self.inp_seq_len = -1
        self.max_position_embeddings = config.max_position_embeddings

    # def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
    #     cache_shape = (batch_size, self.num_attention_heads, max_seq_len, self.head_dim)
    #     device = self.k_proj.weight.device
    #     dtype = self.config.torch_dtype
    #     self.k_cache.allocate(inp_seq_len, dtype, device, cache_shape)
    #     self.v_cache.allocate(inp_seq_len, dtype, device, cache_shape)

    # def reorder(self, tensor, beam_idx):
    #     updated = tensor.index_select(0, beam_idx)
    #     tensor.copy_(updated)

    # def reorder_kv_cache(self, beam_idx: torch.LongTensor):
    #     if self.k_cache.cache is None:
    #         return (None, None)

    #     self.reorder(self.k_cache.cache, beam_idx)
    #     self.reorder(self.v_cache.cache, beam_idx)

    #     return (self.k_cache.cache.shape, self.v_cache.cache.shape)

    def update_sincos_cache(self, seq_len):
        # Call rotary emb forward() to update cos/sin cache when infering more than self.max_position_embeddings
        # This helps in avoiding creation of these caches during actual model forward pass and
        # reduce memory consumption and improve performance.
        if seq_len > self.max_position_embeddings:
            self.max_position_embeddings = seq_len
            # Update register 'bias' buffer size
            self.bias = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).view(1, 1, seq_len, seq_len)
            # TODO: implement rotary_emb()
            # _, _ = self.rotary_emb(self.k_proj.weight, seq_len=seq_len)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        attn_weights = self.matmul_qk(query, key.transpose(-1, -2))
        attn_weights = attn_weights / self.scale_attn

        if attention_mask is not None: # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            # print(f"query {query.shape}, key {key.shape}, attention_mask {attention_mask.shape}, attn_weights {attn_weights.shape}, causal_mask {causal_mask.shape}")
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = self.matmul_av(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_past: Optional[Cache] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        # token_idx: Optional[torch.Tensor] = None,
        # sin: Optional[torch.Tensor] = None,
        # cos: Optional[torch.Tensor] = None,
        # reuse_cache: Optional[bool] = False,
        # cache_idx: Optional[int] = None,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        """
        Copied from GPTJAttention.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_gptj.py#194
        The only differences are:
        - add new args token_idx
        - add new args reuse_cache
        - add new args cache_idx
        - remove is_torch_fx_proxy
        - optimize KV cache
        - pass sin and cos from upper level as they are identical for each attn block
        """
        # _, q_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim, True).contiguous()
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, True).contiguous()
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, False).contiguous()

        embed_positions = self._get_embed_positions(position_ids)

        repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        sincos = torch.gather(embed_positions, 1, repeated_position_ids)
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]
            # Note: it appears that if we use bf16 RoPE(whether use fused kernel or not), there could be acc issue, hence use fp32 RoPE here Fused kernel feasibility needs to be confirmed in the future
            k_rot = apply_rotary_pos_emb(k_rot.to(torch.float32), sin, cos).to(torch.bfloat16)
            q_rot = apply_rotary_pos_emb(q_rot.to(torch.float32), sin, cos).to(torch.bfloat16)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            key = apply_rotary_pos_emb(key.to(torch.float32), sin, cos).to(torch.bfloat16)
            query = apply_rotary_pos_emb(query.to(torch.float32), sin, cos).to(torch.bfloat16)

        key = key.permute(0, 2, 1, 3).contiguous()
        query = query.permute(0, 2, 1, 3).contiguous()

        if layer_past is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_dim,
                "cache_position": cache_position,
            }
            key, value = layer_past.update(key, value, self.layer_idx, cache_kwargs)

            '''
            key, value = layer_past.update(key, value, self.layer_idx, cache_kwargs)
            past_key = layer_past[0]
            past_value = layer_past[1]

            if token_idx is not None:
                past_key.index_copy_(2, token_idx - 1, key)
                past_value.index_copy_(2, token_idx - 1, value)
                key = past_key
                value = past_value
            else:
                key = torch.cat([past_key, key], dim=-2)
                value = torch.cat([past_value, value], dim=-2)
            '''
        '''
        if use_cache is True:
            if reuse_cache:
                key = self.k_cache(key, 2, token_idx)
                value = self.v_cache(value, 2, token_idx)
                present = (self.k_cache.get_shape(), self.v_cache.get_shape())
            else:
                if token_idx is not None:
                    if layer_past is None:
                        past_key = torch.zeros(key.shape, dtype=self.k_proj.weight.dtype, device=key.device)
                        past_value = torch.zeros(key.shape, dtype=self.k_proj.weight.dtype, device=key.device)
                        layer_past = (past_key, past_value)
                    key = self.k_cache.update(layer_past[0], key, 2, token_idx, self.inp_seq_len)
                    value = self.v_cache.update(layer_past[1], value, 2, token_idx, self.inp_seq_len)
                    present = layer_past
                else:
                    present = (key.to(hidden_states.dtype), value)

            if cache_idx is not None and q_len == 1:
                key = key[:, :, :cache_idx, :]
                value = value[:, :, :cache_idx, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :, :, :cache_idx]
        else:
            present = None
        '''
        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # outputs = (attn_output, present)
        outputs = (attn_output, layer_past)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GaudiGPTJBlock(GPTJBlock):
    """
    Inherits from GPTJBlock: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/gptj/modeling_gptj.py#291
    """

    def __init__(self, config: GPTJConfig, layer_idx=None):
        super().__init__(config, layer_idx=None)
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = torch.nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GaudiGPTJAttention(config, layer_idx)
        self.mlp = GPTJMLP(inner_dim, config)

    # def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
    #     self.attn.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    # def reorder_kv_cache(self, beam_idx: torch.LongTensor):
    #     return self.attn.reorder_kv_cache(beam_idx)

    # def update_sincos_cache(self, seq_len):
    #     self.attn.update_sincos_cache(seq_len)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Cache] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        # token_idx: Optional[torch.Tensor] = None,
        # sin: Optional[torch.Tensor] = None,
        # cos: Optional[torch.Tensor] = None,
        # reuse_cache: Optional[bool] = False,
        # cache_idx: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        """
        Copied from GPTJBlock.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_gptj.py
        The only differences are:
        - add new args token_idx
        - pass sin and cos from upper level as they are identical for each attn block
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            # token_idx=token_idx,
            # reuse_cache=reuse_cache,
            # cache_idx=cache_idx,
            # sin=sin,
            # cos=cos,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class GaudiGPTJModel(GPTJModel):
    """
    Copied from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/gptj/modeling_gptj.py#L480
    """

    # def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
    #     for layer in self.h:
    #         layer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    # def reorder_kv_cache(self, beam_idx: torch.LongTensor):
    #     return tuple(layer.reorder_kv_cache(beam_idx) for layer in self.h)

    # def update_sincos_cache(self, seq_len):
    #     for layer in self.h:
    #         layer.update_sincos_cache(seq_len)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor]]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # token_idx: Optional[torch.Tensor] = None,
        # reuse_cache: Optional[bool] = False,
        # cache_idx: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Copied from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/gptj/modeling_gptj.py#L554
        The only differences are:
        - add new args token_idx
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
                )
        '''
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        '''
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        seq_length = inputs_embeds.shape[1]
        if cache_position is None:
            past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_attention_heads x N x N
        # head_mask has shape n_layer x batch x num_attention_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, seq_length)
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = (-1, seq_length, hidden_states.size(-1))

        next_decoder_cache = None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        htcore.mark_step()
        for i, block in enumerate(self.h):
            htcore.mark_step()
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    causal_mask,
                    position_ids,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                    cache_position,
                )
            else:
                outputs = block(
                    hidden_states=hidden_states,
                    layer_past=past_key_values,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                    # token_idx=token_idx,
                    # reuse_cache=reuse_cache,
                    # cache_idx=cache_idx,
                    # sin=sin,
                    # cos=cos,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                next_decoder_cache = outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GaudiGPTJForCausalLM(GPTJForCausalLM):
    """
    Inherits from GPTJForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_gptj.py
    The only differences are:
    - add new args token_idx
    - add token_idx into model_inputs
    - from step2 when enable KV cache, slice next_input_ids from input_ids base on the token_idx
    - from step2 when enable KV cache, slice next_position_ids from position_ids base on the token_idx
    - from step2 when enable KV cache, slice next_token_type_ids from token_type_ids base on the token_idx
    """

    # def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
    #     self.transformer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    # def reorder_kv_cache(self, beam_idx: torch.LongTensor):
    #     return self.transformer.reorder_kv_cache(beam_idx)

    # def update_sincos_cache(self, seq_len):
    #     self.transformer.update_sincos_cache(seq_len)

    # Copied from transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        # token_idx=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here

        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if inputs_embeds is not None:
                batch_size, sequence_length = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                # "token_idx": token_idx,
                # "reuse_cache": reuse_cache,
                # "cache_idx": kwargs.get("cache_idx"),
            }
        )

        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor]]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # token_idx: Optional[torch.Tensor] = None,
        # reuse_cache: Optional[bool] = False,
        # cache_idx: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(f"GaudiCPTJCausalLm -> forward -> attention_mask  : {attention_mask.shape}")
        # print(f"                                position_ids    : {position_ids}")
        # print(f"                                past_key_values : {past_key_values}")
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            # token_idx=token_idx,
            # reuse_cache=reuse_cache,
            # cache_idx=cache_idx,
        )
        hidden_states = transformer_outputs[0]

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

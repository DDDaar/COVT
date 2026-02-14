# from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
# import torch
# from typing import Optional, List, Union, Tuple
# from torch.nn import CrossEntropyLoss
# import numpy as np
# import transformers.models.qwen2_vl.modeling_qwen2_vl
# import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
# from flash_attn.layers.rotary import apply_rotary_emb
# from liger_kernel.transformers.fused_linear_cross_entropy import (
#     LigerFusedLinearCrossEntropyLoss
# )
# from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
# from liger_kernel.transformers.rms_norm import LigerRMSNorm
# from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb


# def apply_rotary_pos_emb_flashatt_fp32(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
#     tensor_ = tensor.float()
#     cos = freqs.cos().float()
#     sin = freqs.sin().float()
#     output = apply_rotary_emb(tensor_, cos, sin).type_as(tensor)
#     return output

# def replace_qwen_2_with_mixed_modality_forward(use_liger=True):
#     if use_liger:
#         transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen_2_mixed_modality_forward_with_flce
#     else:
#         transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen_2_mixed_modality_forward

# def replace_qwen2_5_with_mixed_modality_forward(use_liger=True):
#     if use_liger:
#         transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_with_flce
#         transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt = (apply_rotary_pos_emb_flashatt_fp32)
#         transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP
#         transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
#         transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = (liger_multimodal_rotary_pos_emb)
#     else:
#         transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward
#         transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt = (apply_rotary_pos_emb_flashatt_fp32)

# def qwen_2_mixed_modality_forward_with_flce(
#     self,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     labels: Optional[torch.LongTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     pixel_values: Optional[torch.Tensor] = None,
#     pixel_values_videos: Optional[torch.FloatTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     rope_deltas: Optional[torch.LongTensor] = None,
#     cache_position: Optional[torch.LongTensor] = None,
# ):
    
#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     if inputs_embeds is None:
#         inputs_embeds = self.model.embed_tokens(input_ids)

#         # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
#         if pixel_values is None and pixel_values_videos is None:
#             # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
#             dummy_pixel = torch.zeros(14308, 1176).to(self.visual.get_device())
#             dummy_grid = torch.tensor([[1, 98, 146]]).to(self.visual.get_device())
            
#             dummy_pixel = dummy_pixel.type(self.visual.get_dtype())
#             image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
#             # Operates as maksed_scatter for the image tokens
#             # However the values are all zeros so it dosen't affect the embeddings.
#             # This could avoid deepspeed error when some batch only has texts.
#             inputs_embeds += image_embeds.mean() * 0

#         if pixel_values is not None:
#             pixel_values = pixel_values.type(self.visual.get_dtype())
#             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
#             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#             n_image_features = image_embeds.shape[0]
#             if n_image_tokens != n_image_features:
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )
#             image_mask = (
#                 (input_ids == self.config.image_token_id)
#                 .unsqueeze(-1)
#                 .expand_as(inputs_embeds)
#                 .to(inputs_embeds.device)
#             )
#             image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

#         if pixel_values_videos is not None:
#             pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
#             video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
#             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
#             n_video_features = video_embeds.shape[0]
#             if n_video_tokens != n_video_features:
#                 raise ValueError(
#                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
#                 )
#             video_mask = (
#                 (input_ids == self.config.video_token_id)
#                 .unsqueeze(-1)
#                 .expand_as(inputs_embeds)
#                 .to(inputs_embeds.device)
#             )
#             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

#         if attention_mask is not None:
#             attention_mask = attention_mask.to(inputs_embeds.device)

#     # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
#     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#         # calculate RoPE index once per generation in the pre-fill stage only
#         if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
#             position_ids, rope_deltas = self.get_rope_index(
#                 input_ids, image_grid_thw, video_grid_thw, attention_mask
#             )
#             self.rope_deltas = rope_deltas
#         # then use the prev pre-calculated rope-deltas to get the correct position ids
#         else:
#             batch_size, seq_length, _ = inputs_embeds.shape
#             delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
#             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#             if cache_position is not None:  # otherwise `deltas` is an int `0`
#                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#             position_ids = position_ids.add(delta)
#             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#     outputs = self.model(
#         input_ids=None,
#         position_ids=position_ids,
#         attention_mask=attention_mask,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#     )

#     hidden_states = outputs[0]

#     loss = None
#     logits = None

#     if self.training and (labels is not None):
#         shift_hidden_states = hidden_states[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()

#         # Flatten tokens
#         shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
#         shift_labels = shift_labels.view(-1)

#         lce = LigerFusedLinearCrossEntropyLoss()
#         loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
#     else:
#         logits = self.lm_head(hidden_states)
#         if labels is not None:
#             # Upcast to float if we need to compute the loss to avoid potential precision issues
#             logits = logits.float()
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return Qwen2VLCausalLMOutputWithPast(
#         loss=loss,
#         logits=logits,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         rope_deltas=self.rope_deltas,
#     )

# def qwen_2_mixed_modality_forward(
#     self,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     labels: Optional[torch.LongTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     pixel_values: Optional[torch.Tensor] = None,
#     pixel_values_videos: Optional[torch.FloatTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     rope_deltas: Optional[torch.LongTensor] = None,
#     cache_position: Optional[torch.LongTensor] = None,
# ):
    
#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     if inputs_embeds is None:
#         inputs_embeds = self.model.embed_tokens(input_ids)

#         # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
#         if pixel_values is None and pixel_values_videos is None:
#             # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
#             dummy_pixel = torch.zeros(14308, 1176).to(self.visual.get_device())
#             dummy_grid = torch.tensor([[1, 98, 146]]).to(self.visual.get_device())
            
#             dummy_pixel = dummy_pixel.type(self.visual.get_dtype())
#             image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
#             # Operates as maksed_scatter for the image tokens
#             # However the values are all zeros so it dosen't affect the embeddings.
#             # This could avoid deepspeed error when some batch only has texts.
#             inputs_embeds += image_embeds.mean() * 0

#         if pixel_values is not None:
#             pixel_values = pixel_values.type(self.visual.get_dtype())
#             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
#             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#             n_image_features = image_embeds.shape[0]
#             if n_image_tokens != n_image_features:
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )
#             image_mask = (
#                 (input_ids == self.config.image_token_id)
#                 .unsqueeze(-1)
#                 .expand_as(inputs_embeds)
#                 .to(inputs_embeds.device)
#             )
#             image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

#         if pixel_values_videos is not None:
#             pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
#             video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
#             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
#             n_video_features = video_embeds.shape[0]
#             if n_video_tokens != n_video_features:
#                 raise ValueError(
#                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
#                 )
#             video_mask = (
#                 (input_ids == self.config.video_token_id)
#                 .unsqueeze(-1)
#                 .expand_as(inputs_embeds)
#                 .to(inputs_embeds.device)
#             )
#             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

#         if attention_mask is not None:
#             attention_mask = attention_mask.to(inputs_embeds.device)

#     # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
#     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#         # calculate RoPE index once per generation in the pre-fill stage only
#         if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
#             position_ids, rope_deltas = self.get_rope_index(
#                 input_ids, image_grid_thw, video_grid_thw, attention_mask
#             )
#             self.rope_deltas = rope_deltas
#         # then use the prev pre-calculated rope-deltas to get the correct position ids
#         else:
#             batch_size, seq_length, _ = inputs_embeds.shape
#             delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
#             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#             if cache_position is not None:  # otherwise `deltas` is an int `0`
#                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#             position_ids = position_ids.add(delta)
#             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#     outputs = self.model(
#         input_ids=None,
#         position_ids=position_ids,
#         attention_mask=attention_mask,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#     )

#     hidden_states = outputs[0]
#     logits = self.lm_head(hidden_states)

#     loss = None
#     if labels is not None:
#         # Upcast to float if we need to compute the loss to avoid potential precision issues
#         logits = logits.float()
#         # Shift so that tokens < n predict n
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         # Flatten the tokens
#         loss_fct = CrossEntropyLoss()
#         shift_logits = shift_logits.view(-1, self.config.vocab_size)
#         shift_labels = shift_labels.view(-1)
#         # Enable model parallelism
#         shift_labels = shift_labels.to(shift_logits.device)
#         loss = loss_fct(shift_logits, shift_labels)

#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return Qwen2VLCausalLMOutputWithPast(
#         loss=loss,
#         logits=logits,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         rope_deltas=self.rope_deltas,
#     )

# def qwen2_5_mixed_modality_forward_with_flce(
#     self,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     labels: Optional[torch.LongTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     pixel_values: Optional[torch.Tensor] = None,
#     pixel_values_videos: Optional[torch.FloatTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     rope_deltas: Optional[torch.LongTensor] = None,
#     cache_position: Optional[torch.LongTensor] = None,
#     second_per_grid_ts: Optional[torch.Tensor] = None,
# ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     if inputs_embeds is None:
#         inputs_embeds = self.model.embed_tokens(input_ids)
    
#         # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
#         if pixel_values is None and pixel_values_videos is None:
#             # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
#             dummy_pixel = torch.zeros(14308, 1176).to(self.visual.device)
#             dummy_grid = torch.tensor([[1, 98, 146]]).to(self.visual.device)
            
#             dummy_pixel = dummy_pixel.type(self.visual.dtype)
#             image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
#             # Operates as maksed_scatter for the image tokens
#             # However the values are all zeros so it dosen't affect the embeddings.
#             # This could avoid deepspeed error when some batch only has texts.
#             inputs_embeds += image_embeds.mean() * 0
            
#         if pixel_values is not None:
#             pixel_values = pixel_values.type(self.visual.dtype)
#             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
#             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#             n_image_features = image_embeds.shape[0]
#             if n_image_tokens != n_image_features:
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )

#             mask = input_ids == self.config.image_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             image_mask = mask_expanded.to(inputs_embeds.device)

#             image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

#         if pixel_values_videos is not None:
#             pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
#             video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
#             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
#             n_video_features = video_embeds.shape[0]
#             if n_video_tokens != n_video_features:
#                 raise ValueError(
#                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
#                 )

#             mask = input_ids == self.config.video_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             video_mask = mask_expanded.to(inputs_embeds.device)

#             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

#         if attention_mask is not None:
#             attention_mask = attention_mask.to(inputs_embeds.device)

#     # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
#     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#         # calculate RoPE index once per generation in the pre-fill stage only
#         if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
#             position_ids, rope_deltas = self.get_rope_index(
#                 input_ids,
#                 image_grid_thw,
#                 video_grid_thw,
#                 second_per_grid_ts,
#                 attention_mask,
#             )
#             self.rope_deltas = rope_deltas
#         # then use the prev pre-calculated rope-deltas to get the correct position ids
#         else:
#             batch_size, seq_length, _ = inputs_embeds.shape
#             delta = (
#                 (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
#                 if cache_position is not None
#                 else 0
#             )
#             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#             if cache_position is not None:  # otherwise `deltas` is an int `0`
#                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#             position_ids = position_ids.add(delta)
#             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#     outputs = self.model(
#         input_ids=None,
#         position_ids=position_ids,
#         attention_mask=attention_mask,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#         cache_position=cache_position,
#     )

#     hidden_states = outputs[0]
    
#     loss = None
#     logits = None

#     if self.training and (labels is not None):
#         shift_hidden_states = hidden_states[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()

#         # Flatten tokens
#         shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
#         shift_labels = shift_labels.view(-1)

#         lce = LigerFusedLinearCrossEntropyLoss()
#         loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
#     else:
#         logits = self.lm_head(hidden_states)
#         if labels is not None:
#             # Upcast to float if we need to compute the loss to avoid potential precision issues
#             logits = logits.float()
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return Qwen2_5_VLCausalLMOutputWithPast(
#         loss=loss,
#         logits=logits,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         rope_deltas=self.rope_deltas,
#     )


# # forward function without using fused linear cross entropy
# def qwen2_5_mixed_modality_forward(
#     self,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     labels: Optional[torch.LongTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     pixel_values: Optional[torch.Tensor] = None,
#     pixel_values_videos: Optional[torch.FloatTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     rope_deltas: Optional[torch.LongTensor] = None,
#     cache_position: Optional[torch.LongTensor] = None,
#     second_per_grid_ts: Optional[torch.Tensor] = None,
# ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     if inputs_embeds is None:
#         inputs_embeds = self.model.embed_tokens(input_ids)
    
#         # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
#         if pixel_values is None and pixel_values_videos is None:
#             # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
#             dummy_pixel = torch.zeros(14308, 1176).to(self.visual.device)
#             dummy_grid = torch.tensor([[1, 98, 146]]).to(self.visual.device)
            
#             dummy_pixel = dummy_pixel.type(self.visual.dtype)
#             image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
#             # Operates as maksed_scatter for the image tokens
#             # However the values are all zeros so it dosen't affect the embeddings.
#             # This could avoid deepspeed error when some batch only has texts.
#             inputs_embeds += image_embeds.mean() * 0
            
#         if pixel_values is not None:
#             pixel_values = pixel_values.type(self.visual.dtype)
#             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
#             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#             n_image_features = image_embeds.shape[0]
#             if n_image_tokens != n_image_features:
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )

#             mask = input_ids == self.config.image_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             image_mask = mask_expanded.to(inputs_embeds.device)

#             image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

#         if pixel_values_videos is not None:
#             pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
#             video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
#             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
#             n_video_features = video_embeds.shape[0]
#             if n_video_tokens != n_video_features:
#                 raise ValueError(
#                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
#                 )

#             mask = input_ids == self.config.video_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             video_mask = mask_expanded.to(inputs_embeds.device)

#             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

#         if attention_mask is not None:
#             attention_mask = attention_mask.to(inputs_embeds.device)

#     # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
#     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#         # calculate RoPE index once per generation in the pre-fill stage only
#         if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
#             position_ids, rope_deltas = self.get_rope_index(
#                 input_ids,
#                 image_grid_thw,
#                 video_grid_thw,
#                 second_per_grid_ts,
#                 attention_mask,
#             )
#             self.rope_deltas = rope_deltas
#         # then use the prev pre-calculated rope-deltas to get the correct position ids
#         else:
#             batch_size, seq_length, _ = inputs_embeds.shape
#             delta = (
#                 (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
#                 if cache_position is not None
#                 else 0
#             )
#             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#             if cache_position is not None:  # otherwise `deltas` is an int `0`
#                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#             position_ids = position_ids.add(delta)
#             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#     outputs = self.model(
#         input_ids=None,
#         position_ids=position_ids,
#         attention_mask=attention_mask,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#         cache_position=cache_position,
#     )

#     hidden_states = outputs[0]
#     logits = self.lm_head(hidden_states)

#     loss = None
#     if labels is not None:
#         # Upcast to float if we need to compute the loss to avoid potential precision issues
#         logits = logits.float()
#         # Shift so that tokens < n predict n
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         # Flatten the tokens
#         loss_fct = CrossEntropyLoss()
#         shift_logits = shift_logits.view(-1, self.config.vocab_size)
#         shift_labels = shift_labels.view(-1)
#         # Enable model parallelism
#         shift_labels = shift_labels.to(shift_logits.device)
#         loss = loss_fct(shift_logits, shift_labels)

#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return Qwen2_5_VLCausalLMOutputWithPast(
#         loss=loss,
#         logits=logits,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         rope_deltas=self.rope_deltas,
#     )

















# #########  训练后，格式不对。。。

# import torch
# from typing import Optional, List, Union, Tuple
# from torch.nn import CrossEntropyLoss
# import numpy as np
# import transformers.models.qwen2_vl.modeling_qwen2_vl
# import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
# from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast

# # --- 修改说明 ---
# # 1. 移除了 flash_attn 和 liger_kernel 的 import
# # 2. 移除了 apply_rotary_pos_emb_flashatt_fp32 (依赖 flash_attn)
# # 3. 移除了带有 _with_flce 后缀的函数 (依赖 liger_kernel)
# # 4. 简化了 replace 函数，仅替换 forward 以修复 DeepSpeed 问题，不强行修改 RoPE 或 MLP/Norm 层

# def replace_qwen_2_with_mixed_modality_forward(use_liger=False):
#     # 无论 use_liger 传什么，都强制使用标准 PyTorch 版本
#     transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen_2_mixed_modality_forward

# def replace_qwen2_5_with_mixed_modality_forward(use_liger=False):
#     # 无论 use_liger 传什么，都强制使用标准 PyTorch 版本
#     transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward
#     # 注意：这里不再替换 apply_rotary_pos_emb_flashatt，
#     # 这里的默认行为会回退到 Transformers 库自带的实现，通常在 NPU 上能正常工作。

# # 这是一个纯 PyTorch 实现的 forward，保留了对 Dummy Image 的处理逻辑
# def qwen_2_mixed_modality_forward(
#     self,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     labels: Optional[torch.LongTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     pixel_values: Optional[torch.Tensor] = None,
#     pixel_values_videos: Optional[torch.FloatTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     rope_deltas: Optional[torch.LongTensor] = None,
#     cache_position: Optional[torch.LongTensor] = None,
# ):
    
#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     if inputs_embeds is None:
#         inputs_embeds = self.model.embed_tokens(input_ids)

#         # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
#         if pixel_values is None and pixel_values_videos is None:
#             # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
#             # 确保 dummy 数据也在正确的 device 上
#             device = self.visual.get_device() if hasattr(self.visual, "get_device") else inputs_embeds.device
#             dtype = self.visual.get_dtype() if hasattr(self.visual, "get_dtype") else inputs_embeds.dtype
            
#             dummy_pixel = torch.zeros(14308, 1176).to(device)
#             dummy_grid = torch.tensor([[1, 98, 146]]).to(device)
            
#             dummy_pixel = dummy_pixel.type(dtype)
#             image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
#             # Operates as maksed_scatter for the image tokens
#             # However the values are all zeros so it dosen't affect the embeddings.
#             # This could avoid deepspeed error when some batch only has texts.
#             inputs_embeds += image_embeds.mean() * 0

#         if pixel_values is not None:
#             pixel_values = pixel_values.type(self.visual.get_dtype())
#             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
#             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#             n_image_features = image_embeds.shape[0]
#             if n_image_tokens != n_image_features:
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )
#             image_mask = (
#                 (input_ids == self.config.image_token_id)
#                 .unsqueeze(-1)
#                 .expand_as(inputs_embeds)
#                 .to(inputs_embeds.device)
#             )
#             image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

#         if pixel_values_videos is not None:
#             pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
#             video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
#             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
#             n_video_features = video_embeds.shape[0]
#             if n_video_tokens != n_video_features:
#                 raise ValueError(
#                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
#                 )
#             video_mask = (
#                 (input_ids == self.config.video_token_id)
#                 .unsqueeze(-1)
#                 .expand_as(inputs_embeds)
#                 .to(inputs_embeds.device)
#             )
#             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

#         if attention_mask is not None:
#             attention_mask = attention_mask.to(inputs_embeds.device)

#     # if we get 4D attention mask we cannot calculate rope deltas anymore.
#     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#         # calculate RoPE index once per generation in the pre-fill stage only
#         if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
#             position_ids, rope_deltas = self.get_rope_index(
#                 input_ids, image_grid_thw, video_grid_thw, attention_mask
#             )
#             self.rope_deltas = rope_deltas
#         # then use the prev pre-calculated rope-deltas to get the correct position ids
#         else:
#             batch_size, seq_length, _ = inputs_embeds.shape
#             delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
#             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#             if cache_position is not None:  # otherwise `deltas` is an int `0`
#                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#             position_ids = position_ids.add(delta)
#             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#     outputs = self.model(
#         input_ids=None,
#         position_ids=position_ids,
#         attention_mask=attention_mask,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#     )

#     hidden_states = outputs[0]
#     logits = self.lm_head(hidden_states)

#     loss = None
#     if labels is not None:
#         # Upcast to float if we need to compute the loss to avoid potential precision issues
#         logits = logits.float()
#         # Shift so that tokens < n predict n
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         # Flatten the tokens
#         loss_fct = CrossEntropyLoss()
#         shift_logits = shift_logits.view(-1, self.config.vocab_size)
#         shift_labels = shift_labels.view(-1)
#         # Enable model parallelism
#         shift_labels = shift_labels.to(shift_logits.device)
#         loss = loss_fct(shift_logits, shift_labels)

#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return Qwen2VLCausalLMOutputWithPast(
#         loss=loss,
#         logits=logits,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         rope_deltas=self.rope_deltas,
#     )

# # 这是一个纯 PyTorch 实现的 forward，保留了对 Dummy Image 的处理逻辑
# def qwen2_5_mixed_modality_forward(
#     self,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     labels: Optional[torch.LongTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     pixel_values: Optional[torch.Tensor] = None,
#     pixel_values_videos: Optional[torch.FloatTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     rope_deltas: Optional[torch.LongTensor] = None,
#     cache_position: Optional[torch.LongTensor] = None,
#     second_per_grid_ts: Optional[torch.Tensor] = None,
# ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     if inputs_embeds is None:
#         inputs_embeds = self.model.embed_tokens(input_ids)
    
#         # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
#         if pixel_values is None and pixel_values_videos is None:
#             # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
#             dummy_pixel = torch.zeros(14308, 1176).to(self.visual.device)
#             dummy_grid = torch.tensor([[1, 98, 146]]).to(self.visual.device)
            
#             dummy_pixel = dummy_pixel.type(self.visual.dtype)
#             image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
#             # Operates as maksed_scatter for the image tokens
#             # However the values are all zeros so it dosen't affect the embeddings.
#             # This could avoid deepspeed error when some batch only has texts.
#             inputs_embeds += image_embeds.mean() * 0
            
#         if pixel_values is not None:
#             pixel_values = pixel_values.type(self.visual.dtype)
#             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
#             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#             n_image_features = image_embeds.shape[0]
#             if n_image_tokens != n_image_features:
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )

#             mask = input_ids == self.config.image_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             image_mask = mask_expanded.to(inputs_embeds.device)

#             image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

#         if pixel_values_videos is not None:
#             pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
#             video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
#             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
#             n_video_features = video_embeds.shape[0]
#             if n_video_tokens != n_video_features:
#                 raise ValueError(
#                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
#                 )

#             mask = input_ids == self.config.video_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             video_mask = mask_expanded.to(inputs_embeds.device)

#             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

#         if attention_mask is not None:
#             attention_mask = attention_mask.to(inputs_embeds.device)

#     # if we get 4D attention mask we cannot calculate rope deltas anymore.
#     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#         # calculate RoPE index once per generation in the pre-fill stage only
#         if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
#             position_ids, rope_deltas = self.get_rope_index(
#                 input_ids,
#                 image_grid_thw,
#                 video_grid_thw,
#                 second_per_grid_ts,
#                 attention_mask,
#             )
#             self.rope_deltas = rope_deltas
#         # then use the prev pre-calculated rope-deltas to get the correct position ids
#         else:
#             batch_size, seq_length, _ = inputs_embeds.shape
#             delta = (
#                 (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
#                 if cache_position is not None
#                 else 0
#             )
#             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#             if cache_position is not None:  # otherwise `deltas` is an int `0`
#                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#             position_ids = position_ids.add(delta)
#             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#     outputs = self.model(
#         input_ids=None,
#         position_ids=position_ids,
#         attention_mask=attention_mask,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#         cache_position=cache_position,
#     )

#     hidden_states = outputs[0]
#     logits = self.lm_head(hidden_states)

#     loss = None
#     if labels is not None:
#         # Upcast to float if we need to compute the loss to avoid potential precision issues
#         logits = logits.float()
#         # Shift so that tokens < n predict n
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         # Flatten the tokens
#         loss_fct = CrossEntropyLoss()
#         shift_logits = shift_logits.view(-1, self.config.vocab_size)
#         shift_labels = shift_labels.view(-1)
#         # Enable model parallelism
#         shift_labels = shift_labels.to(shift_logits.device)
#         loss = loss_fct(shift_logits, shift_labels)

#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return Qwen2_5_VLCausalLMOutputWithPast(
#         loss=loss,
#         logits=logits,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         rope_deltas=self.rope_deltas,
#     )












from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
import torch
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss
import numpy as np
import transformers.models.qwen2_vl.modeling_qwen2_vl
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
# from flash_attn.layers.rotary import apply_rotary_emb
# from liger_kernel.transformers.fused_linear_cross_entropy import (
#     LigerFusedLinearCrossEntropyLoss
# )
# from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
# from liger_kernel.transformers.rms_norm import LigerRMSNorm
# from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb


# --- 修改 2: 手动实现 RoPE 的数学逻辑 (原生 PyTorch) ---
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb_torch(x, cos, sin):
    """
    原生 PyTorch 实现的 rotary embedding。
    公式: x_out = x * cos + rotate_half(x) * sin
    """
    # 确保 cos 和 sin 的维度可以广播到 x
    # flash_attn 的实现通常会自动处理广播，这里手动确保一下
    # x shape: [batch, seq, head, head_dim]
    # cos/sin shape: [seq, head_dim] or similar
    
    # 简单的 unsqueeze 以匹配维度 (如果 cos/sin 维度较少)
    # 注意：Qwen2-VL 的 freqs 通常已经是处理好的形状，但为了保险：
    if cos.ndim < x.ndim:
        print(f'cos的shape是{cos.shape},x的shape是{x.shape}')
        cos = cos.unsqueeze(0).unsqueeze(2) # adjust based on actual input shape logic
        sin = sin.unsqueeze(0).unsqueeze(2)

    return (x * cos) + (rotate_half(x) * sin)

# --- 修改 3: 更新调用函数 ---
def apply_rotary_pos_emb_flashatt_fp32(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # 保持 FP32 精度，因为 RoPE 对精度敏感
    tensor_ = tensor.float()
    cos = freqs.cos().float()
    sin = freqs.sin().float()
    
    # 使用原生 PyTorch 实现替代 flash_attn
    output = apply_rotary_emb_torch(tensor_, cos, sin).type_as(tensor)
    return output


# def apply_rotary_pos_emb_flashatt_fp32(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
#     tensor_ = tensor.float()
#     cos = freqs.cos().float()
#     sin = freqs.sin().float()
#     output = apply_rotary_emb(tensor_, cos, sin).type_as(tensor)
#     return output

def replace_qwen_2_with_mixed_modality_forward(use_liger=True):
    if use_liger:
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen_2_mixed_modality_forward_with_flce
    else:
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen_2_mixed_modality_forward

def replace_qwen2_5_with_mixed_modality_forward(use_liger=True):
    if use_liger:
        # 替换rope、mlp、rmsnorm、forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_with_flce
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt = (apply_rotary_pos_emb_flashatt_fp32)
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = (liger_multimodal_rotary_pos_emb)
    else:
        # 替换forward、rope
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward
        #transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt = (apply_rotary_pos_emb_flashatt_fp32)

def qwen_2_mixed_modality_forward_with_flce(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
):
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)

        # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
        if pixel_values is None and pixel_values_videos is None:
            # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
            dummy_pixel = torch.zeros(14308, 1176).to(self.visual.get_device())
            dummy_grid = torch.tensor([[1, 98, 146]]).to(self.visual.get_device())
            
            dummy_pixel = dummy_pixel.type(self.visual.get_dtype())
            image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
            # Operates as maksed_scatter for the image tokens
            # However the values are all zeros so it dosen't affect the embeddings.
            # This could avoid deepspeed error when some batch only has texts.
            inputs_embeds += image_embeds.mean() * 0

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]

    loss = None
    logits = None

    if self.training and (labels is not None):
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten tokens
        shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
        shift_labels = shift_labels.view(-1)

        lce = LigerFusedLinearCrossEntropyLoss()
        loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
    else:
        logits = self.lm_head(hidden_states)
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

    # 传统模式：元组返回 (if not return_dict)
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    # 结构化对象返回
    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )

def qwen_2_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
):
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)

        # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
        if pixel_values is None and pixel_values_videos is None:
            # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
            dummy_pixel = torch.zeros(14308, 1176).to(self.visual.get_device())
            dummy_grid = torch.tensor([[1, 98, 146]]).to(self.visual.get_device())
            
            dummy_pixel = dummy_pixel.type(self.visual.get_dtype())
            image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
            # Operates as maksed_scatter for the image tokens
            # However the values are all zeros so it dosen't affect the embeddings.
            # This could avoid deepspeed error when some batch only has texts.
            inputs_embeds += image_embeds.mean() * 0

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )

# 带flce优化的qwen2.5 forward函数（liger-kernel）
def qwen2_5_mixed_modality_forward_with_flce(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

#     self,  # 模型实例自身
#     input_ids: torch.LongTensor = None,  # 文本输入ID
#     attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
#     position_ids: Optional[torch.LongTensor] = None,  # 位置编码ID
#     past_key_values: Optional[List[torch.FloatTensor]] = None,  # 缓存的键值对（用于增量生成）
#     inputs_embeds: Optional[torch.FloatTensor] = None,  # 直接输入的嵌入向量
#     labels: Optional[torch.LongTensor] = None,  # 训练标签（用于计算损失）
#     use_cache: Optional[bool] = None,  # 是否使用键值缓存
#     output_attentions: Optional[bool] = None,  # 是否输出注意力权重
#     output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
#     return_dict: Optional[bool] = None,  # 是否以字典形式返回
#     pixel_values: Optional[torch.Tensor] = None,  # 图像像素值
#     pixel_values_videos: Optional[torch.FloatTensor] = None,  # 视频像素值
#     image_grid_thw: Optional[torch.LongTensor] = None,  # 图像网格尺寸（时间、高度、宽度）
#     video_grid_thw: Optional[torch.LongTensor] = None,  # 视频网格尺寸
#     rope_deltas: Optional[torch.LongTensor] = None,  # RoPE位置编码的偏移量
#     cache_position: Optional[torch.LongTensor] = None,  # 缓存位置信息
#     second_per_grid_ts: Optional[torch.Tensor] = None,  # 每个网格的时间（秒）
#     Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:  # 返回类型
    
    # 设置输出配置，优先使用传入参数，否则使用模型配置
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        # 如果没有提供预计算的嵌入，就从input_ids计算
        inputs_embeds = self.model.embed_tokens(input_ids)
    
        # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
        # 如果没有图像和视频输入，创建虚拟数据避免DeepSpeed错误
        if pixel_values is None and pixel_values_videos is None:
            # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
            # 创建虚拟像素值和网格尺寸
            dummy_pixel = torch.zeros(14308, 1176).to(self.visual.device)
            dummy_grid = torch.tensor([[1, 98, 146]]).to(self.visual.device)
            # 确保数据类型与视觉模型一致
            dummy_pixel = dummy_pixel.type(self.visual.dtype)
            # 通过视觉模型处理虚拟数据
            image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
            # Operates as maksed_scatter for the image tokens
            # However the values are all zeros so it dosen't affect the embeddings.
            # This could avoid deepspeed error when some batch only has texts.
            # 添加零值避免影响嵌入，但保持计算图完整
            inputs_embeds += image_embeds.mean() * 0
            
        if pixel_values is not None:
            # 转换数据类型以匹配视觉模型
            pixel_values = pixel_values.type(self.visual.dtype)
            # 通过视觉模型提取图像特征
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            # 检查文本中有多少个图像token，确保与提取的特征数量一致；验证图像token数量与特征数量匹配，防止特征和token位置不匹配
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            # 创建掩码标识图像token的位置
            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)
            
            # 用图像特征替换对应位置的文本嵌入
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            ###########  masked_scatter操作：将image_embeds中的特征复制到inputs_embeds中image_mask为True的位置
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # 视频处理类似于上面的图像
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    # 当 position_ids 没有显式提供，并且：
    # attention_mask 为 None，或者
    # attention_mask 是二维的（batch_size × sequence_length）
    # 这通常发生在第一次生成（prefill）阶段，需要计算初始的位置编码。
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
            # 在预填充阶段（首次生成时）计算RoPE索引
            # 三维rope
            # position_ids：是给当前看得到的 Token 分配的具体坐标。
            # rope_deltas：是给模型记下的“账本”，记录了目前为止所有输入一共占用了多少时空步长。
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            # 保存这个值，是为了下一个 token 生成时，不用重新扫描前面的大图片，直接用这个差值推算新位置
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            # cache_position[0]：代表当前 KV Cache 中已经存了多少个 Token。
            # self.rope_deltas：这是你在上一段代码中存下的“时空跨度”。
            # 含义：delta 算出的是“当前这个新 Token 应该从哪个物理位置开始排队”。它包含了之前所有文本和图片所占用的位置总和。
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            # delta用于计算position_id 
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # 主模型前向传播
    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds, # 传入融合了文本和视觉特征的嵌入
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    # outputs[0]通常是最后一个隐藏层状态
    hidden_states = outputs[0]
    
    loss = None
    logits = None

    if self.training and (labels is not None):
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten tokens
        shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
        shift_labels = shift_labels.view(-1)
        
        # LigerFusedLinearCrossEntropyLoss是优化的融合损失函数，同时执行线性变换和交叉熵
        lce = LigerFusedLinearCrossEntropyLoss()
        loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
    else:
        # 推理时或非融合版本计算
        logits = self.lm_head(hidden_states)
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
    
    # 传统模式：元组返回 (if not return_dict)
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    # 结构化对象返回
    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )


# forward function without using fused linear cross entropy
# 不使用融合线性交叉熵的前向传播
# 始终计算logits，不使用融合损失
def qwen2_5_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
    
        # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
        if pixel_values is None and pixel_values_videos is None:
            # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
            dummy_pixel = torch.zeros(14308, 1176).to(self.visual.device)
            dummy_grid = torch.tensor([[1, 98, 146]]).to(self.visual.device)
            
            dummy_pixel = dummy_pixel.type(self.visual.dtype)
            image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
            # Operates as maksed_scatter for the image tokens
            # However the values are all zeros so it dosen't affect the embeddings.
            # This could avoid deepspeed error when some batch only has texts.
            inputs_embeds += image_embeds.mean() * 0
            
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
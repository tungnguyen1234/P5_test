from dataclasses import dataclass

from transformers import T5ForConditionalGeneration, T5Config

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import copy

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


class P5(T5ForConditionalGeneration):
    """
    P5: Pretrain, Personalized Prompts, and Prediction Paradigm for Recommendation

    This extends T5ForConditionalGeneration with whole word embeddings in the encoder.
    Compatible with transformers 5.x using only the public API.
    """
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
        r"whole_word_embeddings\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)

        # Add whole word embeddings (maximum 512 whole words in source text)
        self.whole_word_embeddings = nn.Embedding(512, config.d_model)
        # Initialize with small values to avoid numerical issues
        nn.init.normal_(self.whole_word_embeddings.weight, mean=0.0, std=0.02)

        self.model_parallel = False
        self.device_map = None

    def _post_init_check(self):
        """Check and fix any NaN weights after initialization."""
        # Check shared embeddings
        if torch.isnan(self.shared.weight).any():
            print("WARNING: NaN detected in shared embeddings, reinitializing...")
            nn.init.normal_(self.shared.weight, mean=0.0, std=0.02)

        # Check whole_word_embeddings
        if torch.isnan(self.whole_word_embeddings.weight).any():
            print("WARNING: NaN detected in whole_word_embeddings, reinitializing...")
            nn.init.normal_(self.whole_word_embeddings.weight, mean=0.0, std=0.02)

        # Check lm_head
        if torch.isnan(self.lm_head.weight).any():
            print("WARNING: NaN detected in lm_head, reinitializing...")
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def extend_vocab(self, vocab_size):
        """Extend vocabulary size for user/item tokens."""
        new_shared = nn.Embedding(vocab_size, self.config.d_model)
        old_weight = self.shared.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_shared.weight.data[:old_vocab_size, :] = old_weight
        self.shared = new_shared

        new_lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        old_weight = self.lm_head.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_lm_head.weight.data[:old_vocab_size, :] = old_weight
        self.lm_head = new_lm_head

        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        self.lm_head.weight = self.shared.weight

        self.config.vocab_size = vocab_size

    def forward(
        self,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,
        return_hidden_state=False,
        **kwargs,
    ):
        # Handle whole word embeddings by computing inputs_embeds
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.shared(input_ids)

            if whole_word_ids is not None:
                # Clamp whole_word_ids to max embedding size (512) to avoid index errors
                whole_word_ids = torch.clamp(whole_word_ids, min=0, max=511)
                whole_word_embeds = self.whole_word_embeddings(whole_word_ids)
                inputs_embeds = inputs_embeds + whole_word_embeds

        # Create attention_mask if not provided (since we pass input_ids=None to parent)
        if attention_mask is None and input_ids is not None:
            pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
            attention_mask = (input_ids != pad_token_id).long()

        # Call parent's forward with inputs_embeds instead of input_ids
        outputs = super().forward(
            input_ids=None,  # Use inputs_embeds instead
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if return_hidden_state:
            return outputs.decoder_hidden_states[-1] if outputs.decoder_hidden_states else outputs.logits

        # Handle custom loss reduction if needed
        if labels is not None and not reduce_loss:
            # Recompute loss without reduction
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            loss = loss_fct(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels.view(-1)
            )
            return P5Seq2SeqLMOutput(
                loss=loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        return P5Seq2SeqLMOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None,
        encoder_outputs=None, **kwargs
    ):
        # Call parent's method
        output = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            encoder_outputs=encoder_outputs,
            **kwargs
        )
        return output


@dataclass
class P5Seq2SeqLMOutput(ModelOutput):
    """
    Output class for P5 sequence-to-sequence models.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

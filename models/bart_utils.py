import copy
import math
from collections import namedtuple
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
import random


class BARTMultiGPUWrapper(nn.Module):
    def __init__(self, device):
        super().__init__()

        self._device = device

        # we shard the model into multiple gpus if possible
        self._device_encoder1 = self._device_encoder2 = None
        self._device_decoder1 = self._device_decoder2 = None

        self._interface = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-6-6')
        self._tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-6-6')

        self._mode = None

    def set_mode(self, mode):
        assert mode in ['train', 'infer']

        if self._mode == mode:
            return

        if mode == 'train' and self._device == 'cuda' \
                and torch.cuda.device_count() >= 4:
            self._device_encoder1 = 'cuda:0'
            self._device_encoder2 = 'cuda:1'
            self._device_decoder1 = 'cuda:2'
            self._device_decoder2 = 'cuda:3'
            self.cuda()

        elif mode == 'train' and self._device == 'cuda' \
                and torch.cuda.device_count() >= 2:
            self._device_encoder1 = self._device_encoder2 = 'cuda:0'
            self._device_decoder1 = self._device_decoder2 = 'cuda:1'
            self.cuda()

        elif self._device == 'cuda':
            self._device_encoder1 = self._device_encoder2 \
                = self._device_decoder1 = self._device_decoder2 = 'cuda:0'
            self.cuda()

        else:
            self._device_encoder1 = self._device_encoder2 \
                = self._device_decoder1 = self._device_decoder2 = self._device

        # Model Sharding
        self.encoder.to(self._device_encoder1)
        self.decoder.to(self._device_decoder1)

        # we shard the second half of encoder and decoder into another gpu if possible
        encoder_layer_num = len(self.encoder.layers)
        for i in range(encoder_layer_num):
            if i >= (encoder_layer_num // 2):
                self.encoder.layers[i].to(self._device_encoder2)
        if self.encoder.layer_norm:
            self.encoder.layer_norm.to(self._device_encoder2)

        decoder_layer_num = len(self.decoder.layers)
        for i in range(decoder_layer_num):
            if i >= (decoder_layer_num // 2):
                self.decoder.layers[i].to(self._device_decoder2)
        if self.decoder.layer_norm:
            self.decoder.layer_norm.to(self._device_decoder2)

        torch.cuda.empty_cache()

        # set mode
        if mode == 'train':
            self.train()
        else:
            self.eval()

        self._mode = mode

    def encode(self, sentence, max_length=1024):
        tokens = self._tokenizer([sentence], return_tensors='pt')['input_ids'][0].tolist()
        while len(tokens) > max_length:
            # cut sentence to max length, keep the eos token
            tokens = tokens[:-2] + tokens[-1:]
        return torch.tensor(tokens).long()

    def encode_long(self, sentence):
        """ encode full text """
        # TODO: Finish this method
        pass

    def forward(self, src_tokens, prev_output_tokens):
        attention_mask = src_tokens.eq(self.config.pad_token_id)

        _, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
            config=self.config,
            input_ids=src_tokens,
            decoder_input_ids=prev_output_tokens,
            causal_mask_dtype=self.model.shared.weight.dtype,
        )

        encoder_out, _, _ = forward_encoder(self=self.encoder,
                                            src_tokens=src_tokens,
                                            attention_mask=attention_mask)

        print("encoder_out: ", encoder_out)
        print(encoder_out.shape)
        x, _, _, _ = forward_decoder(self=self.decoder,
                                     tgt_tokens=prev_output_tokens,
                                     encoder_hidden_states=encoder_out,
                                     encoder_padding_mask=attention_mask,
                                     decoder_padding_mask=decoder_padding_mask,
                                     decoder_causal_mask=causal_mask)
        print("x: ", x)
        print(x.shape)
        lm_logits = F.linear(x, self.model.shared.weight, bias=self._interface.final_logits_bias)

        return lm_logits

    @property
    def config(self):
        return self._interface.model.config

    @property
    def model(self):
        return self._interface.model

    @property
    def encoder(self):
        return self._interface.model.encoder

    @property
    def decoder(self):
        return self._interface.model.decoder


def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


def forward_embedding(self,
                      src_tokens,
                      device_encoder):
    """ Embed the tokens, here the device_encoder is the device of
        first half of encoder"""
    inputs_embeds = self.embed_tokens(src_tokens.to(device_encoder)) \
                    * self.embed_scale
    embed_pos = self.embed_positions(src_tokens.to(device_encoder))
    x = inputs_embeds + embed_pos
    x = self.layernorm_embedding(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    print("after embed: ", x)
    return x


def forward_encoder(self,
                    src_tokens,
                    attention_mask=None,
                    output_attentions=False,
                    output_hidden_states=False):
    """
    Args:
        self: In the model, self is self.encoder
        src_tokens (LongTensor): tokens in the source language of shape
            `(batch, src_len)`
        device_encoder: device of the second half of encoder
        attention_mask (torch.LongTensor): indicating which indices are padding tokens.
    Returns:
        Tuple comprised of:
            - **x** (Tensor): the last encoder layer's output of
              shape `(src_len, batch, embed_dim)`
            - **encoder_states** (List[Tensor]): all intermediate
              hidden states of shape `(src_len, batch, embed_dim)`.
              Only populated if *output_hidden_states:* is True.
            - **all_attentions** (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
    """

    # check attention mask and invert
    if attention_mask is not None:
        attention_mask = invert_mask(attention_mask)

    # B x T x C -> T x B x C
    x = forward_embedding(
        self=self,
        src_tokens=src_tokens,
        device_encoder=self.embed_tokens.weight.device
    )
    x = x.transpose(0, 1)

    encoder_states, all_attentions = [], []
    for i, encoder_layer in enumerate(self.layers):
        # first half and second half of encoder not on the same device
        print(f"we are forward the {i} th layer")
        x = x.to(encoder_layer.fc1.weight.device)
        if output_hidden_states:
            encoder_states.append(x)
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = random.uniform(0, 1)
        if self.training and (dropout_probability < self.layerdrop):  # skip the layer
            print("we skip this layer")
            attn = None
        else:
            print("we enter this layer")
            x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)
            print("we get new x: ", x)

        if output_attentions:
            all_attentions.append(attn)

    if self.layer_norm:
        x = x.to(self.layer_norm.weight.device)
        x = self.layer_norm(x)
    if output_hidden_states:
        encoder_states.append(x)

    # T x B x C -> B x T x C
    encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
    x = x.transpose(0, 1)

    return x, encoder_states, all_attentions


def forward_decoder(
        self,
        tgt_tokens,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_cached_states=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        **unused,
):
    """
    Includes several features from "Jointly Learning to Align and
    Translate with Transformer Models" (Garg et al., EMNLP 2019).

    Args:
        self: In the model, self is self.decoder
        tgt_tokens (LongTensor): previous decoder outputs of shape
            `(batch, tgt_len)`, for teacher forcing
        encoder_hidden_states: output from the encoder, used for
            encoder-side attention
        encoder_padding_mask: for ignoring pad tokens
        decoder_cached_states (dict or None): dictionary used for storing state during generation

    Returns:
        tuple:
            - the decoder's features of shape `(batch, tgt_len, embed_dim)`
            - hidden states
            - attentions
    """
    # check attention mask and invert
    if encoder_padding_mask is not None:
        encoder_padding_mask = invert_mask(encoder_padding_mask)

    # embed positions
    positions = self.embed_positions(tgt_tokens, use_cache=use_cache)

    if use_cache:
        tgt_tokens = tgt_tokens[:, -1:]
        positions = positions[:, -1:]  # happens after we embed them
        # assert input_ids.ne(self.padding_idx).any()

    x = self.embed_tokens(tgt_tokens) * self.embed_scale
    x += positions
    x = self.layernorm_embedding(x)
    x = F.dropout(x, p=self.dropout, training=self.training)

    # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
    x = x.transpose(0, 1)
    encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

    # decoder layers
    all_hidden_states = ()
    all_self_attns = ()
    next_decoder_cache = []
    for idx, decoder_layer in enumerate(self.layers):
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        if output_hidden_states:
            all_hidden_states += (x,)
        dropout_probability = random.uniform(0, 1)
        if self.training and (dropout_probability < self.layerdrop):
            continue

        layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None

        x, layer_self_attn, layer_past = decoder_layer(
            x,
            encoder_hidden_states,
            encoder_attn_mask=encoder_padding_mask,
            decoder_padding_mask=decoder_padding_mask,
            layer_state=layer_state,
            causal_mask=decoder_causal_mask,
            output_attentions=output_attentions,
        )

        if use_cache:
            next_decoder_cache.append(layer_past.copy())

        if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
            x = self.layer_norm(x)
        if output_attentions:
            all_self_attns += (layer_self_attn,)

    # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
    all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
    x = x.transpose(0, 1)
    encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

    if use_cache:
        next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
    else:
        next_cache = None
    return x, next_cache, all_hidden_states, list(all_self_attns)


def _prepare_bart_decoder_inputs(
        config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

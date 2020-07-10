# Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bart.py
import random
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List, Dict

from .generation_utils import *

HIDDEN_SIZE = 1024
LABEL_NUM = 16


class BARTMultiGPUWrapper(nn.Module):
    def __init__(self, device):
        super().__init__()

        self._device = device

        # we shard the model into multiple gpus if possible
        self._device_encoder1 = self._device_encoder2 = None
        self._device_decoder1 = self._device_decoder2 = None

        self._interface = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-6-6')
        self._tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-6-6')

        self._label_output_layer = nn.Linear(HIDDEN_SIZE, LABEL_NUM)

        # initialize the label output layer
        self._label_output_layer.weight.data.normal_(mean=0.0, std=0.02)

        self.pad_label_index = -100
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

        # we shard the second half of encoder and decoder into other gpus if possible
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

        # for calculating lm logits
        self._interface.final_logits_bias = move_device(
            self._interface.final_logits_bias, self._device_decoder2)
        self.model.shared = move_device(self.model.shared, self._device_decoder2)

        # for calculating sequence labeling output
        self._label_output_layer = self._label_output_layer.to(self._device_decoder2)

        torch.cuda.empty_cache()

        # set mode
        if mode == 'train':
            self.train()
        else:
            self.eval()

        self._mode = mode

    def encode(self, sentence, max_length=1024):
        """ encode partial text
            Example output:
            tensor([[0, 9226, 16, 41, 15162, 2]])
        """
        return self._tokenizer([sentence], max_length=max_length,
                               truncation=True, return_tensors='pt')['input_ids']

    def encode_long(self, sentence, max_length=1024):
        """ encode full text using chunking
            Example output:
            [
                tensor([[0, 9226, ..., 6]]),
                tensor([[32, 17, ..., 2]])
            ]
        """
        return_list = []
        token_list = self._tokenizer([sentence], return_tensors='pt')['input_ids'][0].tolist()
        for i in range(0, len(token_list), max_length):
            return_list.append(torch.tensor(token_list[i: i + max_length]).long().unsqueeze(0))
        return return_list

    def encode_target(self, tgt_word_list: List[str], tgt_label_list: List[str],
                      max_length: int, label_to_id: Dict):
        """ use for encoding both tokens and labels on the target side
            Example output:
            (tensor([[0, 627, 2225, 16, 157, 1982, 8, 1365, 7, 1407, 2]]),
             tensor([[7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7]]))
        """
        tokens, labels = [], []
        begin_token = self.config.bos_token_id
        end_token = self.config.eos_token_id
        pad_label_index = self.pad_label_index

        for i, (word, label) in enumerate(zip(tgt_word_list, tgt_label_list)):
            if i == 0:
                current_tokens = self._tokenizer([word], return_tensors='pt')['input_ids'][0].tolist()[1:-1]
            else:
                current_tokens = self._tokenizer([" " + word], return_tensors='pt')['input_ids'][0].tolist()[1:-1]
            if len(current_tokens) > 0:
                tokens.extend(current_tokens)
                labels.extend([label_to_id[label]] + [pad_label_index] * (len(current_tokens) - 1))

        tokens = [begin_token] + tokens + [end_token]

        labels = [label_to_id["O"]] + labels + [label_to_id["O"]]

        assert len(tokens) == len(labels)

        while len(tokens) > max_length:
            tokens = tokens[:-2] + tokens[-1:]
            labels = labels[:-2] + labels[-1:]

        return torch.tensor(tokens).long().unsqueeze(0), torch.tensor(labels).long().unsqueeze(0)

    def get_encoder_out(self, src_list):
        # get the fulltext encoder out
        encoder_out_list = []
        for src_tokens in src_list:
            attention_mask = src_tokens.ne(self.config.pad_token_id)
            individual_encoder_out, _, _ = forward_encoder(
                self=self.encoder,
                src_tokens=src_tokens,
                attention_mask=attention_mask
            )
            encoder_out_list.append(individual_encoder_out)
        encoder_out = torch.cat(encoder_out_list, dim=1)
        return encoder_out

    def get_decoder_out(self, src_tokens, prev_output_tokens,
                        encoder_out, encoder_attention_mask):
        _, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
            config=self.config,
            input_ids=src_tokens,
            decoder_input_ids=prev_output_tokens,
            causal_mask_dtype=self.model.shared.weight.dtype,
        )

        x, _, _ = forward_decoder(
            self=self.decoder,
            tgt_tokens=prev_output_tokens,
            encoder_hidden_states=encoder_out,
            encoder_padding_mask=encoder_attention_mask,
            decoder_padding_mask=decoder_padding_mask,
            decoder_causal_mask=causal_mask
        )
        return x

    def forward(self, src_tokens, prev_output_tokens):
        """ forward partial text """
        attention_mask = src_tokens.ne(self.config.pad_token_id)

        encoder_out, _, _ = forward_encoder(
            self=self.encoder,
            src_tokens=src_tokens,
            attention_mask=attention_mask
        )

        x = self.get_decoder_out(src_tokens, prev_output_tokens, encoder_out, attention_mask)
        lm_logits = F.linear(x, self.model.shared.weight, bias=self._interface.final_logits_bias)

        return lm_logits

    def forward_long(self, src_list, prev_output_tokens, aspect=False):
        """ Forward full text using chunking.
            If aspect=True, then jointly seq2seq and seq labeling
        """
        encoder_out = self.get_encoder_out(src_list)
        src_tokens = torch.cat(src_list, dim=1)

        attention_mask = src_tokens.ne(self.config.pad_token_id)

        seqlab_logits = None
        x = self.get_decoder_out(src_tokens, prev_output_tokens, encoder_out, attention_mask)

        if aspect:
            seqlab_logits = self._label_output_layer(x)

        lm_logits = F.linear(x, self.model.shared.weight, bias=self._interface.final_logits_bias)

        return lm_logits, seqlab_logits

    def long_input_generate(self, src_sent: str, max_length, min_length, num_beams,
                            length_penalty, no_repeat_ngram_size):
        self.set_mode('infer')
        self._interface.eval()

        src_list = self.encode_long(src_sent)

        summary_ids = long_input_generate(
            self=self,
            src_list=src_list,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size
        )
        return [self._tokenizer.decode(g, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False).strip()
                for g in summary_ids]

    @property
    def generate(self):
        return self._interface.generate

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

    @property
    def tokenizer(self):
        return self._tokenizer


def forward_embedding(self, tokens):
    """ Embed the tokens, here the device is the device of
        first half of encoder"""
    inputs_embeds = self.embed_tokens(tokens.to(self.embed_tokens.weight.device)) \
                    * self.embed_scale
    embed_pos = self.embed_positions(tokens.to(self.embed_positions.weight.device))

    inputs_embeds = move_device(inputs_embeds, embed_pos.device)
    x = inputs_embeds + embed_pos

    x = move_device(x, self.layernorm_embedding.weight.device)
    x = self.layernorm_embedding(x)

    x = F.dropout(x, p=self.dropout, training=self.training)
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
    x = forward_embedding(self=self, tokens=src_tokens)
    x = x.transpose(0, 1)

    encoder_states, all_attentions = [], []
    for idx, encoder_layer in enumerate(self.layers):
        # first half and second half of encoder not on the same device
        current_device = encoder_layer.fc1.weight.device
        x = move_device(x, current_device)
        attention_mask = move_device(attention_mask, current_device)
        if output_hidden_states:
            encoder_states.append(x)
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = random.uniform(0, 1)
        if self.training and (dropout_probability < self.layerdrop):  # skip the layer
            attn = None
        else:
            x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

        if output_attentions:
            all_attentions.append(attn)

    if self.layer_norm:
        x = move_device(x, self.layer_norm.weight.device)
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

    Returns:
        tuple:
            - the decoder's features of shape `(batch, tgt_len, embed_dim)`
            - hidden states
            - attentions
    """
    # check attention mask and invert
    if encoder_padding_mask is not None:
        encoder_padding_mask = invert_mask(encoder_padding_mask)

    x = forward_embedding(self=self, tokens=tgt_tokens)

    # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
    x = x.transpose(0, 1)
    encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

    # decoder layers
    all_hidden_states = ()
    all_self_attns = ()
    for idx, decoder_layer in enumerate(self.layers):
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        current_device = decoder_layer.fc1.weight.device

        x = move_device(x, current_device)
        encoder_padding_mask = move_device(encoder_padding_mask, current_device)
        decoder_padding_mask = move_device(decoder_padding_mask, current_device)
        encoder_hidden_states = move_device(encoder_hidden_states, current_device)
        decoder_causal_mask = move_device(decoder_causal_mask, current_device)

        if output_hidden_states:
            all_hidden_states += (x,)
        dropout_probability = random.uniform(0, 1)
        if self.training and (dropout_probability < self.layerdrop):
            continue

        layer_state = None

        x, layer_self_attn, layer_past = decoder_layer(
            x,
            encoder_hidden_states,
            encoder_attn_mask=encoder_padding_mask,
            decoder_padding_mask=decoder_padding_mask,
            layer_state=layer_state,
            causal_mask=decoder_causal_mask,
            output_attentions=output_attentions,
        )

        if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
            x = move_device(x, self.layer_norm.weight.device)
            x = self.layer_norm(x)
        if output_attentions:
            all_self_attns += (layer_self_attn,)

    # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
    all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
    x = x.transpose(0, 1)

    return x, all_hidden_states, list(all_self_attns)


def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


def move_device(tensor, device):
    if tensor is None:
        return None
    else:
        tensor = tensor.to(device)
        return tensor


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


# ===================================================== #
#                      Generation                       #
# ===================================================== #
@torch.no_grad()
def long_input_generate(self: BARTMultiGPUWrapper, src_list, max_length,
                        min_length, num_beams, length_penalty,
                        no_repeat_ngram_size, **model_specific_kwargs) -> torch.LongTensor:
    batch_size = 1

    effective_batch_size = batch_size
    effective_batch_mult = 1

    # Here the input_ids is used to get the encoder output
    encoder_outputs = self.get_encoder_out(src_list)
    input_ids = torch.cat(src_list, dim=1)
    input_ids_len = encoder_outputs.shape[1]
    attention_mask = input_ids.ne(self.config.pad_token_id).long()

    encoder_outputs = (encoder_outputs, [], [])

    # Expand input ids if num_beams > 1 or num_return_sequences > 1

    attention_mask = attention_mask.unsqueeze(1).expand(
        batch_size, effective_batch_mult * num_beams, input_ids_len
    )
    attention_mask = attention_mask.contiguous().view(
        effective_batch_size * num_beams, input_ids_len
    )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

    # create empty decoder_input_ids
    input_ids = torch.full(
        (effective_batch_size * num_beams, 1),
        self.config.decoder_start_token_id,
        dtype=torch.long,
        device=next(self._interface.parameters()).device,
    )
    cur_len = 1

    # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and
    # num_return_sequences > 1)
    expanded_batch_idxs = (
        torch.arange(batch_size)
            .view(-1, 1)
            .repeat(1, num_beams * effective_batch_mult)
            .view(-1)
            .to(input_ids.device)
    )
    # expand encoder_outputs
    encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

    output = _generate_beam_search(
        self=self,
        input_ids=input_ids,
        cur_len=cur_len,
        max_length=max_length,
        min_length=min_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        batch_size=effective_batch_size,
        length_penalty=length_penalty,
        num_beams=num_beams,
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        model_specific_kwargs=model_specific_kwargs,
    )

    return output


def _generate_beam_search(self, input_ids, cur_len, max_length, min_length,
                          no_repeat_ngram_size, batch_size, length_penalty,
                          num_beams, encoder_outputs, attention_mask,
                          model_specific_kwargs):
    """ Generate sequences for each example with beam search.
    """
    # Configuration
    early_stopping = self.config.early_stopping
    use_cache = self.config.use_cache
    bad_words_ids = self.config.bad_words_ids
    eos_token_id = self.config.eos_token_id
    pad_token_id = self.config.pad_token_id
    repetition_penalty = self.config.repetition_penalty
    vocab_size = self.config.vocab_size
    num_return_sequences = self.config.num_return_sequences

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the
    # exact same tokens three times
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # cache compute states
    past = (encoder_outputs, None) if encoder_outputs is not None else None

    # done sentences
    done = [False for _ in range(batch_size)]

    while cur_len < max_length:
        # this prepare inputs for generation is different than the one above
        model_inputs = self._interface.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
        )

        outputs = self._interface(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

        # if model has past, then set the past variable to speed up decoding
        if use_cache(self, outputs, use_cache):
            past = outputs[1]
        if self.config.is_encoder_decoder:
            next_token_logits = self._interface.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

        scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

        scores = postprocess_next_token_scores(
            scores=scores,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=num_beams,
        )

        # We don't do sample
        next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

        # re-organize to group the beam together (we are keeping top hypothesis accross beams)
        next_scores = next_scores.view(
            batch_size, num_beams * vocab_size
        )  # (batch_size, num_beams * vocab_size)

        next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        # next batch beam content
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence, add a pad token
            if done[batch_idx]:
                assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                        eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content, this will get added to next_batch_beam
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(), beam_token_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # once the beam for next step is full, don't add more tokens to it.
                if len(next_sent_beam) == num_beams:
                    break

            # Check if we are done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch and update current length
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        cur_len = cur_len + 1

        # re-order internal states
        if past is not None:
            past = self._interface._reorder_cache(past, beam_idx)

        # extend attention_mask for new generated input if only decoder
        if self.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
                (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
            )

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # depending on whether greedy generation is wanted or not define different output_batch_size and
    # output_num_return_sequences_per_batch
    output_batch_size = batch_size * num_return_sequences
    output_num_return_sequences_per_batch = num_return_sequences

    # select the best hypotheses
    sent_lengths = input_ids.new(output_batch_size)
    best = []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)

    # shorter batches are padded
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = torch.stack(best).type(torch.long).to(next(self._interface.parameters()).device)

    return decoded

import copy
import math
from collections import namedtuple
import torch
from fairseq import search
from fairseq.models import FairseqIncrementalDecoder
from torch import nn
from torch.nn import functional as F

import random

from typing import List, Dict

TextPairData = namedtuple('TextPairData', [
    'src_text', 'tgt_text', 'src_tokens', 'tgt_tokens'])

TransformerEncoderOut = namedtuple('TransformerEncoderOut', [
    'encoder_out', 'encoder_padding_mask', 'encoder_embedding', 'encoder_states'
])

HIDDEN_SIZE = 1024
LABEL_NUM = 16


class BARTModelMultiGPUWrapper(nn.Module):
    def __init__(self, device):
        nn.Module.__init__(self)

        self._device = device

        # we shard the model into 4 gpus if possible
        self._device_encoder1 = self._device_encoder2 = None
        self._device_decoder1 = self._device_decoder2 = None

        self._interface = torch.hub.load('pytorch/fairseq', 'bart.large.cnn')

        self._label_output_layer = nn.Linear(HIDDEN_SIZE, LABEL_NUM)

        # initialize the label output layer
        self._label_output_layer.weight.data.normal_(mean=0.0, std=0.02)

        self.pad_label_index = -100
        self._mode = None

    def set_mode(self, mode):
        assert mode in ['train', 'infer']

        if self._mode == mode:
            return

        if mode == 'train' and self._device == 'cuda' and \
                torch.cuda.device_count() >= 4:
            self._device_encoder1 = 'cuda:0'
            self._device_encoder2 = 'cuda:1'
            self._device_decoder1 = 'cuda:2'
            self._device_decoder2 = 'cuda:3'
            self.cuda()

        elif mode == 'train' and self._device == 'cuda' and \
                torch.cuda.device_count() >= 2:
            self._device_encoder1 = self._device_encoder2 = 'cuda:0'
            self._device_decoder1 = self._device_decoder2 = 'cuda:1'
            self.cuda()

        elif self._device == 'cuda':
            self._device_encoder1 = self._device_encoder2 \
                = self._device_decoder1 = self._device_decoder2 \
                = 'cuda:0'
            self.cuda()

        else:
            self._device_encoder1 = self._device_encoder2 \
                = self._device_decoder1 = self._device_decoder2 \
                = self._device

        self.encoder.to(self._device_encoder1)
        # we shard the second half of encoder into another gpu if possible
        for i in range(len(self.encoder.layers)):
            if i >= 6:
                self.encoder.layers[i] = self.encoder.layers[i].to(self._device_encoder2)
        if self.encoder.layer_norm:
            self.encoder.layer_norm = self.encoder.layer_norm.to(self._device_encoder2)

        self.decoder.to(self._device_decoder1)
        # like the encoder, we also shard the second half of decoder into another gpu
        for i in range(len(self.decoder.layers)):
            if i >= 6:
                self.decoder.layers[i] = self.decoder.layers[i].to(self._device_decoder2)
        if self.decoder.layer_norm:
            self.decoder.layer_norm = self.decoder.layer_norm.to(self._device_decoder2)
        if self.decoder.project_out_dim:
            self.decoder.project_out_dim = self.decoder.project_out_dim.to(self._device_decoder2)

        self._label_output_layer = self._label_output_layer.to(self._device_decoder2)

        torch.cuda.empty_cache()

        if mode == 'train':
            self.train()
        else:
            self.eval()

        self._mode = mode

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = forward_encoder(
            self=self.encoder,
            src_tokens=src_tokens.to(self._device_encoder1),
            src_lengths=src_lengths.to(self._device_encoder1),
            device_encoder=self._device_encoder1)

        for key in encoder_out:
            if isinstance(encoder_out[key], torch.Tensor):
                encoder_out[key] = encoder_out[key].to(self._device_decoder1)

        x, extra = forward_decoder(
            self=self.decoder,
            device_decoder=self._device_decoder1,
            prev_output_tokens=prev_output_tokens.to(self._device_decoder1),
            encoder_out=encoder_out,
            features_only=False)

        return x, extra

    def forward_long(self, src_token_list, src_lengths, prev_output_tokens):
        # the device problem is handled in forward_encoder_long()
        encoder_out = forward_encoder_long(
            self=self.encoder,
            src_token_list=src_token_list,
            src_lengths=src_lengths,
            device_encoder=self._device_encoder1,
            device_decoder=self._device_decoder1
        )

        x, extra = forward_decoder(
            self=self.decoder,
            device_decoder=self._device_decoder1,
            prev_output_tokens=prev_output_tokens.to(self._device_decoder1),
            encoder_out=encoder_out,
            features_only=False
        )

        return x, extra

    def forward_long_double_output(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = forward_encoder_long(
            self=self.encoder,
            src_token_list=src_tokens,
            src_lengths=src_lengths,
            device_encoder=self._device_encoder1,
            device_decoder=self._device_decoder1
        )

        x, extra = forward_decoder(
            self=self.decoder,
            device_decoder=self._device_decoder1,
            prev_output_tokens=prev_output_tokens.to(self._device_decoder1),
            encoder_out=encoder_out,
            features_only=True
        )

        seqlab_output = self._label_output_layer(x)
        original_device = x.deivce
        if self.decoder.share_input_output_embed:
            x = x.to(self.decoder.embed_tokens.weight.device)
        seq2seq_output = self.decoder.output_layer(x)
        seq2seq_output = seq2seq_output.to(original_device)

        return seq2seq_output, seqlab_output

    def forward_double_output(self, src_tokens, src_lengths, prev_output_tokens):
        """ jointly seq2seq and seq labeling using first max_len tokens"""
        # print(src_tokens)  tensor([[    0, 27814,  1001,  ...,   118, 24303,     2]])
        # print(src_lengths)  tensor([1024])
        # print(prev_output_tokens)  tensor([[0, ...., 2]])

        encoder_out = forward_encoder(
            self=self.encoder,
            src_tokens=src_tokens.to(self._device_encoder1),
            src_lengths=src_lengths.to(self._device_encoder1),
            device_encoder=self._device_encoder1
        )

        for key in encoder_out:
            if isinstance(encoder_out[key], torch.Tensor):
                encoder_out[key] = encoder_out[key].to(self._device_decoder1)

        x, extra = forward_decoder(
            self=self.decoder,
            device_decoder=self._device_decoder1,
            prev_output_tokens=prev_output_tokens.to(self._device_decoder1),
            encoder_out=encoder_out,
            features_only=True
        )

        seqlab_output = self._label_output_layer(x)
        original_device = x.deivce
        if self.decoder.share_input_output_embed:
            x = x.to(self.decoder.embed_tokens.weight.device)
        seq2seq_output = self.decoder.output_layer(x)
        seq2seq_output = seq2seq_output.to(original_device)

        return seq2seq_output, seqlab_output

    @property
    def model(self):
        return self._interface.model

    def encode(self, sentence, max_length):
        """ encode the source sentence and preserve only max_length of the text """
        bpe_sentence = '<s> ' + self._interface.bpe.encode(sentence) + ' </s>'
        tokens = self._interface.task.source_dictionary.encode_line(
            bpe_sentence, append_eos=False).tolist()
        while len(tokens) > max_length:
            # cut sentence to max_length
            tokens = tokens[:-2] + tokens[-1:]

        return torch.tensor(tokens).long()

    def encode_long(self, sentence):
        """ encode full text """
        text = sentence
        counter = 1
        token_list = []
        while len(text) > 0:
            tokens = self._interface.encode(text)
            encoded_text = self._interface.decode(tokens)
            text = text[len(encoded_text):]
            if counter == 1:
                # keey only one <bos>
                token_list.append(tokens[:-1])
            else:
                token_list.append(tokens[1:-1])
            counter += 1
        # keey only one <eos>
        end_token = torch.tensor([self._interface.task.source_dictionary.eos()])
        torch.cat((token_list[-1], end_token))
        return token_list

    def encode_target(self,
                      tgt_word_list: List[str],
                      tgt_label_list: List[str],
                      max_length: int,
                      label_to_id: Dict):
        """ use for encoding both tokens and labels on the target side """

        tokens, labels = [], []
        begin_token = self._interface.task.source_dictionary.bos()
        end_token = self._interface.task.source_dictionary.eos()
        pad_label_index = self.pad_label_index

        for i, (word, label) in enumerate(zip(tgt_word_list, tgt_label_list)):
            if i == 0:
                current_tokens = self._interface.encode(word).tolist()[1:-1]
            else:
                current_tokens = self._interface.encode(" " + word).tolist()[1:-1]
            if len(current_tokens) > 0:
                tokens.extend(current_tokens)
                labels.extend([label_to_id[label]] + [pad_label_index] * (len(current_tokens) - 1))

        tokens = [begin_token] + tokens + [end_token]
        # print(self._interface.decode(torch.tensor(tokens)))

        labels = [label_to_id["O"]] + labels + [label_to_id["O"]]

        assert len(tokens) == len(labels)

        while len(tokens) > max_length:
            tokens = tokens[:-2] + tokens[-1:]
            labels = labels[:-2] + labels[-1:]

        return torch.tensor(tokens).long(), torch.tensor(labels).long()

    @property
    def sample(self):
        return self._interface.sample

    def long_input_sample(self, src_sents, beam, lenpen, max_len_b, min_len, no_repeat_ngram_size):
        # generate one at a time
        generator = SequenceGenerator(
            device_encoder=self._device_encoder1,
            device_decoder=self._device_decoder1,
            tgt_dict=self.dictionary,
            beam_size=beam,
            len_penalty=lenpen,
            max_len_b=max_len_b,
            min_len=min_len,
            no_repeat_ngram_size=no_repeat_ngram_size
        )

        src_token_list = self.encode_long(src_sents)
        src_tokens = torch.cat(src_token_list, dim=0)
        src_len_list = torch.tensor([len(t) for t in src_token_list])

        outputs = generator.generate(
            models=[self.model],
            sample={
                'src_tokens': src_tokens.unsqueeze(0),
                'src_lengths': torch.tensor([len(src_tokens)]),
                'src_token_list': src_token_list,
                'src_len_list': src_len_list
            })

        return self.decode(outputs[0][0]['tokens'].cpu())

    @property
    def decode(self):
        return self._interface.decode

    @property
    def encoder(self):
        return self._interface.model.encoder

    @property
    def decoder(self):
        return self._interface.model.decoder

    @property
    def dictionary(self):
        return self._interface.model.decoder.dictionary


def forward_embedding(self, src_tokens, device_embed_tokens, device_encoder):
    # embed tokens and positions
    embed = self.embed_scale * self.embed_tokens(
        src_tokens.to(device_embed_tokens)).to(device_encoder)

    if self.embed_positions is not None:
        x = embed + self.embed_positions(src_tokens)
    if self.layernorm_embedding:
        x = self.layernorm_embedding(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    return x, embed


def forward_encoder(self, src_tokens, src_lengths, device_encoder,
                    cls_input=None, return_all_hiddens=False, **unused):
    """
    Args:
        src_tokens (LongTensor): tokens in the source language of shape
            `(batch, src_len)`
        src_lengths (torch.LongTensor): lengths of each source sentence of
            shape `(batch)`
        return_all_hiddens (bool, optional): also return all of the
            intermediate hidden states (default: False).

    Returns:
        dict:
            - **encoder_out** (Tensor): the last encoder layer's output of
              shape `(src_len, batch, embed_dim)`
            - **encoder_padding_mask** (ByteTensor): the positions of
              padding elements of shape `(batch, src_len)`
            - **encoder_states** (List[Tensor]): all intermediate
              hidden states of shape `(src_len, batch, embed_dim)`.
              Only populated if *return_all_hiddens* is True.
    """
    if self.layer_wise_attention:
        return_all_hiddens = True

    x, encoder_embedding = forward_embedding(
        self=self,
        src_tokens=src_tokens,
        device_embed_tokens=self.embed_tokens.weight.device,
        device_encoder=device_encoder)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    # compute padding mask
    encoder_padding_mask = src_tokens.eq(self.padding_idx)
    if not encoder_padding_mask.any():
        encoder_padding_mask = None

    encoder_states = [] if return_all_hiddens else None

    # encoder layers: 12 in total, 6 in cuda:0, 6 in cuda:1
    for layer in self.layers:
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = random.uniform(0, 1)
        if not self.training or (dropout_probability > self.encoder_layerdrop):
            x = x.to(layer.fc1.weight.device)
            if encoder_padding_mask:
                encoder_padding_mask = encoder_padding_mask.to(layer.fc1.weight.device)
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

    if self.layer_norm:
        x = x.to(self.layer_norm.weight.device)
        x = self.layer_norm(x)
        if return_all_hiddens:
            encoder_states[-1] = x

    return {
        'encoder_out': x,  # T x B x C
        'encoder_padding_mask': encoder_padding_mask,  # B x T
        'encoder_embedding': encoder_embedding,  # B x T x C
        'encoder_states': encoder_states,  # List[T x B x C]
    }


def forward_encoder_long(self, src_token_list, src_lengths,
                         device_encoder, device_decoder,
                         cls_input=None, return_all_hiddens=False,
                         **unused):
    """ when using whole text, for each batch we only put one training sample """
    encoder_out = {
        'encoder_out': [],  # T x B x C
        'encoder_padding_mask': [],  # B x T
        'encoder_embedding': [],  # B x T x C
        'encoder_states': [],  # List[T x B x C]
    }

    for i, src_tokens in enumerate(src_token_list):
        print(f"we are forward the {i} th element in src_token_list")
        if i == 4:
            break
        individual_encoder_out = forward_encoder(
            self=self,
            src_tokens=src_tokens.unsqueeze(0).to(device_encoder),
            src_lengths=src_lengths[i].to(device_encoder),
            device_encoder=device_encoder
        )

        for key in individual_encoder_out:
            if isinstance(individual_encoder_out, torch.Tensor):
                individual_encoder_out[key] = individual_encoder_out[key].to(device_decoder)

        for key in encoder_out:
            if individual_encoder_out[key] is not None:
                encoder_out[key].append(individual_encoder_out[key])

    # concate the results
    if len(encoder_out['encoder_out']) > 0:
        encoder_out['encoder_out'] = torch.cat(encoder_out['encoder_out'], dim=0)
    else:
        encoder_out['encoder_out'] = None

    if len(encoder_out['encoder_padding_mask']) > 0:
        encoder_out['encoder_padding_mask'] = torch.cat(encoder_out['encoder_padding_mask'], dim=1)
    else:
        encoder_out['encoder_padding_mask'] = None

    if len(encoder_out['encoder_embedding']) > 0:
        encoder_out['encoder_embedding'] = torch.cat(encoder_out['encoder_embedding'], dim=1)
    else:
        encoder_out['encoder_embedding'] = None

    if len(encoder_out['encoder_states']) > 0:
        all_encoder_states = []
        num_of_states = len(encoder_out['encoder_states'][0])
        for i in range(num_of_states):
            to_concat = [elem[i] for elem in encoder_out['encoder_states']]
            all_encoder_states.append(torch.cat(to_concat, dim=0))
        encoder_out['encoder_states'] = all_encoder_states
    else:
        encoder_out['encoder_states'] = None

    print("We finished the encoder part")
    return encoder_out


def forward_decoder(
        self,
        prev_output_tokens,
        device_decoder,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args,
):
    """
    Args:
        prev_output_tokens (LongTensor): previous decoder outputs of shape
            `(batch, tgt_len)`, for teacher forcing
        encoder_out (Tensor, optional): output from the encoder, used for
            encoder-side attention
        incremental_state (dict): dictionary used for storing state during
            :ref:`Incremental decoding`
        features_only (bool, optional): only return features without
            applying output layer (default: False).

    Returns:
        tuple:
            - the decoder's output of shape `(batch, tgt_len, vocab)`
            - a dictionary with any model-specific outputs
    """
    x, extra = extract_features(
        self=self,
        device_embed_tokens=self.embed_tokens.weight.device,
        device_decoder=device_decoder,
        prev_output_tokens=prev_output_tokens,
        encoder_out=encoder_out,
        incremental_state=incremental_state,
        **extra_args)

    original_device = x.device
    if not features_only:
        if self.share_input_output_embed:
            x = x.to(self.embed_tokens.weight.device)
        x = self.output_layer(x)
    x = x.to(original_device)
    print("we finished the decoder part")
    return x, extra


def extract_features(
        self,
        prev_output_tokens,
        device_embed_tokens,
        device_decoder,
        encoder_out=None,
        incremental_state=None,
        full_context_alignment=False,
        alignment_layer=None,
        alignment_heads=None,
        **unused,
):
    if alignment_layer is None:
        alignment_layer = len(self.layers) - 1

    # embed positions
    positions = self.embed_positions(
        prev_output_tokens,
        incremental_state=incremental_state,
    ) if self.embed_positions is not None else None

    if incremental_state is not None:
        prev_output_tokens = prev_output_tokens[:, -1:]
        if positions is not None:
            positions = positions[:, -1:]

    # embed tokens and positions
    prev_output_tokens_embedding = self.embed_tokens(
        prev_output_tokens.to(device_embed_tokens)).to(device_decoder)

    x = self.embed_scale * prev_output_tokens_embedding

    if self.project_in_dim is not None:
        x = self.project_in_dim(x)

    if positions is not None:
        x += positions

    if self.layernorm_embedding:
        x = self.layernorm_embedding(x)

    x = F.dropout(x, p=self.dropout, training=self.training)
    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    self_attn_padding_mask = None
    if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
        self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

    # decoder layers
    attn = None
    inner_states = [x]
    for idx, layer in enumerate(self.layers):
        encoder_state = None
        if encoder_out is not None:
            if self.layer_wise_attention:
                encoder_state = encoder_out['encoder_states'][idx]
            else:
                encoder_state = encoder_out['encoder_out']

        if incremental_state is None and not full_context_alignment:
            self_attn_mask = self.buffered_future_mask(x)
        else:
            self_attn_mask = None

        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = random.uniform(0, 1)
        if not self.training or (dropout_probability > self.decoder_layerdrop):
            x = x.to(layer.fc1.weight.device)
            if encoder_state is not None:
                encoder_state = encoder_state.to(layer.fc1.weight.device)
            if self_attn_mask is not None:
                self_attn_mask = self_attn_mask.to(layer.fc1.weight.device)
            if self_attn_padding_mask is not None:
                self_attn_padding_mask = self_attn_padding_mask.to(layer.fc1.weight.device)
            x, layer_attn = layer(
                x,
                encoder_state,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=(idx == alignment_layer),
                need_head_weights=(idx == alignment_layer),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float()

    if attn is not None:
        if alignment_heads is not None:
            attn = attn[:alignment_heads]

        # average probabilities over heads
        attn = attn.mean(dim=0)

    if self.layer_norm:
        x = self.layer_norm(x)

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    if self.project_out_dim is not None:
        x = self.project_out_dim(x)

    return x, {'attn': attn, 'inner_states': inner_states}


class SequenceGenerator(object):
    def __init__(
            self,
            device_encoder,
            device_decoder,
            tgt_dict,
            beam_size=1,
            max_len_a=0,
            max_len_b=200,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.,
            unk_penalty=0.,
            retain_dropout=False,
            sampling=False,
            sampling_topk=-1,
            sampling_topp=-1.0,
            temperature=1.,
            diverse_beam_groups=-1,
            diverse_beam_strength=0.5,
            match_source_len=False,
            no_repeat_ngram_size=0,
    ):
        """Generates translations of a given source sentence.

        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            sampling (bool, optional): sample outputs instead of beam search
                (default: False)
            sampling_topk (int, optional): only sample among the top-k choices
                at each step (default: -1)
            sampling_topp (float, optional): only sample among the smallest set
                of words whose cumulative probability mass exceeds p
                at each step (default: -1.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            diverse_beam_groups/strength (float, optional): parameters for
                Diverse Beam Search sampling
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        self._device_encoder = device_encoder
        self._device_decoder = device_decoder
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert sampling_topp < 0 or sampling, '--sampling-topp requires --sampling'
        assert temperature > 0, '--temperature must be greater than 0'

        if sampling:
            self.search = search.Sampling(tgt_dict, sampling_topk, sampling_topp)
        elif diverse_beam_groups > 0:
            self.search = search.DiverseBeamSearch(tgt_dict, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            self.search = search.LengthConstrainedBeamSearch(
                tgt_dict, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        else:
            self.search = search.BeamSearch(tgt_dict)

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        model = EnsembleModel(models)
        return self._generate(model, sample, **kwargs)

    @torch.no_grad()
    def _generate(
            self,
            model,
            sample,
            prefix_tokens=None,
            bos_token=None,
            **kwargs
    ):
        if not self.retain_dropout:
            model.eval()

        src_tokens = sample['src_tokens']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        src_token_list = sample['src_token_list']
        src_len_list = sample['src_len_list']

        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )

        # compute the encoder output for each beam
        with torch.no_grad():
            raw_encoder_outs = [forward_encoder_long(self=elem.encoder,
                                                     src_token_list=src_token_list,
                                                     src_lengths=src_len_list,
                                                     device_encoder=self._device_encoder,
                                                     device_decoder=self._device_decoder)
                                for elem in model.models]

        encoder_outs = [TransformerEncoderOut(
            encoder_out=raw_encoder_outs[0]['encoder_out'],
            encoder_padding_mask=raw_encoder_outs[0]['encoder_padding_mask'],
            encoder_embedding=raw_encoder_outs[0]['encoder_embedding'],
            encoder_states=raw_encoder_outs[0]['encoder_states']
        )]

        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)

        # initialize buffers
        scores = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn, attn_buf = None, None

        # The blacklist indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then the blacklist would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        blacklist = src_tokens.new_zeros(bsz, beam_size).eq(-1)  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfin_idx):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size or step == max_len:
                return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            bbsz_idx = bbsz_idx.to(tokens.device)
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            assert not tokens_clone.eq(self.eos).any()
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step + 2] if attn is not None else None

            # compute scores per token position
            bbsz_idx = bbsz_idx.to(scores.device)
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                if self.match_source_len and step > src_lengths[unfin_idx]:
                    score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i]
                    else:
                        hypo_attn = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': None,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfin_idx):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                model.reorder_incremental_state(reorder_state)
                encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state)

            lprobs, avg_attn_scores = model.forward_decoder(
                tokens[:, :step + 1], encoder_outs, temperature=self.temperature,
            )

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:
                prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
                prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_mask = prefix_toks.ne(self.pad)
                lprobs[prefix_mask] = -math.inf
                lprobs[prefix_mask] = lprobs[prefix_mask].scatter_(
                    -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
                )
                # if prefix includes eos, then we should make sure tokens and
                # scores are the same across all beams
                eos_mask = prefix_toks.eq(self.eos)
                if eos_mask.any():
                    # validate that the first beam matches the prefix
                    first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
                    eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
                    target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
                    assert (first_beam == target_prefix).all()

                    def replicate_first_beam(tensor, mask):
                        tensor = tensor.view(-1, beam_size, tensor.size(-1))
                        tensor[mask] = tensor[mask][:, :1, :]
                        return tensor.view(-1, tensor.size(-1))

                    # copy tokens, scores and lprobs from the first beam to all beams
                    tokens = replicate_first_beam(tokens, eos_mask_batch_dim)
                    scores = replicate_first_beam(scores, eos_mask_batch_dim)
                    lprobs = replicate_first_beam(lprobs, eos_mask_batch_dim)
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            if self.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                            gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), max_len + 2)
                    attn_buf = attn.clone()
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)

            self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                    return gen_ngrams[bbsz_idx].get(ngram_index, [])

                if step + 2 - self.no_repeat_ngram_size >= 0:
                    # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                    banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
                else:
                    banned_tokens = [[] for bbsz_idx in range(bsz * beam_size)]

                for bbsz_idx in range(bsz * beam_size):
                    lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -math.inf

            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            bbsz_offsets = bbsz_offsets.to(cand_beams.device)
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos, except for blacklisted ones
            # or candidates with a score of -inf
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][blacklist] = 0

            # only consider eos when it's among the top beam_size indices
            eos_bbsz_idx = eos_bbsz_idx.to(cand_bbsz_idx.device)
            torch.masked_select(
                cand_bbsz_idx[:, :beam_size],
                mask=eos_mask[:, :beam_size],
                out=eos_bbsz_idx,
            )

            finalized_sents = set()
            if eos_bbsz_idx.numel() > 0:
                torch.masked_select(
                    cand_scores[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_scores,
                )
                finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores)
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                blacklist = blacklist[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos or
            # blacklisted hypos and values < cand_size indicate candidate
            # active hypos. After this, the min values per row are the top
            # candidate active hypos.

            eos_mask = eos_mask.to('cpu')
            active_mask = buffer('active_mask')
            eos_mask[:, :beam_size] |= blacklist
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, new_blacklist = buffer('active_hypos'), buffer('new_blacklist')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(new_blacklist, active_hypos)
            )

            # update blacklist to ignore any finalized hypos
            blacklist = new_blacklist.ge(cand_size)[:, :beam_size]
            assert (~blacklist).any(dim=1).all()

            active_bbsz_idx = buffer('active_bbsz_idx')
            cand_bbsz_idx = cand_bbsz_idx.to('cpu')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )

            cand_scores = cand_scores.to('cpu')
            scores = scores.to('cpu')
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )

            cand_indices = cand_indices.to('cpu')
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )

            scores_buf = scores_buf.to('cpu')
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )

            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)
        return finalized


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        if all(isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1.):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )

    def _decode_one(
            self, tokens, model, encoder_out, incremental_states, log_probs,
            temperature=1.,
    ):
        tokens = tokens.to(encoder_out.encoder_out.device)
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens, encoder_out=encoder_out, incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            new_order = new_order.to(list(self.incremental_states[model].values())[0]['prev_key'].device)
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)


def reorder_encoder_out(encoder_out, new_order):
    """
    Reorder encoder output according to *new_order*.

    Args:
        encoder_out: output from the ``forward()`` method
        new_order (LongTensor): desired order

    Returns:
        *encoder_out* rearranged according to *new_order*
    """
    if encoder_out.encoder_out is not None:
        new_order = new_order.to(encoder_out.encoder_out.device)
        encoder_out = encoder_out._replace(
            encoder_out=encoder_out.encoder_out.index_select(1, new_order)
        )

    if encoder_out.encoder_padding_mask is not None:
        new_order = new_order.to(encoder_out.encoder_padding_mask.device)
        encoder_out = encoder_out._replace(
            encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(0, new_order)
        )

    if encoder_out.encoder_embedding is not None:
        new_order = new_order.to(encoder_out.encoder_embedding.device)
        encoder_out = encoder_out._replace(
            encoder_embedding=encoder_out.encoder_embedding.index_select(0, new_order)
        )

    if encoder_out.encoder_states is not None:
        for idx, state in enumerate(encoder_out.encoder_states):
            new_order = new_order.to(encoder_out.encoder_states[idx].device)
            encoder_out.encoder_states[idx] = state.index_select(1, new_order)

    return encoder_out

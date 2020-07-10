import pickle
from collections import namedtuple
import random
from tqdm import tqdm, trange
import os
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import List

from .distilbart_utils import BARTMultiGPUWrapper

TextPairData = namedtuple('TextPairData', [
    'src_text', 'tgt_text', 'src_tokens', 'tgt_tokens'
])

TextLabelData = namedtuple('TextLabelData', [
    'src_text', 'tgt_text', 'src_tokens', 'tgt_tokens', 'tgt_labels'
])


class BART:
    def __init__(self, device, src_max_length, tgt_max_length):
        self._device = device

        self._src_max_length = src_max_length
        self._tgt_max_length = tgt_max_length

        self._bart = BARTMultiGPUWrapper(device=device)

        self._optimizer = None
        self._lr_scheduler = None

        self._dataset = {}
        self._best_dev_loss = None

        self._id_to_label = None
        self._label_to_id = None

    def construct_dict(self, labels: List[str]):
        self._id_to_label = {i: label for i, label in enumerate(labels)}
        self._label_to_id = {label: i for i, label in enumerate(labels)}

    def get_optimizer(self, lr, train_steps, warmup_steps,
                      weight_decay, adam_epsilon):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self._bart.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self._bart.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        self._optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
        self._lr_scheduler = get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=train_steps)

    def save_model(self, path):
        torch.save(self._bart.state_dict(), path)
        print(f'Model saved in {path}.')

    def load_model(self, path):
        self._bart.load_state_dict(torch.load(path, map_location=self._device))
        print(f'Model {path} loaded.')

    def load_data(self, set_type, src_texts, tgt_texts, long=False):
        # Load data from pkl file if exists
        if os.path.exists(f'data_{set_type}.pkl'):
            with open(f'data_{set_type}.pkl', 'rb') as f:
                self._dataset[set_type] = pickle.load(f)
                print(f'Loading {set_type} data from data_{set_type}.pkl')
                print(f'#{set_type}: {len(self._dataset[set_type])}')
            return

        assert len(src_texts) == len(tgt_texts)

        self._dataset[set_type] = []
        for src_text, tgt_text in tqdm(zip(src_texts, tgt_texts),
                                       total=len(src_texts),
                                       desc=f'loading {set_type} data'):
            if long:
                src_tokens = self._bart.encode_long(src_text)
            else:
                src_tokens = self._bart.encode(src_text, self._src_max_length)

            tgt_tokens = self._bart.encode(tgt_text, self._tgt_max_length)

            self._dataset[set_type].append(TextPairData(
                src_text=src_text,
                tgt_text=tgt_text,
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens
            ))

        print(f'#{set_type}: {len(self._dataset[set_type])}')

        # Write file to disk if it's the first time
        with open(f'data_{set_type}.pkl', 'wb') as f:
            pickle.dump(self._dataset[set_type], f)

    def load_aspect_data(self, set_type, src_texts, tgt_texts, tgt_words, tgt_labels):
        # Load data from pkl file if exists
        if os.path.exists(f'data_{set_type}.pkl'):
            with open(f'data_{set_type}.pkl', 'rb') as f:
                self._dataset[set_type] = pickle.load(f)
                print(f'Loading {set_type} data from data_{set_type}.pkl')
                print(f'#{set_type}: {len(self._dataset[set_type])}')
            return

        self._dataset[set_type] = []
        for src_text, tgt_text, tgt_word_list, tgt_label_list in tqdm(zip(src_texts,
                                                                          tgt_texts,
                                                                          tgt_words,
                                                                          tgt_labels),
                                                                      total=len(src_texts),
                                                                      desc=f'loading {set_type} data'):
            src_token_list = self._bart.encode_long(src_text)

            tgt_tokens, tgt_labels = self._bart.encode_target(
                tgt_word_list=tgt_word_list,
                tgt_label_list=tgt_label_list,
                max_length=self._tgt_max_length,
                label_to_id=self._label_to_id
            )

            self._dataset[set_type].append(TextLabelData(
                src_text=src_text,
                tgt_text=tgt_text,
                src_tokens=src_token_list,
                tgt_tokens=tgt_tokens,
                tgt_labels=tgt_labels
            ))

        print(f'#{set_type}: {len(self._dataset[set_type])}')

        # Write file to disk if it's the first time
        with open(f'data_{set_type}.pkl', 'wb') as f:
            pickle.dump(self._dataset[set_type], f)

    def train_epoch(self, batch_size, long=False):
        """ Train with only sequence to sequence loss """
        assert 'train' in self._dataset

        random.shuffle(self._dataset['train'])
        for i in trange(0, len(self._dataset['train']), batch_size,
                        desc='BART Training'):
            self._bart.set_mode('train')
            self._bart.train()

            batch = self._dataset['train'][i:i + batch_size]

            self._optimizer.zero_grad()

            # We access the data one by one
            for j in range(0, len(batch)):
                data = batch[j]
                src_tokens = data.src_tokens
                tgt_tokens = data.tgt_tokens

                loss = self._get_seq2seq_loss(
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens,
                    long=long
                )

                loss = loss / batch_size
                loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

    def train_aspect_epoch(self, batch_size):
        """ Train with sequence to sequence loss and sequence labeling loss """
        assert 'train' in self._dataset

        random.shuffle(self._dataset['train'])
        for i in trange(0, len(self._dataset['train']), batch_size,
                        desc='BART Training'):
            self._bart.set_mode('train')
            self._bart.train()

            batch = self._dataset['train'][i:i + batch_size]

            self._optimizer.zero_grad()

            # We process each training data individually
            for j in range(len(batch)):
                src_tokens = batch[j].src_tokens
                tgt_tokens = batch[j].tgt_tokens.unsqueeze(0)

                tgt_labels = batch[j].tgt_labels.unsqueeze(0)

                seq2seq_loss, seqlab_loss = self._get_both_loss(
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens,
                    tgt_labels=tgt_labels
                )

                loss = seq2seq_loss + seqlab_loss
                loss = loss / batch_size

                loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

    def evaluate(self, long=False):
        """ Evaluate with only sequence to sequence loss """
        assert 'val' in self._dataset
        self._bart.set_mode('train')
        self._bart.eval()

        loss_list = []
        for i in range(0, len(self._dataset['val'])):
            data = self._dataset['val'][i]

            src_tokens = data.src_tokens
            tgt_tokens = data.tgt_tokens

            with torch.no_grad():
                loss = self._get_seq2seq_loss(
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens,
                    long=long
                )

            loss_list.append(loss.item())

        return sum(loss_list) / len(loss_list)

    def evaluate_aspect(self):
        """ Evaluate with sequence to sequence loss and sequence labeling loss """
        assert 'val' in self._dataset
        self._bart.set_mode('train')
        self._bart.eval()

        loss_list = []
        for i in range(0, len(self._dataset['val'])):
            data = self._dataset['val'][i]
            src_tokens = data.src_tokens
            tgt_tokens = data.tgt_tokens
            tgt_labels = data.tgt_labels

            with torch.no_grad():
                seq2seq_loss, seqlab_loss = self._get_both_loss(
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens,
                    tgt_labels=tgt_labels
                )
                loss = seq2seq_loss + seqlab_loss

            loss_list.append(loss.item())

        return sum(loss_list) / len(loss_list)

    def generate(self, src_sents: List[str], beam=4, lenpen=2.0, max_len=1024,
                 min_len=100, no_repeat_ngram_size=3):
        """ Using only the first src_max_length tokens to generate """

        # when do generation, we run on one gpu
        self._bart.set_mode('infer')
        self._bart.eval()

        input_ids = self._bart.tokenizer(
            src_sents,
            max_length=self._src_max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )['input_ids'].to(self._device)

        summary_ids = self._bart.generate(
            input_ids=input_ids,
            max_length=max_len,
            min_length=min_len,
            num_beams=beam,
            length_penalty=lenpen,
            no_repeat_ngram_size=no_repeat_ngram_size
        )
        return [self._bart.tokenizer.decode(g, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False).strip()
                for g in summary_ids]

    def long_input_generate(self, src_sent: str, beam=4, lenpen=2.0, max_len=1024,
                            min_len=100, no_repeat_ngram_size=3):
        """ Using the full text tokens to generate """
        self._bart.set_mode('infer')
        self._bart.eval()

        output = self._bart.long_input_generate(
            src_sent=src_sent,
            max_length=max_len,
            min_length=min_len,
            num_beams=beam,
            length_penalty=lenpen,
            no_repeat_ngram_size=no_repeat_ngram_size
        )
        return output

    def _get_seq2seq_loss(self, src_tokens, tgt_tokens, long=False):
        if long:
            logits, _ = self._bart.forward_long(
                src_list=src_tokens,
                prev_output_tokens=tgt_tokens
            )
        else:
            logits = self._bart(
                src_tokens=src_tokens,
                prev_output_tokens=tgt_tokens
            )

        tgt_tokens = tgt_tokens.to(logits.device)

        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = tgt_tokens[:, 1:].contiguous()

        # Flatten the tokens
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self._bart.config.pad_token_id)
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    def _get_both_loss(self, src_tokens, tgt_tokens, tgt_labels):
        seq2seq_output, seqlab_output = self._bart.forward_long(
            src_list=src_tokens,
            prev_output_tokens=tgt_tokens,
            aspect=True
        )

        tgt_tokens = tgt_tokens.to(seq2seq_output.device)
        tgt_labels = tgt_labels.to(seq2seq_output.device)

        # Shift so that tokens < n predict n
        seq2seq_shift_logits = seq2seq_output[:, :-1].contiguous()
        seq2seq_target = tgt_tokens[:, 1:].contiguous()

        seqlab_shift_logits = seqlab_output[:, :-1].contiguous()
        seqlab_target = tgt_labels[:, 1:].contiguous()

        # Flatten the tokens
        criterion_seq2seq = torch.nn.CrossEntropyLoss(
            ignore_index=self._bart.config.pad_token_id
        )
        criterion_seqlab = torch.nn.CrossEntropyLoss(
            ignore_index=self._bart.pad_label_index
        )

        seq2seq_loss = criterion_seq2seq(
            seq2seq_shift_logits.view(-1, seq2seq_shift_logits.size(-1)), seq2seq_target.view(-1)
        )

        seqlab_loss = criterion_seqlab(
            seqlab_shift_logits.view(-1, seqlab_shift_logits.size(-1)), seqlab_target.view(-1)
        )

        return seq2seq_loss, seqlab_loss

    @property
    def dataset(self):
        return self._dataset

    @property
    def train_dataset(self):
        return self._dataset['train']

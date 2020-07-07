import pickle
from collections import namedtuple
import random
from tqdm import tqdm, trange
import os

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from .bart_utils import BARTMultiGPUWrapper

TextPairData = namedtuple('TextPairData', [
    'src_text', 'tgt_text', 'src_tokens', 'tgt_tokens'
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

    def load_data(self, set_type, src_texts, tgt_texts):
        assert len(src_texts) == len(tgt_texts)

        self._dataset[set_type] = []
        for src_text, tgt_text in tqdm(zip(src_texts, tgt_texts),
                                       total=len(src_texts),
                                       desc=f'loading {set_type} data'):
            src_tokens = self._bart.encode(src_text, self._src_max_length)
            tgt_tokens = self._bart.encode(tgt_text, self._tgt_max_length)

            self._dataset[set_type].append(TextPairData(
                src_text=src_text,
                tgt_text=tgt_text,
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens
            ))

        print(f'#{set_type}: {len(self._dataset[set_type])}')

    def train_epoch(self, batch_size):
        assert 'train' in self._dataset

        random.shuffle(self._dataset['train'])
        for i in trange(0, len(self._dataset['train']), batch_size,
                        desc='BART Training'):
            self._bart.set_mode('train')
            self._bart.train()

            batch = self._dataset['train'][i:i + batch_size]

            self._optimizer.zero_grad()

            # we access the data one by one
            for j in range(0, len(batch)):
                data = batch[j]
                src_tokens = data.src_tokens.unsqueeze(0)
                tgt_tokens = data.tgt_tokens.unsqueeze(0)

                loss = self._get_seq2seq_loss(
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens
                )

                loss = loss / batch_size
                loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

    def evaluate(self):
        assert 'val' in self._dataset
        self._bart.set_mode('train')
        self._bart.eval()

        loss_list = []
        for i in range(0, len(self._dataset['val'])):
            data = self._dataset['val'][i]

            src_tokens = data.src_tokens.unsqueeze(0)
            tgt_tokens = data.tgt_tokens.unsqueeze(0)

            with torch.no_grad():
                loss = self._get_seq2seq_loss(
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens
                )

            loss_list.append(loss.item())

        return sum(loss_list) / len(loss_list)

    def generate(self):
        pass

    def _get_seq2seq_loss(self, src_tokens, tgt_tokens):
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

    @property
    def dataset(self):
        return self._dataset

    @property
    def train_dataset(self):
        return self._dataset['train']

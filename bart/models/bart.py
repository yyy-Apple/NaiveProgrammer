from collections import namedtuple
import random
from tqdm import tqdm, trange
import os
import nltk

import torch

from fairseq.data.data_utils import collate_tokens
# collate_tokens just convert a list of 1d tensors into a padded 2d tensor.
from fairseq.sequence_generator import SequenceGenerator

from transformers import AdamW, get_linear_schedule_with_warmup
# AdamW: """ Implements Adam algorithm with weight decay fix.
# get_linear_schedule_with_warmup: lr first goes up and then goes down


from .bart_utils import BARTModelWrapper
from fairseq.models.bart import hub_interface

LIL_BATCH_SIZE = 1


TextPairData = namedtuple('TextPairData', [
    'src_text', 'tgt_text', 'src_tokens', 'tgt_tokens'])


class BART:
    def __init__(self, device, src_max_length, tgt_max_length):
        self._device = device

        self._src_max_length = src_max_length
        self._tgt_max_length = tgt_max_length

        self._bart = BARTModelWrapper(device=device)

        self._optimizer = None
        self._lr_scheduler = None
        self._global_step = 0

        self._dataset = {}

        self._log_dir = None
        self._eval_steps = None
        self._log_file = None
        self._best_dev_loss = None

    def create_training_log(self, eval_steps, label):
        self._log_dir = f'{label}_training_logs'
        self._eval_steps = eval_steps
        self._best_dev_loss = float('inf')

        os.makedirs(os.path.join(self._log_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self._log_dir, 'generations'), exist_ok=True)
        self._log_file = open(os.path.join(self._log_dir, 'log.txt'), 'w')

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
            src_tokens = self._bart.encode(
                src_text, max_length=self._src_max_length)
            tgt_tokens = self._bart.encode(
                tgt_text, max_length=self._tgt_max_length)

            self._dataset[set_type].append(TextPairData(
                src_text=src_text,
                tgt_text=tgt_text,
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens))

        print(f'#{set_type}: {len(self._dataset[set_type])}')

    def train_epoch(self, batch_size):
        assert 'train' in self._dataset

        random.shuffle(self._dataset['train'])
        for i in trange(0, len(self._dataset['train']), batch_size,
                        desc='BART Training'):
            self._bart.set_mode('train')
            self._bart.train()

            # self._dataset['train'] is a list of tuple
            batch = self._dataset['train'][i:i + batch_size]

            self._optimizer.zero_grad()

            for j in range(0, len(batch), LIL_BATCH_SIZE):
                lil_batch = batch[j:j + LIL_BATCH_SIZE]

                src_lengths = torch.tensor(
                    [len(t.src_tokens) for t in lil_batch])
                src_tokens = collate_tokens(
                    [t.src_tokens for t in lil_batch],
                    pad_idx=self._bart.dictionary.pad())
                tgt_tokens = collate_tokens(
                    [t.tgt_tokens for t in lil_batch],
                    pad_idx=self._bart.dictionary.pad())

                loss = self._get_seq2seq_loss(
                    src_lengths=src_lengths,
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens)
                loss = loss * len(lil_batch) / batch_size
                loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

            # self._global_step += 1
            # if self._global_step % self._eval_steps == 0:
            #     self.gen_log()

    def evaluate(self):
        assert 'val' in self._dataset
        self._bart.set_mode('train')
        self._bart.eval()

        loss_list = []
        for i in range(0, len(self._dataset['val']), LIL_BATCH_SIZE):
            batch = self._dataset['val'][i:i + LIL_BATCH_SIZE]

            src_lengths = torch.tensor(
                [len(t.src_tokens) for t in batch])
            src_tokens = collate_tokens(
                [t.src_tokens for t in batch],
                pad_idx=self._bart.dictionary.pad())
            tgt_tokens = collate_tokens(
                [t.tgt_tokens for t in batch],
                pad_idx=self._bart.dictionary.pad())

            with torch.no_grad():
                loss = self._get_seq2seq_loss(
                    src_lengths=src_lengths,
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens)

            loss_list.append(loss.item())

        return sum(loss_list) / len(loss_list)

    def generate(self, src_sents, beam=4, lenpen=2.0, max_len_b=140,
                 min_len=55, no_repeat_ngram_size=3):
        self._bart.set_mode('infer')
        self._bart.eval()

        return self._bart.sample(
            src_sents,
            beam=beam,
            lenpen=lenpen,
            max_len_b=max_len_b,
            min_len=min_len,
            no_repeat_ngram_size=no_repeat_ngram_size)

    def gen_log(self):
        eval_loss = self.evaluate()

        print(f'Global Step: {self._global_step}, Eval Loss: {eval_loss}',
              file=self._log_file)

        if eval_loss < self._best_dev_loss:
            self._best_dev_loss = eval_loss
            self.save_model(f'{self._log_dir}/models/best_model.pt')
            print('Best Model Updated.', file=self._log_file)

        self._log_file.flush()

    def _get_seq2seq_loss(self, src_lengths, src_tokens,  tgt_tokens):
        logits, extra = self._bart(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            prev_output_tokens=tgt_tokens)

        tgt_tokens = tgt_tokens.to(logits.device)

        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = tgt_tokens[:, 1:].contiguous()

        # Flatten the tokens
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self._bart.dictionary.pad())
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    def get_test_nll(self):
        assert 'test' in self._dataset
        self._bart.set_mode('infer')
        self._bart.eval()

        all_nll, n_words = [], 0
        for i in trange(0, len(self._dataset['test']),
                        desc='Getting BART Test NLL'):
            batch = self._dataset['test'][i:i + 1]

            src_lengths = torch.tensor(
                [len(t.src_tokens) for t in batch])
            src_tokens = collate_tokens(
                [t.src_tokens for t in batch],
                pad_idx=self._bart.dictionary.pad())
            tgt_tokens = collate_tokens(
                [t.tgt_tokens for t in batch],
                pad_idx=self._bart.dictionary.pad())

            text = self._bart.decode(tgt_tokens[0])
            n_words += len(nltk.word_tokenize(text))

            with torch.no_grad():
                logits, extra = self._bart(
                    src_tokens=src_tokens,
                    src_lengths=src_lengths,
                    prev_output_tokens=tgt_tokens)

                tgt_tokens = tgt_tokens.to(logits.device)

                # Shift so that tokens < n predict n
                shift_logits = logits[0, :-1].contiguous()
                shift_labels = tgt_tokens[0, 1:].contiguous()

                # Flatten the tokens
                criterion = torch.nn.CrossEntropyLoss(
                    ignore_index=self._bart.dictionary.pad(), reduction='none')
                nll = criterion(shift_logits, shift_labels)

                all_nll.extend(nll.tolist())

        return all_nll, n_words

    @property
    def train_dataset(self):
        return self._dataset['train']

    @property
    def dataset(self):
        return self._dataset

    @property
    def get_lr(self):
        return self._lr_scheduler.get_lr()

import pickle
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
from typing import List

from .multigpu_utils import BARTModelMultiGPUWrapper
from fairseq.models.bart import hub_interface

LIL_BATCH_SIZE = 1

TextPairData = namedtuple('TextPairData', [
    'src_text', 'tgt_text', 'src_tokens', 'tgt_tokens'])

TextLabelData = namedtuple('TextLabelData', [
    'src_text', 'tgt_text', 'src_tokens', 'tgt_tokens', 'tgt_labels'
])


class BART:
    def __init__(self, device, src_max_length, tgt_max_length):
        self._device = device

        self._src_max_length = src_max_length
        self._tgt_max_length = tgt_max_length

        self._bart = BARTModelMultiGPUWrapper(device=device)

        self._optimizer = None
        self._lr_scheduler = None
        self._global_step = 0

        self._dataset = {}
        self._aspect_dataset = {}
        self._long_dataset = {}
        self._long_aspect_dataset = {}

        self._log_dir = None
        self._eval_steps = None
        self._log_file = None
        self._best_dev_loss = None

        self._id_to_label = None
        self._label_to_id = None

    def create_training_log(self, eval_steps, label):
        self._log_dir = f'{label}_training_logs'
        self._eval_steps = eval_steps
        self._best_dev_loss = float('inf')

        os.makedirs(os.path.join(self._log_dir, 'distilBart'), exist_ok=True)
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

    def load_long_data(self, set_type, src_texts, tgt_texts):
        if os.path.exists(f'long_data_{set_type}.pkl'):
            with open(f'long_data_{set_type}.pkl', 'rb') as f:
                self._long_dataset[set_type] = pickle.load(f)
                print(f'Loading {set_type} data from long_data_{set_type}.pkl')
                print(f'#{set_type}: {len(self._long_dataset[set_type])}')
            return

        assert len(src_texts) == len(tgt_texts)

        self._long_dataset[set_type] = []
        for src_text, tgt_text in tqdm(zip(src_texts, tgt_texts),
                                       total=len(src_texts),
                                       desc=f'loading {set_type} data'):
            src_token_list = self._bart.encode_long(src_text)
            tgt_tokens = self._bart.encode(tgt_text,
                                           max_length=self._tgt_max_length)

            self._long_dataset[set_type].append(TextPairData(
                src_text=src_text,
                tgt_text=tgt_text,
                src_tokens=src_token_list,
                tgt_tokens=tgt_tokens
            ))
        print(f'#{set_type}: {len(self._long_dataset[set_type])}')

        with open(f'long_data_{set_type}.pkl', 'wb') as f:
            pickle.dump(self._long_dataset[set_type], f)

    def construct_dict(self, labels: List[str]):
        self._id_to_label = {i: label for i, label in enumerate(labels)}
        self._label_to_id = {label: i for i, label in enumerate(labels)}

    def load_aspect_data(self, set_type, src_texts, tgt_texts, tgt_words, tgt_labels):
        self._aspect_dataset[set_type] = []
        for src_text, tgt_text, tgt_word_list, tgt_label_list in tqdm(zip(src_texts,
                                                                          tgt_texts,
                                                                          tgt_words,
                                                                          tgt_labels),
                                                                      total=len(src_texts),
                                                                      desc=f'loading {set_type} data'):
            src_tokens = self._bart.encode(
                src_text, max_length=self._src_max_length
            )

            tgt_tokens, tgt_labels = self._bart.encode_target(
                tgt_word_list=tgt_word_list,
                tgt_label_list=tgt_label_list,
                max_length=self._tgt_max_length,
                label_to_id=self._label_to_id
            )

            self._aspect_dataset[set_type].append(TextLabelData(
                src_text=src_text,
                tgt_text=tgt_text,
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens,
                tgt_labels=tgt_labels
            ))

        print(f'#{set_type}: {len(self._aspect_dataset[set_type])}')

    def load_long_aspect_data(self, set_type, src_texts, tgt_texts, tgt_words, tgt_labels):
        if os.path.exists(f'long_aspect_data_{set_type}.pkl'):
            with open(f'long_aspect_data_{set_type}.pkl', 'rb') as f:
                self._long_aspect_dataset[set_type] = pickle.load(f)
                print(f'Loading {set_type} data from long_aspect_data_{set_type}.pkl')
                print(f'#{set_type}: {len(self._long_aspect_dataset[set_type])}')
            return

        self._long_aspect_dataset[set_type] = []
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

            self._long_aspect_dataset[set_type].append(TextLabelData(
                src_text=src_text,
                tgt_text=tgt_text,
                src_tokens=src_token_list,
                tgt_tokens=tgt_tokens,
                tgt_labels=tgt_labels
            ))

        print(f'#{set_type}: {len(self._long_aspect_dataset[set_type])}')

        with open(f'long_aspect_data_{set_type}.pkl', 'wb') as f:
            pickle.dump(self._long_aspect_dataset[set_type], f)

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

    def train_long_epoch(self, batch_size):
        assert 'train' in self._long_dataset

        random.shuffle(self._long_dataset['train'])
        for i in trange(0, len(self._long_dataset['train']), batch_size,
                        desc='BART Training'):
            self._bart.set_mode('train')
            self._bart.train()

            batch = self._long_dataset['train'][i:i + batch_size]

            self._optimizer.zero_grad()

            # for full text, we process each training data individually
            for j in range(len(batch)):
                # batch[j] is a data sample
                src_lengths = torch.tensor(
                    [len(t) for t in batch[j].src_tokens]
                )
                src_tokens = batch[j].src_tokens
                tgt_tokens = batch[j].tgt_tokens.unsqueeze(0)

                loss = self._get_seq2seq_loss(
                    src_lengths=src_lengths,
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens,
                    long=True
                )

                loss = loss / batch_size
                loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

    def train_aspect_epoch(self, batch_size):
        assert 'train' in self._aspect_dataset

        random.shuffle(self._aspect_dataset['train'])
        for i in trange(0, len(self._aspect_dataset['train']), batch_size,
                        desc='BART Training'):
            self._bart.set_mode('train')
            self._bart.train()

            batch = self._aspect_dataset['train'][i:i + batch_size]

            self._optimizer.zero_grad()

            for j in range(0, len(batch), LIL_BATCH_SIZE):
                lil_batch = batch[j:j + LIL_BATCH_SIZE]

                src_lengths = torch.tensor(
                    [len(t.src_tokens) for t in lil_batch]
                )
                src_tokens = collate_tokens(
                    [t.src_tokens for t in lil_batch],
                    pad_idx=self._bart.dictionary.pad()
                )
                tgt_tokens = collate_tokens(
                    [t.tgt_tokens for t in lil_batch],
                    pad_idx=self._bart.dictionary.pad()
                )

                tgt_labels = collate_tokens(
                    [t.tgt_labels for t in lil_batch],
                    pad_idx=self._bart.pad_label_index
                )

                seq2seq_loss, seqlab_loss = self._get_both_loss(
                    src_lengths=src_lengths,
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens,
                    tgt_labels=tgt_labels
                )

                loss = seq2seq_loss + seqlab_loss
                loss = loss * len(lil_batch) / batch_size

                loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

    def train_long_aspect_epoch(self, batch_size):
        assert 'train' in self._long_aspect_dataset

        random.shuffle(self._long_aspect_dataset['train'])
        for i in trange(0, len(self._long_aspect_dataset['train']), batch_size,
                        desc='BART Training'):
            self._bart.set_mode('train')
            self._bart.train()

            batch = self._long_aspect_dataset['train'][i:i + batch_size]

            self._optimizer.zero_grad()

            # for full text, we process each training data individually
            for j in range(len(batch)):
                src_lengths = torch.tensor(
                    [len(t) for t in batch[j].src_tokens]
                )
                src_token_list = batch[j].src_tokens
                tgt_tokens = batch[j].tgt_tokens.unsqueeze(0)

                tgt_labels = batch[j].tgt_labels.unsqueeze(0)

                seq2seq_loss, seqlab_loss = self._get_both_loss(
                    src_lengths=src_lengths,
                    src_tokens=src_token_list,
                    tgt_tokens=tgt_tokens,
                    tgt_labels=tgt_labels,
                    long=True
                )

                loss = seq2seq_loss + seqlab_loss
                loss = loss / batch_size

                loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

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

    def evaluate_long(self):
        assert 'val' in self._long_dataset
        self._bart.set_mode('train')
        self._bart.eval()

        loss_list = []
        for i in range(0, len(self._long_dataset['val'])):
            data = self._long_dataset['val'][i]
            src_lengths = torch.tensor(
                [len(t) for t in data.src_tokens]
            )
            src_tokens = data.src_tokens
            tgt_tokens = data.tgt_tokens.unsqueeze(0)

            with torch.no_grad():
                loss = self._get_seq2seq_loss(
                    src_lengths=src_lengths,
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens,
                    long=True
                )
            loss_list.append(loss.item())

        return sum(loss_list) / len(loss_list)

    def evaluate_aspect(self):
        assert 'val' in self._aspect_dataset
        self._bart.set_mode('train')
        self._bart.eval()

        loss_list = []
        for i in range(0, len(self._aspect_dataset['val']), LIL_BATCH_SIZE):
            batch = self._aspect_dataset['val'][i:i + LIL_BATCH_SIZE]

            src_lengths = torch.tensor(
                [len(t.src_tokens) for t in batch]
            )
            src_tokens = collate_tokens(
                [t.src_tokens for t in batch],
                pad_idx=self._bart.dictionary.pad()
            )
            tgt_tokens = collate_tokens(
                [t.tgt_tokens for t in batch],
                pad_idx=self._bart.dictionary.pad()
            )

            tgt_labels = collate_tokens(
                [t.tgt_labels for t in batch],
                pad_idx=self._bart.pad_label_index
            )

            with torch.no_grad():
                seq2seq_loss, seqlab_loss = self._get_both_loss(
                    src_lengths=src_lengths,
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens,
                    tgt_labels=tgt_labels
                )
                loss = seq2seq_loss + seqlab_loss

            loss_list.append(loss.item())

        return sum(loss_list) / len(loss_list)

    def evaluate_long_aspect(self):
        assert 'val' in self._long_aspect_dataset
        self._bart.set_mode('train')
        self._bart.eval()

        loss_list = []
        for i in range(0, len(self._long_aspect_dataset['val'])):
            data = self._long_aspect_dataset['val'][i]
            src_lengths = torch.tensor(
                [len(t) for t in data.src_tokens]
            )
            src_tokens = data.src_tokens
            tgt_tokens = data.tgt_tokens.unsqueeze(0)
            tgt_labels = data.tgt_labels.unsqueeze(0)

            with torch.no_grad():
                seq2seq_loss, seqlab_loss = self._get_both_loss(
                    src_lengths=src_lengths,
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens,
                    tgt_labels=tgt_labels,
                    long=True
                )
                loss = seq2seq_loss + seqlab_loss

            loss_list.append(loss.item())

        return sum(loss_list) / len(loss_list)

    def generate(self, src_sents: List[str], beam=4, lenpen=2.0, max_len_b=20,
                 min_len=10, no_repeat_ngram_size=3):
        self._bart.set_mode('infer')
        self._bart.eval()

        return self._bart.sample(
            src_sents,
            beam=beam,
            lenpen=lenpen,
            max_len_b=max_len_b,
            min_len=min_len,
            no_repeat_ngram_size=no_repeat_ngram_size)

    def long_input_generate(self, src_sents: str, beam=4, lenpen=2.0, max_len_b=20,
                            min_len=10, no_repeat_ngram_size=3):
        # generate one sentence at a time
        self._bart.set_mode('infer')
        self._bart.eval()

        return self._bart.long_input_sample(
            src_sents=src_sents,
            beam=beam,
            lenpen=lenpen,
            max_len_b=max_len_b,
            min_len=min_len,
            no_repeat_ngram_size=no_repeat_ngram_size
        )

    def gen_log(self):
        eval_loss = self.evaluate()

        print(f'Global Step: {self._global_step}, Eval Loss: {eval_loss}',
              file=self._log_file)

        if eval_loss < self._best_dev_loss:
            self._best_dev_loss = eval_loss
            self.save_model(f'{self._log_dir}/distilBart/best_model.pt')
            print('Best Model Updated.', file=self._log_file)

        self._log_file.flush()

    def _get_seq2seq_loss(self, src_lengths, src_tokens, tgt_tokens, long=False):
        if long:
            logits, extra = self._bart.forward_long(
                src_token_list=src_tokens,
                src_lengths=src_lengths,
                prev_output_tokens=tgt_tokens
            )
        else:
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

    def _get_both_loss(self, src_lengths, src_tokens, tgt_tokens,
                       tgt_labels, long=False):
        if long:
            seq2seq_output, seqlab_output = self._bart.forward_long_double_output(
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                prev_output_tokens=tgt_tokens
            )
        else:
            seq2seq_output, seqlab_output = self._bart.forward_double_output(
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                prev_output_tokens=tgt_tokens
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
            ignore_index=self._bart.dictionary.pad()
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
    def train_long_dataset(self):
        return self._long_dataset['train']

    @property
    def train_aspect_dataset(self):
        return self._aspect_dataset['train']

    @property
    def train_long_aspect_dataset(self):
        return self._long_aspect_dataset['train']

    @property
    def dataset(self):
        return self._dataset

    @property
    def long_dataset(self):
        return self._long_dataset

    @property
    def aspect_dataset(self):
        return self._aspect_dataset

    @property
    def long_aspect_dataset(self):
        return self._long_aspect_dataset

    @property
    def get_lr(self):
        return self._lr_scheduler.get_lr()

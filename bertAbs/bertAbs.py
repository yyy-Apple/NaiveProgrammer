from collections import namedtuple
import random
import torch
import torch.nn as nn
from fairseq.data.data_utils import collate_tokens
from transformers import AutoModel, AutoTokenizer, AdamW, BertTokenizer
from models.modeling_bertabs import BertAbs, BertSumOptimizer, build_predictor
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup

TextPairData = namedtuple('TextPairData', [
    'src_text', 'tgt_text', 'src_tokens', 'tgt_tokens'])

LIL_BATCH_SIZE = 1


class BERTABS:
    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.bertAbs = BertAbs.from_pretrained('bertabs-finetuned-cnndm')
        self.bertAbs.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.bertAbs.decoder.embeddings = self.bertAbs.bert.embeddings.word_embeddings
        self.bertAbs.generator[0].weight = self.bertAbs.decoder.embeddings.weight
        self.bertAbs.generator[0].bias = nn.Parameter(torch.randn(31090))

        self._optimizer = None
        self._lr_scheduler = None

        self._best_dev_loss = None

        self._dataset = {}

        self._device = device

    def get_optimizer(self, lr, train_steps, warmup_steps,
                      weight_decay, adam_epsilon):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.bertAbs.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self.bertAbs.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        self._optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
        self._lr_scheduler = get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=train_steps)
        lr = {'encoder': 0.002, 'decoder': 0.2}

    def load_data(self, set_type, src_texts, tgt_texts):
        assert len(src_texts) == len(tgt_texts)

        self._dataset[set_type] = []
        for src_text, tgt_text in tqdm(zip(src_texts, tgt_texts),
                                       total=len(src_texts),
                                       desc=f'loading {set_type} data'):
            src_tokens = torch.tensor(self.tokenizer.encode(src_text)[: 510])
            tgt_tokens = torch.tensor(self.tokenizer.encode(tgt_text)[: 510])

            self._dataset[set_type].append(TextPairData(
                src_text=src_text,
                tgt_text=tgt_text,
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens))

        print(f'#{set_type}: {len(self._dataset[set_type])}')

    def save_model(self, path):
        torch.save(self.bertAbs.state_dict(), path)
        print(f'Model saved in {path}.')

    def load_model(self, path):
        self.bertAbs.load_state_dict(torch.load(path, map_location=self._device))
        print(f'Model {path} loaded.')

    def _get_seq2seq_loss(self, src_lengths, src_tokens, tgt_tokens):
        raw_out = self.bertAbs(src_tokens.to(self._device),
                               tgt_tokens.to(self._device),
                               None, None, None)  # B x Sequence_len x 768
        logits = self.bertAbs.generator(raw_out)

        tgt_tokens = tgt_tokens.to(logits.device)

        # Shift so that tokens < n predict n
        shift_logits = logits.contiguous()
        shift_labels = tgt_tokens[:, 1:].contiguous()

        # Flatten the tokens
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id)
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    def train_epoch(self, batch_size):
        assert 'train' in self._dataset

        random.shuffle(self._dataset['train'])
        for i in trange(0, len(self._dataset['train']), batch_size,
                        desc='BERTABS Training'):

            self.bertAbs = self.bertAbs.to(self._device)
            self.bertAbs.train()

            # self._dataset['train'] is a list of tuple
            batch = self._dataset['train'][i:i + batch_size]

            self._optimizer.zero_grad()

            for j in range(0, len(batch)):
                lil_batch = batch[j:j + LIL_BATCH_SIZE]

                src_lengths = torch.tensor(
                    [len(t.src_tokens) for t in lil_batch])
                src_tokens = collate_tokens(
                    [t.src_tokens for t in lil_batch],
                    pad_idx=self.tokenizer.pad_token_id)
                tgt_tokens = collate_tokens(
                    [t.tgt_tokens for t in lil_batch],
                    pad_idx=self.tokenizer.pad_token_id)

                loss = self._get_seq2seq_loss(
                    src_lengths=src_lengths,
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens)
                loss = loss * len(lil_batch) / batch_size
                loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

    def evaluate(self):
        assert 'val' in self._dataset
        self.bertAbs.eval()

        loss_list = []
        for i in range(0, len(self._dataset['val']), LIL_BATCH_SIZE):
            batch = self._dataset['val'][i:i + LIL_BATCH_SIZE]

            src_lengths = torch.tensor(
                [len(t.src_tokens) for t in batch])
            src_tokens = collate_tokens(
                [t.src_tokens for t in batch],
                pad_idx=self.tokenizer.pad_token_id)
            tgt_tokens = collate_tokens(
                [t.tgt_tokens for t in batch],
                pad_idx=self.tokenizer.pad_token_id)

            with torch.no_grad():
                loss = self._get_seq2seq_loss(
                    src_lengths=src_lengths,
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens)

            loss_list.append(loss.item())

        return sum(loss_list) / len(loss_list)

    @property
    def train_dataset(self):
        return self._dataset['train']

    @property
    def dataset(self):
        return self._dataset

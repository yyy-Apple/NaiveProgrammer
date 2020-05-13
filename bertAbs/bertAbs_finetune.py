import fire
import torch
from bertAbs import BERTABS
import os
import sys

BATCH_SIZE = 32
LR = 4e-5
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.
WARMUP_PROPORTION = 0.1

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(split):
    src_file = open(f'toy/full_text.{split}')
    tgt_file = open(f'toy/review.{split}')
    src_texts, tgt_texts = [], []
    for src, tgt in zip(src_file.readlines(), tgt_file.readlines()):
        src_texts.append(src.strip())
        tgt_texts.append(tgt.strip())
    return src_texts, tgt_texts


def main(n_epochs=1):
    bertAbs = BERTABS(device=device)
    print(bertAbs.bertAbs.bert.embeddings.word_embeddings)
    print(bertAbs.bertAbs.decoder.embeddings)
    for split in ['train', 'val']:
        src_texts, tgt_texts = load_data(split)
        bertAbs.load_data(
            set_type=split,
            src_texts=src_texts,
            tgt_texts=tgt_texts)

    train_steps = n_epochs * (len(bertAbs.train_dataset) // BATCH_SIZE + 1)
    warmup_steps = int(train_steps * WARMUP_PROPORTION)
    bertAbs.get_optimizer(
        lr=LR,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILON)

    best_loss = 1e9

    for epoch in range(n_epochs):
        print(f"On epoch {epoch}")
        bertAbs.train_epoch(batch_size=BATCH_SIZE)
        current_loss = bertAbs.evaluate()
        if current_loss < best_loss:
            best_loss = current_loss
            bertAbs.save_model('bertAbs_best.pth')


if __name__ == '__main__':
    fire.Fire(main)


import fire

from bertAbs import BERTABS


BATCH_SIZE = 5
LR = 4e-5
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.
WARMUP_PROPORTION = 0.1


def load_data(split):
    src_file = open(f'toy/full_text.{split}')
    tgt_file = open(f'toy/review.{split}')
    src_texts, tgt_texts = [], []
    for src, tgt in zip(src_file.readlines(), tgt_file.readlines()):
        src_texts.append(src.strip())
        tgt_texts.append(tgt.strip())
    return src_texts, tgt_texts


def main(n_epochs=5):
    bertAbs = BERTABS()

    for split in ['train', 'val']:
        src_texts, tgt_texts = load_data(split)
        bertAbs.load_data(
            set_type=split,
            src_texts=src_texts,
            tgt_texts=tgt_texts)

    train_steps = n_epochs * (len(bertAbs.train_dataset) // BATCH_SIZE + 1)
    warmup = int(train_steps * WARMUP_PROPORTION)
    warmup_steps = {'encoder': warmup, 'decoder': warmup}
    bertAbs.get_optimizer(train_steps, warmup_steps)

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


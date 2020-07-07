import fire
import os
from tqdm import trange

from distilBart.distilbart import BART

import logging
logging.disable(logging.WARNING)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    bart = BART(device='cpu', src_max_length=1024, tgt_max_length=1024)
    bart.load_model('bart_best.pth')
    test_src_file = open('data/fulltext.test')

    test_hypo_file = open('data/test.hypo', 'w')

    src_sents = [line.strip() for line in test_src_file.readlines()]
    batch_size = 8
    for i in trange(0, len(src_sents), batch_size):
        hypos = bart.generate(src_sents[i: i + batch_size])
        for hypo in hypos:
            print(hypo, file=test_hypo_file)
        test_hypo_file.flush()


if __name__ == '__main__':
    fire.Fire(main)

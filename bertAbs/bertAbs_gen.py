import fire
from tqdm import trange
import torch
from collections import namedtuple
from argparse import Namespace
from models.modeling_bertabs import build_predictor
from bertAbs import BERTABS

Batch = namedtuple("Batch", ["document_names", "batch_size", "src", "segs", "mask_src", "tgt_str"])

args = Namespace(
    documents_dir='toy/full_text.test',
    summaries_output_dir='toy/hypo.test',
    compute_rouge=False,
    no_cuda=True,
    batch_size=1,
    min_length=50,
    max_length=200,
    beam_size=5,
    alpha=0.95,
    block_trigram=True
)


def get_batch(text, tokenizer):
    batch = Batch(
        document_names="doc",
        batch_size=1,
        src=torch.tensor(tokenizer.encode(text)[:510]).unsqueeze(dim=0),
        segs=None,
        mask_src=None,
        tgt_str="tgt",
    )
    return batch


def format_summary(translation):
    """ Transforms the output of the `from_batch` function
    into nicely formatted summaries.
    """
    raw_summary, _, _ = translation
    summary = (
        raw_summary.replace("[unused0]", "")
            .replace("[unused3]", "")
            .replace("[PAD]", "")
            .replace("[unused1]", "")
            .replace(r" +", " ")
            .replace(" [unused2] ", ". ")
            .replace("[unused2]", "")
            .strip()
    )

    return summary


def main():
    bertAbs = BERTABS()
    bertAbs.load_model('bart_best.pth')

    symbols = {
        "BOS": bertAbs.tokenizer.vocab["[unused0]"],
        "EOS": bertAbs.tokenizer.vocab["[unused1]"],
        "PAD": bertAbs.tokenizer.vocab["[PAD]"],
    }

    predictor = build_predictor(args, bertAbs.tokenizer, symbols, bertAbs.bertAbs)

    test_src_file = open('toy/full_text.test')

    test_hypo_file = open('toy/test.hypo', 'w')

    src_sents = [line.strip() for line in test_src_file.readlines()]
    hypos = []
    for i in trange(0, len(src_sents)):
        hypo = predictor.translate(get_batch(src_sents[i], bertAbs.tokenizer))
        hypo = format_summary(hypo[0])
        hypos.append(hypo)

    for hypo in hypos:
        print(hypo, file=test_hypo_file)
    test_hypo_file.flush()


if __name__ == '__main__':
    fire.Fire(main)

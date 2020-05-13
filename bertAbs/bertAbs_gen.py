#! /usr/bin/python3
import argparse
from argparse import Namespace
import logging
import os
import sys
from collections import namedtuple

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from models.modeling_bertabs import BertAbs, build_predictor
from transformers import BertTokenizer, AutoTokenizer
from bertAbs import BERTABS
from models.utils_summarization import (
    CNNDMDataset,
    build_mask,
    compute_token_type_ids,
    encode_for_summarization,
    truncate_or_pad,
)

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

Batch = namedtuple("Batch", ["document_names", "batch_size", "src", "segs", "mask_src", "tgt_str"])


def evaluate(args):
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    # model = BertAbs.from_pretrained("bertabs-finetuned-cnndm")
    bertabs = BERTABS(device='cpu')
    bertabs.load_model('bertAbs_best.pth')
    model = bertabs.bertAbs
    model.to(args.device)
    model.eval()

    symbols = {
        "BOS": tokenizer.vocab["[unused0]"],
        "EOS": tokenizer.vocab["[unused1]"],
        "PAD": tokenizer.vocab["[PAD]"],
    }

    # these (unused) arguments are defined to keep the compatibility
    # with the legacy code and will be deleted in a next iteration.
    args.result_path = ""
    args.temp_dir = ""

    data_iterator = build_data_iterator(args, tokenizer)
    predictor = build_predictor(args, tokenizer, symbols, model)

    logger.info("***** Running evaluation *****")
    logger.info("  Number examples = %d", len(data_iterator.dataset))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("")
    logger.info("***** Beam Search parameters *****")
    logger.info("  Beam size = %d", args.beam_size)
    logger.info("  Minimum length = %d", args.min_length)
    logger.info("  Maximum length = %d", args.max_length)
    logger.info("  Alpha (length penalty) = %.2f", args.alpha)
    logger.info("  Trigrams %s be blocked", ("will" if args.block_trigram else "will NOT"))

    for batch in tqdm(data_iterator):
        batch_data = predictor.translate_batch(batch)
        translations = predictor.from_batch(batch_data)
        summaries = [format_summary(t) for t in translations]
        save_summaries(summaries, args.summaries_output_dir, batch.document_names)


def save_summaries(summaries, path, original_document_name):
    """ Write the summaries in fies that are prefixed by the original
    files' name with the `_summary` appended.

    Attributes:
        original_document_names: List[string]
            Name of the document that was summarized.
        path: string
            Path were the summaries will be written
        summaries: List[string]
            The summaries that we produced.
    """
    summary_file = open(path, 'a', encoding='utf8')
    for summary in summaries:
        print(summary, file=summary_file)
    summary_file.flush()
    # for summary, document_name in zip(summaries, original_document_name):
    #     # Prepare the summary file's name
    #     if "." in document_name:
    #         bare_document_name = ".".join(document_name.split(".")[:-1])
    #         extension = document_name.split(".")[-1]
    #         name = bare_document_name + "_summary." + extension
    #     else:
    #         name = document_name + "_summary"
    #
    #     file_path = os.path.join(path, name)
    #     with open(file_path, "w") as output:
    #         output.write(summary)


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


def save_rouge_scores(str_scores):
    with open("rouge_scores.txt", "w") as output:
        output.write(str_scores)


#
# LOAD the dataset
#


def build_data_iterator(args, tokenizer):
    dataset = load_and_cache_examples(args, tokenizer)
    sampler = SequentialSampler(dataset)

    def collate_fn(data):
        return collate(data, tokenizer, block_size=512, device=args.device)

    # collate data
    iterator = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn, )

    return iterator


def load_and_cache_examples(args, tokenizer):
    dataset = CNNDMDataset(args.documents_dir)
    return dataset


def collate(data, tokenizer, block_size, device):
    """ Collate formats the data passed to the data loader.

    In particular we tokenize the data batch after batch to avoid keeping them
    all in memory. We output the data as a namedtuple to fit the original BertAbs's
    API.
    """
    data = [x for x in data if not len(x[1]) == 0]  # remove empty_files
    names = [name for name, _, _ in data]
    summaries = [" ".join(summary_list) for _, _, summary_list in data]

    encoded_text = [encode_for_summarization(story, summary, tokenizer) for _, story, summary in data]
    encoded_stories = torch.tensor(
        [truncate_or_pad(story, block_size, tokenizer.pad_token_id) for story, _ in encoded_text]
    )
    encoder_token_type_ids = compute_token_type_ids(encoded_stories, tokenizer.cls_token_id)
    encoder_mask = build_mask(encoded_stories, tokenizer.pad_token_id)

    batch = Batch(
        document_names=names,
        batch_size=len(encoded_stories),
        src=encoded_stories.to(device),
        segs=encoder_token_type_ids.to(device),
        mask_src=encoder_mask.to(device),
        tgt_str=summaries,
    )

    return batch


def decode_summary(summary_tokens, tokenizer):
    """ Decode the summary and return it in a format
    suitable for evaluation.
    """
    summary_tokens = summary_tokens.to("cpu").numpy()
    summary = tokenizer.decode(summary_tokens)
    sentences = summary.split(".")
    sentences = [s + "." for s in sentences]
    return sentences


def main():
    """ The main function defines the interface with the users.
    """
    args = Namespace(
        documents_dir='./data/test',
        summaries_output_dir='./data/test.hypo',
        compute_rouge=False,
        no_cuda=True,
        batch_size=1,
        min_length=50,
        max_length=200,
        beam_size=5,
        alpha=0.95,
        block_trigram=True
    )

    if os.path.exists(args.summaries_output_dir):
        os.system(f'rm {args.summaries_output_dir}')
        os.system(f'touch {args.summaries_output_dir}')

    # remove the fine .DS_Store on Mac
    os.system(f'rm {args.documents_dir}/.DS_Store')

    # Select device (distibuted not available)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Check the existence of directories
    # if not args.summaries_output_dir:
    #     args.summaries_output_dir = args.documents_dir
    #
    # if not documents_dir_is_valid(args.documents_dir):
    #     raise FileNotFoundError(
    #         "We could not find the directory you specified for the documents to "
    #         "summarize, or it was empty. Please specify a valid path."
    #     )
    # os.makedirs(args.summaries_output_dir, exist_ok=True)

    evaluate(args)


def documents_dir_is_valid(path):
    if not os.path.exists(path):
        return False

    file_list = os.listdir(path)
    if len(file_list) == 0:
        return False

    return True


if __name__ == "__main__":
    main()

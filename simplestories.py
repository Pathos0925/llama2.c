"""
Download, preprocess and serve the SimpleStories dataset as a DataLoader.

SimpleStories (https://huggingface.co/datasets/lennart-finke/SimpleStories) is an
extension of TinyStories with more diverse stories. This module mirrors the
tinystories.py interface so it can be used as a drop-in replacement.

Usage:
    python simplestories.py download
    python simplestories.py pretokenize
    python simplestories.py pretokenize --vocab_size=4096
    python simplestories.py train_vocab --vocab_size=4096
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"


def clean_text(text, lowercase=False):
    """Normalize Unicode to ASCII. Optionally lowercase."""
    if lowercase:
        text = text.lower()
    for src, dst in [
        ("\u2018", "'"), ("\u2019", "'"),  # smart single quotes
        ("\u201c", '"'), ("\u201d", '"'),  # smart double quotes
        ("\u2014", "--"), ("\u2013", "-"),  # em/en dashes
        ("\u2026", "..."),                 # ellipsis
    ]:
        text = text.replace(src, dst)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return text.strip()


def download():
    """Downloads the SimpleStories dataset from HuggingFace and saves as JSON shards."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package is required to download SimpleStories.")
        print("Please run: pip install datasets")
        return

    save_dir = os.path.join(DATA_CACHE_DIR, "SimpleStories")
    if os.path.exists(save_dir) and len(glob.glob(os.path.join(save_dir, "*.json"))) > 0:
        print(f"{save_dir} already has data, skipping download...")
        return

    os.makedirs(save_dir, exist_ok=True)
    print("Downloading SimpleStories from HuggingFace...")
    ds = load_dataset("lennart-finke/SimpleStories")

    # Save train split as JSON shards (~10K stories each)
    train_stories = [{"story": clean_text(ex["story"])} for ex in tqdm(ds["train"], desc="cleaning train")]
    shard_size = 10000
    for i in range(0, len(train_stories), shard_size):
        shard = train_stories[i:i + shard_size]
        fname = os.path.join(save_dir, f"train_{i // shard_size:03d}.json")
        with open(fname, "w") as f:
            json.dump(shard, f)

    # Save test split as a single shard
    test_stories = [{"story": clean_text(ex["story"])} for ex in tqdm(ds["test"], desc="cleaning test")]
    fname = os.path.join(save_dir, "test_000.json")
    with open(fname, "w") as f:
        json.dump(test_stories, f)

    n_train = len(train_stories)
    n_test = len(test_stories)
    n_shards = (n_train + shard_size - 1) // shard_size + 1  # +1 for test
    print(f"Download done. {n_train} train + {n_test} test stories in {n_shards} shards.")
    print(f"Example story:\n{train_stories[0]['story'][:200]}...")


def train_vocab(vocab_size):
    """Trains a custom SentencePiece tokenizer on the SimpleStories dataset."""
    assert vocab_size > 0, "Vocab size must be positive"

    prefix = os.path.join(DATA_CACHE_DIR, f"ss_tok{vocab_size}")
    num_shards = 10

    tiny_file = os.path.join(DATA_CACHE_DIR, "simplestories_vocab.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "SimpleStories")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "train_*.json")))
    assert len(shard_filenames) > 0, f"No train shards found in {data_dir}. Run 'download' first."

    print(f"Writing temporary file {tiny_file} with {min(num_shards, len(shard_filenames))} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                # lowercase for custom tokenizer training so vocab is all lowercase
                of.write(example["story"].lower() + "\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    print("Training the vocab...")
    spm.SentencePieceTrainer.train(
        input=tiny_file,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity",
    )

    dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(tiny_file)

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def process_shard(args, vocab_size):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()
        if vocab_size > 0:
            text = text.lower()  # custom tokenizer was trained on lowercased text
        tokens = enc.encode(text, bos=True, eos=False)
        all_tokens.extend(tokens)
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # save .bin file
    if vocab_size == 0:
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"ss_tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize(vocab_size):
    data_dir = os.path.join(DATA_CACHE_DIR, "SimpleStories")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    assert len(shard_filenames) > 0, f"No JSON shards found in {data_dir}. Run 'download' first."

    if vocab_size > 0:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"ss_tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    fun = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")

        if self.vocab_source == "llama2":
            bin_dir = os.path.join(DATA_CACHE_DIR, "SimpleStories")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif self.vocab_source == "custom":
            bin_dir = os.path.join(DATA_CACHE_DIR, f"ss_tok{self.vocab_size}")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))

        # test split uses shards starting with "test_", train uses "train_"
        if self.split == "train":
            shard_filenames = [s for s in shard_filenames if "train_" in os.path.basename(s)]
        else:
            shard_filenames = [s for s in shard_filenames if "test_" in os.path.basename(s)]
        assert len(shard_filenames) > 0, f"No bin files found for split={self.split}"

        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


# -----------------------------------------------------------------------------
# public interface functions

def get_tokenizer_model_path(vocab_size):
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"ss_tok{vocab_size}.model")


class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


# -----------------------------------------------------------------------------
# CLI

if __name__ == "__main__":
    """
    Usage:
        python simplestories.py download
        python simplestories.py pretokenize
        python simplestories.py pretokenize --vocab_size=4096
        python simplestories.py train_vocab --vocab_size=4096
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab"])
    parser.add_argument("--vocab_size", type=int, default=0,
                        help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    args = parser.parse_args()

    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)

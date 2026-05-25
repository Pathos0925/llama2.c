"""
Download, preprocess and serve a FineWeb subset as a DataLoader.

FineWeb (https://huggingface.co/datasets/HuggingFaceFW/fineweb) is a large-scale
web text dataset. This module downloads the sample-10BT subset, tokenizes it
with the Llama 2 SentencePiece tokenizer, and serves pretokenized shards.

Usage:
    python fineweb.py download
    python fineweb.py pretokenize
"""

import argparse
import glob
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"
FINEWEB_DIR = os.path.join(DATA_CACHE_DIR, "fineweb")
NUM_PROC = 8


def download():
    """Downloads the FineWeb sample-10BT subset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package required.  pip install datasets")
        return

    raw_dir = os.path.join(FINEWEB_DIR, "raw")
    existing = glob.glob(os.path.join(raw_dir, "*.txt"))
    if existing:
        print(f"{raw_dir} already has {len(existing)} shards, skipping download.")
        return

    os.makedirs(raw_dir, exist_ok=True)

    print("Downloading HuggingFaceFW/fineweb (sample-10BT)...")
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train",
                       num_proc=NUM_PROC)
    print(f"Dataset has {len(ds):,} documents.")

    shard_size = 50_000
    n_shards = 0
    for i in range(0, len(ds), shard_size):
        shard = ds[i : i + shard_size]
        fname = os.path.join(raw_dir, f"shard_{n_shards:05d}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            for text in shard["text"]:
                clean = text.strip().replace("\x00", "")
                if clean:
                    f.write(clean + "\n")
        n_shards += 1
    print(f"Saved {n_shards} raw text shards to {raw_dir}/")


def _tokenize_shard(args, vocab_size):
    """Tokenize one raw text shard into a .bin file of uint16 token IDs."""
    shard_id, shard_path = args
    tok_path = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model=tok_path)

    all_tokens = []
    with open(shard_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, position=shard_id % NUM_PROC,
                         desc=os.path.basename(shard_path), leave=False):
            text = line.strip()
            if not text:
                continue
            tokens = enc.encode(text, bos=True, eos=False)
            all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens, dtype=np.uint16)

    if vocab_size == 0 or vocab_size == 32000:
        bin_dir = FINEWEB_DIR
    else:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"fw_tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(shard_path))[0]
    bin_path = os.path.join(bin_dir, f"{base}.bin")
    with open(bin_path, "wb") as f:
        f.write(all_tokens.tobytes())

    print(f"  {bin_path}: {len(all_tokens):,} tokens")


def pretokenize(vocab_size):
    """Tokenize all raw text shards into .bin files."""
    raw_dir = os.path.join(FINEWEB_DIR, "raw")
    shard_files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if not shard_files:
        print(f"No raw shards found in {raw_dir}. Run 'download' first.")
        return

    print(f"Tokenizing {len(shard_files)} shards with vocab_size={vocab_size}...")
    fn = partial(_tokenize_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor(max_workers=NUM_PROC) as executor:
        executor.map(fn, enumerate(shard_files))
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized .bin shards and yields (x, y) pairs."""

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

        if self.vocab_source == "llama2" or self.vocab_size == 32000:
            bin_dir = FINEWEB_DIR
        else:
            bin_dir = os.path.join(DATA_CACHE_DIR, f"fw_tok{self.vocab_size}")

        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        assert len(shard_filenames) > 0, f"No .bin files in {bin_dir}. Run pretokenize first."

        n_val = max(2, len(shard_filenames) // 50)
        if self.split == "val":
            shard_filenames = shard_filenames[-n_val:]
        else:
            shard_filenames = shard_filenames[:-n_val]

        assert len(shard_filenames) > 0, f"No shards for split={self.split}"

        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1
                if num_batches <= 0:
                    continue
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


def get_tokenizer_model_path(vocab_size):
    if vocab_size == 0 or vocab_size == 32000:
        return "tokenizer.model"
    else:
        return os.path.join(DATA_CACHE_DIR, f"fw_tok{vocab_size}.model")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["download", "pretokenize"])
    parser.add_argument("--vocab_size", type=int, default=0,
                        help="0 = use Llama 2 tokenizer (32K vocab)")
    args = parser.parse_args()

    if args.stage == "download":
        download()
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)

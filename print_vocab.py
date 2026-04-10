"""
Print the vocabulary of a SentencePiece tokenizer model.

Usage:
    python print_vocab.py                              # default Llama2 tokenizer
    python print_vocab.py tokenizer.model              # specify .model file
    python print_vocab.py data/ss_tok4096.model        # custom tokenizer
    python print_vocab.py tokenizer.model --top 50     # only first 50 tokens
    python print_vocab.py tokenizer.model --search the # search for tokens containing "the"
"""

import argparse
from sentencepiece import SentencePieceProcessor


def main():
    parser = argparse.ArgumentParser(description="Print SentencePiece tokenizer vocabulary")
    parser.add_argument("model", nargs="?", default="tokenizer.model", help="path to .model file")
    parser.add_argument("--top", type=int, default=0, help="only print first N tokens (0 = all)")
    parser.add_argument("--search", type=str, default=None, help="search for tokens containing this string")
    args = parser.parse_args()

    sp = SentencePieceProcessor(model_file=args.model)
    vocab_size = sp.vocab_size()
    print(f"Tokenizer: {args.model}")
    print(f"Vocab size: {vocab_size}")
    print(f"BOS id: {sp.bos_id()}, EOS id: {sp.eos_id()}, UNK id: {sp.unk_id()}")
    print()

    count = 0
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        display = piece.replace("\u2581", "_")  # show the SentencePiece space marker

        if args.search and args.search.lower() not in display.lower():
            continue

        print(f"{i:6d}  {score:10.4f}  {display}")
        count += 1

        if args.top > 0 and count >= args.top:
            break

    if args.search:
        print(f"\n{count} tokens matching '{args.search}'")


if __name__ == "__main__":
    main()

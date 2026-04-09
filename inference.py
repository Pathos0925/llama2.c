"""
Inference script for generating text from a trained checkpoint.

Usage:
  python inference.py --checkpoint out/ckpt.pt --prompt "Once upon a time"
  python inference.py --checkpoint out/ckpt.pt --num_samples 3 --max_tokens 200
  python inference.py --checkpoint out/ckpt.pt --prompt "The cat" --temperature 0.5 --top_k 40
"""

import argparse
import torch
from model import ModelArgs, Transformer
from tokenizer import Tokenizer


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, gptconf


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, default="out/ckpt.pt", help="path to checkpoint file")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="text prompt to start generation")
    parser.add_argument("--num_samples", type=int, default=1, help="number of samples to generate")
    parser.add_argument("--max_tokens", type=int, default=256, help="max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature (0 = greedy)")
    parser.add_argument("--top_k", type=int, default=50, help="top-k sampling (0 = disabled)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer", type=str, default=None, help="path to tokenizer.model (default: auto-detect)")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.n_layers} layers, {config.dim} dim, {n_params:,} params")

    # Load tokenizer
    tokenizer = Tokenizer(args.tokenizer)
    print(f"Tokenizer: {tokenizer.n_words} tokens")

    # Encode prompt
    prompt_tokens = tokenizer.encode(args.prompt, bos=True, eos=False)
    prompt_tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=args.device)

    top_k = args.top_k if args.top_k > 0 else None

    print(f"\n--- Generating ({args.max_tokens} tokens, temp={args.temperature}, top_k={top_k}) ---\n")

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"=== Sample {i + 1}/{args.num_samples} ===")

        with torch.no_grad():
            output = model.generate(
                prompt_tokens,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=top_k,
            )

        text = tokenizer.decode(output[0].tolist())
        print(text)

        if args.num_samples > 1:
            print()


if __name__ == "__main__":
    main()

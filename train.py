"""
Pretrain GPT on the-verdict.txt (LLMs-from-scratch demo corpus).

Run from project root:
    python scripts/download_datasets.py
    python train.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken
import torch

from config import GPT_CONFIG_124M_TRAIN
from data_loader import create_dataloader_v1
from tokenizer import BPETokenizer
from training import calculate_loss, evaluate, generate_text, train_epoch
from transformer import GPTModel

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "the-verdict.txt"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "model.pth"


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested in ("auto", "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_text(path: Path, skip_tokens: int = 50) -> str:
    raw = path.read_text(encoding="utf-8")
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(raw, allowed_special={"<|endoftext|>"})
    if len(token_ids) <= skip_tokens:
        return raw
    return tokenizer.decode(token_ids[skip_tokens:])


def save_checkpoint(
    model: GPTModel,
    config: dict,
    path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain GPT on the-verdict.txt")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(
            f"Missing {args.data}. Run: python scripts/download_datasets.py"
        )

    device = resolve_device(args.device)
    cfg = dict(GPT_CONFIG_124M_TRAIN)
    context_length = cfg["context_length"]

    torch.manual_seed(args.seed)
    text_data = load_text(args.data)
    split_idx = int(args.train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    print(f"Device: {device}")
    print(f"Context length: {context_length}")
    print(f"Train chars: {len(train_data)}, Val chars: {len(val_data)}")

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=args.batch_size,
        max_length=context_length,
        stride=context_length,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=args.batch_size,
        max_length=context_length,
        stride=context_length,
        shuffle=False,
        drop_last=False,
    )

    print(f"Train batches/epoch: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = GPTModel(cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    tokenizer = BPETokenizer()
    start_context = "Every effort moves you"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device=device)
        val_loss = evaluate(model, val_loader, device=device)
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        sample = generate_text(
            model,
            tokenizer,
            start_context,
            max_new_tokens=30,
            temperature=0.8,
            top_k=50,
            device=device,
        )
        print(f"  sample: {sample[:120]}...")

        save_checkpoint(
            model, cfg, args.checkpoint, epoch, train_loss, val_loss
        )

    print(f"\nCheckpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()

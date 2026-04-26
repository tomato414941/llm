import argparse
from pathlib import Path

import torch

from llm.models import TransformerConfig, TransformerLanguageModel
from llm.tokenizer import CharTokenizer
from llm.training import estimate_loss, get_batch


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def save_checkpoint(
    path: Path,
    model: TransformerLanguageModel,
    tokenizer: CharTokenizer,
    config: TransformerConfig,
    step: int,
    losses: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_chars": tokenizer.chars,
            "config": config.to_dict(),
            "step": step,
            "losses": losses,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mini_gpt.pt"))
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--generate-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()

    torch.manual_seed(1337)

    text = load_text(args.input)
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_index = int(0.9 * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]

    if len(train_data) <= args.block_size or len(val_data) <= args.block_size:
        raise ValueError("input text is too small for the requested block size")

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    model = TransformerLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    latest_losses = {"train": float("nan"), "val": float("nan")}

    for step in range(args.max_iters):
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            latest_losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                block_size=args.block_size,
                batch_size=args.batch_size,
                eval_iters=args.eval_iters,
            )
            print(
                f"step {step}: "
                f"train loss {latest_losses['train']:.4f}, "
                f"val loss {latest_losses['val']:.4f}"
            )

        xb, yb = get_batch(train_data, args.block_size, args.batch_size)
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("loss was not computed")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_checkpoint(
        path=args.checkpoint,
        model=model,
        tokenizer=tokenizer,
        config=config,
        step=args.max_iters - 1,
        losses=latest_losses,
    )
    print(f"\ncheckpoint saved to {args.checkpoint}")

    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(
        context,
        max_new_tokens=args.generate_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    print("\n--- sample ---")
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()

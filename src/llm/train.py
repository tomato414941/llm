import argparse
import csv
import math
from pathlib import Path
import sys

import torch

from llm.config import compact_defaults, config_defaults, load_toml
from llm.models import TransformerConfig, TransformerLanguageModel
from llm.tokenizer import BPETokenizer, CharTokenizer, tokenizer_from_payload
from llm.training import estimate_loss, get_batch, split_train_val


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def perplexity(loss: float) -> float:
    return math.exp(loss)


def format_loss_line(step: int, losses: dict[str, float]) -> str:
    train_loss = losses["train"]
    val_loss = losses["val"]
    return (
        f"step {step}: "
        f"train loss {train_loss:.4f}, train ppl {perplexity(train_loss):.2f}, "
        f"val loss {val_loss:.4f}, val ppl {perplexity(val_loss):.2f}"
    )


def append_metrics_row(path: Path, step: int, losses: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=("step", "train_loss", "val_loss", "train_ppl", "val_ppl"),
        )
        if not file_exists:
            writer.writeheader()
        train_loss = losses["train"]
        val_loss = losses["val"]
        writer.writerow(
            {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_ppl": perplexity(train_loss),
                "val_ppl": perplexity(val_loss),
            }
        )


def save_checkpoint(
    path: Path,
    model: TransformerLanguageModel,
    tokenizer: CharTokenizer | BPETokenizer,
    config: TransformerConfig,
    optimizer: torch.optim.Optimizer,
    step: int,
    losses: dict[str, float],
    metadata: dict[str, int | float | str],
    tokens_seen: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": torch.random.get_rng_state(),
            "tokenizer": tokenizer.to_payload(),
            "config": config.to_dict(),
            "step": step,
            "losses": losses,
            "metadata": metadata,
            "tokens_seen": tokens_seen,
        },
        path,
    )


def load_tokenizer(kind: str, text: str, path: Path | None) -> CharTokenizer | BPETokenizer:
    if kind == "char":
        return CharTokenizer.from_text(text)
    if kind == "bpe":
        if path is None:
            raise ValueError("--tokenizer-path is required when --tokenizer bpe")
        return BPETokenizer.load(path)
    raise ValueError(f"unknown tokenizer: {kind}")


def default_context_ids(tokenizer: CharTokenizer | BPETokenizer) -> list[int]:
    encoded = tokenizer.encode("\n")
    return encoded if encoded else [0]


def tokenizer_type(tokenizer: CharTokenizer | BPETokenizer) -> str:
    if isinstance(tokenizer, CharTokenizer):
        return "char"
    if isinstance(tokenizer, BPETokenizer):
        return "bpe"
    raise TypeError(f"unsupported tokenizer: {type(tokenizer)!r}")


def load_training_data(
    input_path: Path | None,
    tokens_path: Path | None,
    tokenizer_kind: str,
    tokenizer_path: Path | None,
) -> tuple[torch.Tensor, CharTokenizer | BPETokenizer, dict[str, int | float | str]]:
    if (input_path is None) == (tokens_path is None):
        raise ValueError("exactly one of --input or --tokens is required")

    if tokens_path is not None:
        prepared = torch.load(tokens_path, map_location="cpu", weights_only=True)
        tokenizer = tokenizer_from_payload(prepared["tokenizer"])
        tokens = prepared["tokens"].to(dtype=torch.long)
        metadata = dict(prepared.get("metadata", {}))
        metadata["tokens"] = str(tokens_path)
        return tokens, tokenizer, metadata

    if input_path is None:
        raise ValueError("--input is required when --tokens is not used")
    text = load_text(input_path)
    tokenizer = load_tokenizer(tokenizer_kind, text, tokenizer_path)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return tokens, tokenizer, {"input": str(input_path)}


def validate_args(args: argparse.Namespace) -> None:
    positive_int_fields = (
        "max_iters",
        "eval_interval",
        "eval_iters",
        "block_size",
        "batch_size",
        "embedding_dim",
        "num_heads",
        "num_layers",
        "generate_tokens",
    )
    for field in positive_int_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")

    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be positive")


def build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.set_defaults(**defaults)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", choices=("char", "bpe"))
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-iters", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--generate-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--metrics-output", type=Path)
    return parser


def parse_args() -> argparse.Namespace:
    base_defaults = {
        "checkpoint": Path("checkpoints/mini_gpt.pt"),
        "tokenizer": "char",
        "max_iters": 3000,
        "eval_interval": 300,
        "eval_iters": 20,
        "block_size": 32,
        "batch_size": 32,
        "embedding_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "generate_tokens": 300,
        "temperature": 1.0,
        "seed": 1337,
    }
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _unknown = config_parser.parse_known_args()
    defaults = dict(base_defaults)
    if config_args.config is not None:
        loaded_config = load_toml(config_args.config)
        defaults.update(compact_defaults(config_defaults(loaded_config)))
        defaults["config"] = config_args.config
    parser = build_parser(defaults)
    args = parser.parse_args()
    checkpoint_was_explicit = any(
        argument == "--checkpoint" or argument.startswith("--checkpoint=") for argument in sys.argv
    )
    if args.resume is not None and not checkpoint_was_explicit:
        args.checkpoint = args.resume
    if isinstance(args.input, str):
        args.input = Path(args.input)
    if isinstance(args.tokens, str):
        args.tokens = Path(args.tokens)
    if isinstance(args.manifest, str):
        args.manifest = Path(args.manifest)
    if isinstance(args.checkpoint, str):
        args.checkpoint = Path(args.checkpoint)
    if isinstance(args.metrics_output, str):
        args.metrics_output = Path(args.metrics_output)
    if isinstance(args.tokenizer_path, str):
        args.tokenizer_path = Path(args.tokenizer_path)
    return args


def require_resume_key(checkpoint: dict[str, object], key: str) -> object:
    if key not in checkpoint:
        raise ValueError(f"resume checkpoint must contain {key!r}")
    return checkpoint[key]


def main() -> None:
    args = parse_args()
    validate_args(args)

    torch.manual_seed(args.seed)

    data, tokenizer, data_metadata = load_training_data(
        input_path=args.input,
        tokens_path=args.tokens,
        tokenizer_kind=args.tokenizer,
        tokenizer_path=args.tokenizer_path,
    )
    train_data, val_data = split_train_val(data)

    if len(train_data) <= args.block_size or len(val_data) <= args.block_size:
        raise ValueError("input text is too small for the requested block size")

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = TransformerLanguageModel(config)
    parameter_count = count_parameters(model)
    print(f"parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    latest_losses = {"train": float("nan"), "val": float("nan")}
    start_step = 0
    tokens_seen = 0

    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        optimizer_state_dict = require_resume_key(resume_checkpoint, "optimizer_state_dict")
        rng_state = require_resume_key(resume_checkpoint, "rng_state")
        checkpoint_config = TransformerConfig.from_dict(
            require_resume_key(resume_checkpoint, "config")  # type: ignore[arg-type]
        )
        if checkpoint_config != config:
            raise ValueError("resume checkpoint config does not match requested config")
        model.load_state_dict(require_resume_key(resume_checkpoint, "model_state_dict"))
        optimizer.load_state_dict(optimizer_state_dict)  # type: ignore[arg-type]
        torch.random.set_rng_state(rng_state)  # type: ignore[arg-type]
        start_step = int(require_resume_key(resume_checkpoint, "step")) + 1
        tokens_seen = int(resume_checkpoint.get("tokens_seen", 0))
        latest_losses = dict(resume_checkpoint.get("losses", latest_losses))

    for step in range(start_step, args.max_iters):
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            latest_losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                block_size=args.block_size,
                batch_size=args.batch_size,
                eval_iters=args.eval_iters,
            )
            print(format_loss_line(step, latest_losses))
            if args.metrics_output is not None:
                append_metrics_row(args.metrics_output, step, latest_losses)

        xb, yb = get_batch(train_data, args.block_size, args.batch_size)
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("loss was not computed")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        tokens_seen += args.batch_size * args.block_size

    save_checkpoint(
        path=args.checkpoint,
        model=model,
        tokenizer=tokenizer,
        config=config,
        optimizer=optimizer,
        step=args.max_iters - 1,
        losses=latest_losses,
        metadata={
            "run_id": args.run_id or "",
            "config_path": str(args.config) if args.config is not None else "",
            "manifest_path": str(args.manifest) if args.manifest is not None else "",
            "max_iters": args.max_iters,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "parameter_count": parameter_count,
            "tokenizer": tokenizer_type(tokenizer),
            "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path is not None else "",
        }
        | data_metadata,
        tokens_seen=tokens_seen,
    )
    print(f"\ncheckpoint saved to {args.checkpoint}")

    model.eval()
    context = torch.tensor([default_context_ids(tokenizer)], dtype=torch.long)
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

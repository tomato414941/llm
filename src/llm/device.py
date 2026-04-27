import torch


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is not available")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")
    return torch.device(requested)

from collections.abc import Sequence


def build_instruction_contract(
    *,
    prompt: str,
    output_format: str,
    constraints: Sequence[str],
) -> str:
    constraints_text = "\n".join(f"- {constraint}" for constraint in constraints)
    return (
        f"{prompt}\n\n"
        f"Output format: {output_format}\n"
        "Constraints:\n"
        f"{constraints_text}"
    )

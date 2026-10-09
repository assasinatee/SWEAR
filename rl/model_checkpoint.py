"""Resolve and classify SFT checkpoints used to initialize RL.

This module intentionally has no torch/transformers imports so launchers and
CPU-only contract tests can validate a checkpoint before allocating a GPU.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SFTCheckpoint:
    path: Path
    format: str  # "lora", "hf_full", or "zero_full"


def _hf_weight_files(path: Path) -> list[Path]:
    names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return [path / name for name in names if (path / name).is_file()]


def _zero_tag(path: Path) -> str | None:
    latest = path / "latest"
    if latest.is_file():
        tag = latest.read_text(encoding="utf-8").strip()
        step_dir = path / tag
        if tag and step_dir.is_dir() and any(
            step_dir.glob("zero_pp_rank_*_model_states.pt")
        ):
            return tag

    step_dirs = [
        child
        for child in sorted(path.glob("global_step*"))
        if child.is_dir()
        and any(child.glob("zero_pp_rank_*_model_states.pt"))
    ]
    if len(step_dirs) == 1:
        return step_dirs[0].name
    return None


def _classify(path: Path) -> str | None:
    if (path / "adapter_config.json").is_file():
        return "lora"
    if (path / "config.json").is_file() and _hf_weight_files(path):
        return "hf_full"
    if (path / "config.json").is_file() and _zero_tag(path):
        return "zero_full"
    return None


def resolve_sft_checkpoint(value: str | Path) -> SFTCheckpoint:
    """Resolve a checkpoint or a parent containing exactly one checkpoint.

    LLaMA-Factory's ``.../Qwen2.5-Omni-7B/full`` directory is a container;
    the actual completed ZeRO checkpoint is currently its
    ``sft_split_compress`` child. Accepting a uniquely resolvable container
    makes that user-facing path usable without silently guessing if another
    completed full run is added later.
    """

    requested = Path(value).expanduser()
    if not requested.is_dir():
        raise FileNotFoundError(f"SFT checkpoint directory not found: {requested}")

    checkpoint_format = _classify(requested)
    if checkpoint_format:
        return SFTCheckpoint(requested.resolve(), checkpoint_format)

    candidates: list[SFTCheckpoint] = []
    for child in sorted(requested.iterdir()):
        if not child.is_dir():
            continue
        child_format = _classify(child)
        if child_format:
            candidates.append(SFTCheckpoint(child.resolve(), child_format))

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        paths = "\n  - ".join(str(item.path) for item in candidates)
        raise ValueError(
            f"SFT checkpoint path is ambiguous: {requested}\n"
            f"Choose one completed checkpoint explicitly:\n  - {paths}"
        )

    raise ValueError(
        f"Unsupported or incomplete SFT checkpoint: {requested}. Expected "
        "adapter_config.json (LoRA), Hugging Face model weights (full), or "
        "a DeepSpeed ZeRO checkpoint with config.json + latest/global_step*."
    )


def zero_checkpoint_tag(path: str | Path) -> str:
    tag = _zero_tag(Path(path))
    if not tag:
        raise ValueError(f"No unambiguous DeepSpeed ZeRO tag found under {path}")
    return tag

"""Troy YAML config schema and loader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class LoraConfig(BaseModel):
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    layers: Union[int, Literal["all"]] = 16  # decoder layers to adapt

    @property
    def scale(self) -> float:
        return self.alpha / self.r


class DpoConfig(BaseModel):
    beta: float = 0.1


class OrpoConfig(BaseModel):
    lam: float = Field(0.1, alias="lambda")
    model_config = {"populate_by_name": True}


class DataConfig(BaseModel):
    train: str
    valid: Optional[str] = None
    format: Literal[
        "auto", "alpaca", "sharegpt", "chat", "completions", "text", "preference"
    ] = "auto"
    val_split: float = Field(0.1, ge=0.0, lt=1.0)
    mask_prompt: bool = False


class TrainingConfig(BaseModel):
    epochs: Optional[float] = None  # translated to iters from dataset size
    iters: Optional[int] = None  # takes precedence over epochs
    lr: float = 1e-5
    batch_size: Union[int, Literal["auto"]] = "auto"
    seq_len: int = 2048
    lora: LoraConfig = LoraConfig()
    dpo: DpoConfig = DpoConfig()
    orpo: OrpoConfig = OrpoConfig()
    grad_checkpoint: bool = False
    grad_accumulation_steps: int = 1
    save_every: int = 100
    seed: int = 0


class TroyConfig(BaseModel):
    base: str
    task: Literal["sft", "dpo", "orpo"] = "sft"
    data: DataConfig
    training: TrainingConfig = TrainingConfig()
    output: str = "./output"

    @field_validator("base")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("`base` must be a HuggingFace repo id or local path")
        return v

    @property
    def output_path(self) -> Path:
        return Path(self.output).expanduser()

    @property
    def adapter_path(self) -> Path:
        return self.output_path / "adapter"


def load_config(path: Union[str, Path]) -> TroyConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}. Run `troy init` to create one."
        )
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return TroyConfig.model_validate(raw)

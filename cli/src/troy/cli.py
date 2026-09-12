"""Troy CLI: fine-tune LLMs on your MacBook with one YAML file."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__

app = typer.Typer(
    name="troy",
    help="Fine-tune LLMs on your MacBook with one YAML file. Built for Apple Silicon.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _require_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        console.print(
            "[red]Troy runs on Apple Silicon Macs only (M1 or later).[/red]\n"
            f"Detected: {platform.system()} / {platform.machine()}"
        )
        raise typer.Exit(1)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"troy {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version.",
        callback=_version_callback, is_eager=True,
    ),
) -> None:
    pass


@app.command()
def init(
    template: str = typer.Option(
        "chat", help="Template: chat (SFT), dpo, or orpo (preference tuning)."
    ),
    path: Path = typer.Option(Path("troy.yaml"), help="Where to write the config."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config."),
) -> None:
    """Create a troy.yaml config (plus sample data) to start from."""
    from .templates import TEMPLATES

    if template not in TEMPLATES:
        console.print(f"[red]Unknown template `{template}`.[/red] Options: {', '.join(TEMPLATES)}")
        raise typer.Exit(1)
    if path.exists() and not force:
        console.print(f"[red]{path} already exists.[/red] Use --force to overwrite.")
        raise typer.Exit(1)

    config_text, data_name, sample = TEMPLATES[template]
    path.write_text(config_text)

    data_dir = path.parent / "data"
    data_file = data_dir / data_name
    if not data_file.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_file, "w") as f:
            for record in sample:
                f.write(json.dumps(record) + "\n")
        console.print(f"Wrote sample data to [bold]{data_file}[/bold] — replace it with yours.")

    console.print(f"Created [bold]{path}[/bold] ({template} template).")
    console.print("Next: edit the config, then run [bold]troy train[/bold].")


@app.command()
def doctor() -> None:
    """Check this Mac's readiness for local fine-tuning."""
    from .hardware import detect, model_guidance

    hw = detect()
    table = Table(title="troy doctor", show_header=False)
    table.add_column(style="bold")
    table.add_column()

    ok = "[green]OK[/green]"
    fail = "[red]FAIL[/red]"

    table.add_row("Chip", hw.chip)
    table.add_row(
        "Apple Silicon", ok if hw.is_apple_silicon else f"{fail} ({hw.arch})"
    )
    table.add_row("Unified memory", f"{hw.memory_gb:.0f} GB")
    table.add_row("macOS", hw.macos)
    table.add_row("Free disk", f"{hw.free_disk_gb:.0f} GB")
    table.add_row("Python", platform.python_version())

    try:
        import mlx.core as mx

        table.add_row("MLX", f"{ok} (v{mx.__version__})")
        gpu_ok = mx.default_device().type == mx.DeviceType.gpu
        table.add_row("Metal GPU", ok if gpu_ok else f"{fail} (default device is CPU)")
    except ImportError:
        table.add_row("MLX", f"{fail} (not installed — `pip install mlx-lm`)")

    try:
        import mlx_lm

        table.add_row("mlx-lm", f"{ok} (v{mlx_lm.__version__})")
    except ImportError:
        table.add_row("mlx-lm", f"{fail} (not installed)")

    table.add_row("Fine-tunable models", model_guidance(hw.memory_gb))
    console.print(table)

    if not hw.is_apple_silicon:
        raise typer.Exit(1)


@app.command()
def train(
    config: Path = typer.Option(Path("troy.yaml"), "--config", "-c", help="Config file."),
) -> None:
    """Fine-tune a model from a troy.yaml config."""
    _require_apple_silicon()
    from .config import load_config
    from .data import load_and_prepare

    cfg = load_config(config)
    train_records, valid_records, fmt = load_and_prepare(
        cfg.data, cfg.task, cfg.training.seed
    )
    console.print(
        f"Data: {len(train_records)} train / {len(valid_records)} valid "
        f"(format: {fmt})"
    )

    if cfg.task == "sft":
        from .train_sft import run_sft

        run_sft(cfg, train_records, valid_records)
    elif cfg.task == "orpo":
        from .train_orpo import run_orpo

        run_orpo(cfg, train_records, valid_records)
    else:
        from .train_dpo import run_dpo

        run_dpo(cfg, train_records, valid_records)

    console.print(
        f"\nTry it: [bold]troy chat[/bold]   |   "
        f"Export it: [bold]troy export[/bold]"
    )


@app.command()
def chat(
    config: Path = typer.Option(Path("troy.yaml"), "--config", "-c", help="Config file."),
    model: Optional[str] = typer.Option(
        None, help="Model path or HF repo (defaults to the config's base + trained adapter)."
    ),
    base_only: bool = typer.Option(False, help="Chat with the base model, no adapter."),
    max_tokens: int = typer.Option(512),
    temperature: float = typer.Option(0.7),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="One-shot prompt (no REPL)."),
) -> None:
    """Chat with your fine-tuned model."""
    _require_apple_silicon()
    from .chat import run_chat

    adapter: Optional[str] = None
    if model is None:
        from .config import load_config

        cfg = load_config(config)
        model = cfg.base
        if not base_only:
            adapter_file = cfg.adapter_path / "adapters.safetensors"
            if adapter_file.exists():
                adapter = str(cfg.adapter_path)
            else:
                console.print(
                    "[yellow]No trained adapter found — chatting with the base model.[/yellow]"
                )
    run_chat(model, adapter, max_tokens, temperature, prompt)


@app.command()
def serve(
    config: Path = typer.Option(Path("troy.yaml"), "--config", "-c", help="Config file."),
    model: Optional[str] = typer.Option(
        None, help="Model path or HF repo (defaults to the config's base + trained adapter)."
    ),
    base_only: bool = typer.Option(False, help="Serve the base model, no adapter."),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8080),
    max_tokens: int = typer.Option(512, help="Default max tokens per response."),
) -> None:
    """Serve your model over an OpenAI-compatible API."""
    _require_apple_silicon()
    from .serve import run_serve

    adapter: Optional[str] = None
    if model is None:
        from .config import load_config

        cfg = load_config(config)
        model = cfg.base
        if not base_only:
            adapter_file = cfg.adapter_path / "adapters.safetensors"
            if adapter_file.exists():
                adapter = str(cfg.adapter_path)
            else:
                console.print(
                    "[yellow]No trained adapter found — serving the base model.[/yellow]"
                )
    run_serve(model, adapter, host, port, max_tokens)


@app.command()
def export(
    config: Path = typer.Option(Path("troy.yaml"), "--config", "-c", help="Config file."),
    fmt: str = typer.Option("mlx", "--format", "-f", help="Export format: mlx or gguf."),
    save_path: Optional[Path] = typer.Option(None, help="Output directory (default: <output>/fused)."),
    dequantize: bool = typer.Option(False, help="Dequantize when fusing a quantized base."),
) -> None:
    """Merge the trained adapter into the base model and export it."""
    _require_apple_silicon()
    if fmt not in ("mlx", "gguf"):
        console.print("[red]--format must be `mlx` or `gguf`.[/red]")
        raise typer.Exit(1)
    from .config import load_config
    from .export import run_export

    cfg = load_config(config)
    adapter_file = cfg.adapter_path / "adapters.safetensors"
    if not adapter_file.exists():
        console.print(f"[red]No adapter at {adapter_file}. Run `troy train` first.[/red]")
        raise typer.Exit(1)
    out = save_path or (cfg.output_path / "fused")
    run_export(cfg.base, str(cfg.adapter_path), str(out), fmt, dequantize)


@app.command()
def eval(
    config: Path = typer.Option(Path("troy.yaml"), "--config", "-c", help="Config file."),
    prompts: Optional[List[str]] = typer.Option(
        None, "--prompt", "-p",
        help="Prompt for side-by-side base-vs-tuned generation (repeatable).",
    ),
    max_tokens: int = typer.Option(200),
) -> None:
    """Compare the trained adapter against the base model (loss, ppl, samples)."""
    _require_apple_silicon()
    from .config import load_config
    from .data import load_and_prepare
    from .evaluate import run_eval

    cfg = load_config(config)
    _, valid_records, _ = load_and_prepare(cfg.data, cfg.task, cfg.training.seed)
    run_eval(cfg, valid_records, list(prompts) if prompts else None, max_tokens)


@app.command()
def push(
    repo: str = typer.Argument(help="Hub repo id, e.g. username/my-model."),
    config: Path = typer.Option(Path("troy.yaml"), "--config", "-c", help="Config file."),
    fused: bool = typer.Option(False, help="Push the fused model instead of the adapter."),
    public: bool = typer.Option(False, help="Make the Hub repo public."),
) -> None:
    """Upload your trained adapter (or fused model) to the Hugging Face Hub."""
    from .config import load_config
    from .push import run_push

    cfg = load_config(config)
    folder = (cfg.output_path / "fused") if fused else cfg.adapter_path
    if not folder.exists():
        what = "troy export" if fused else "troy train"
        console.print(f"[red]{folder} not found. Run `{what}` first.[/red]")
        raise typer.Exit(1)
    run_push(folder, repo, private=not public)


@app.command()
def data(
    action: str = typer.Argument(help="Action: inspect"),
    path: Path = typer.Argument(help="Dataset file (.jsonl, .json, .csv)."),
) -> None:
    """Inspect a dataset: record count, detected format, sizes."""
    if action != "inspect":
        console.print("[red]Only `troy data inspect <path>` is supported.[/red]")
        raise typer.Exit(1)
    from .data import inspect_stats

    stats = inspect_stats(str(path))
    table = Table(title=str(path), show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Records", str(stats["records"]))
    table.add_row("Detected format", stats["format"])
    table.add_row("Avg record size", f"{stats['avg_chars']:.0f} chars")
    table.add_row("Max record size", f"{stats['max_chars']} chars")
    console.print(table)


if __name__ == "__main__":
    app()

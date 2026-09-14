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

    table.add_row("Comfortable size", model_guidance(hw.memory_gb))
    table.add_row(
        "Supported models",
        "any mlx-lm architecture (Llama, Qwen, Gemma, Phi, Mistral, ...)\n"
        "thousands of ready conversions: hf.co/mlx-community",
    )
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

    if Path(cfg.data.train).expanduser().is_dir():  # folder of images => vision
        if cfg.task != "sft":
            console.print("[red]Vision fine-tuning supports task: sft (for now).[/red]")
            raise typer.Exit(1)
        from .train_vision import run_vision_sft

        run_vision_sft(cfg)
        console.print(
            '\nTry it: [bold]troy chat --image photo.png -p "your question"[/bold]'
        )
        return

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
    image: Optional[Path] = typer.Option(None, help="Image for a vision model (one-shot; needs -p)."),
    tools: Optional[Path] = typer.Option(
        None, "--tools",
        help="JSON file with tool schemas (OpenAI function format) — rendered "
        "into the chat template so a tool-tuned model can emit <tool_call>s.",
    ),
    system: Optional[str] = typer.Option(
        None, "--system", help="System prompt (use the one the model was trained with)."
    ),
) -> None:
    """Chat with your fine-tuned model (pass --tools to exercise tool calling)."""
    _require_apple_silicon()
    import json

    from .chat import run_chat

    tool_schemas = None
    if tools is not None:
        if image is not None:
            console.print("[red]--tools and --image can't be combined.[/red]")
            raise typer.Exit(1)
        try:
            tool_schemas = json.loads(tools.read_text())
        except (OSError, json.JSONDecodeError) as e:
            console.print(f"[red]Can't read {tools}:[/red] {e}")
            raise typer.Exit(1)
        if not isinstance(tool_schemas, list) or not tool_schemas:
            console.print(f"[red]{tools} must hold a JSON list of tool schemas.[/red]")
            raise typer.Exit(1)

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
    if image is not None:
        if prompt is None:
            console.print("[red]--image needs a one-shot prompt: -p \"your question\"[/red]")
            raise typer.Exit(1)
        from .train_vision import run_vision_chat

        run_vision_chat(model, adapter, str(image), prompt, max_tokens, temperature)
        return
    run_chat(model, adapter, max_tokens, temperature, prompt,
             tools=tool_schemas, system=system)


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
    fmt: str = typer.Option("mlx", "--format", "-f", help="Export format: mlx, gguf, or ios."),
    save_path: Optional[Path] = typer.Option(None, help="Output directory (default: <output>/fused)."),
    dequantize: bool = typer.Option(False, help="Dequantize when fusing a quantized base."),
) -> None:
    """Merge the trained adapter into the base model and export it."""
    _require_apple_silicon()
    if fmt not in ("mlx", "gguf", "ios"):
        console.print("[red]--format must be `mlx`, `gguf`, or `ios`.[/red]")
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


data_app = typer.Typer(
    name="data",
    help="Inspect, validate, and synthesize datasets.",
    no_args_is_help=True,
)
app.add_typer(data_app)


@data_app.command()
def inspect(
    path: Path = typer.Argument(help="Dataset file (.jsonl, .json, .csv)."),
) -> None:
    """Inspect a dataset: record count, detected format, sizes."""
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


@data_app.command()
def validate(
    path: Path = typer.Argument(help="Dataset file (.jsonl, .json, .csv)."),
) -> None:
    """Lint a dataset: broken records, mixed formats, empty fields, duplicates."""
    from .data import validate_records

    report = validate_records(str(path))
    console.print(
        f"{report['records']} records, format: [bold]{report['format']}[/bold]"
    )
    if not report["issues"]:
        console.print("[green]No issues found.[/green]")
        return
    for issue in report["issues"]:
        console.print(f"  [yellow]•[/yellow] {issue}")
    if report["truncated"]:
        console.print("  [dim]... more issues not shown[/dim]")
    console.print(f"[red]{len(report['issues'])}{'+' if report['truncated'] else ''} issue(s).[/red]")
    raise typer.Exit(1)


@data_app.command()
def synth(
    source: Optional[Path] = typer.Option(
        None, "--from", help="Ground examples in a file or folder of docs/code."
    ),
    seed: Optional[str] = typer.Option(
        None, "--seed", help='Task description, e.g. "customer support bot for Acme".'
    ),
    n: int = typer.Option(100, "--n", help="Number of examples to generate."),
    fmt: str = typer.Option(
        "chat", "--format", "-f",
        help="Output format: chat (SFT), preference (DPO/ORPO), or tools (tool calling).",
    ),
    tools: Optional[Path] = typer.Option(
        None, "--tools",
        help="For -f tools: JSON file with the tool schemas (OpenAI function format).",
    ),
    no_think: bool = typer.Option(
        False, "--no-think",
        help="For -f tools: omit the <think> reasoning traces.",
    ),
    teacher: str = typer.Option(
        "auto", help="Teacher model (auto = sized to this Mac's memory)."
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o",
        help="Output file (default: data/train.jsonl or data/preferences.jsonl).",
    ),
    max_tokens: int = typer.Option(2048, help="Max tokens per teacher call."),
    temperature: float = typer.Option(0.8),
) -> None:
    """Synthesize a training dataset with a local teacher model."""
    _require_apple_silicon()
    if source is None and seed is None:
        console.print(
            '[red]Give the teacher something to work from:[/red] '
            '--from ./docs and/or --seed "task description".'
        )
        raise typer.Exit(1)
    if fmt not in ("chat", "preference", "tools"):
        console.print("[red]--format must be `chat`, `preference`, or `tools`.[/red]")
        raise typer.Exit(1)
    if fmt == "tools":
        if tools is None or not tools.exists():
            console.print(
                "[red]-f tools needs --tools schemas.json[/red] — a JSON list of "
                "OpenAI-style function specs the assistant can call."
            )
            raise typer.Exit(1)
        if seed is None:
            console.print(
                '[red]-f tools needs --seed[/red] — it becomes the system prompt, '
                'e.g. "travel planning assistant that books nothing".'
            )
            raise typer.Exit(1)

    from .synth import pick_teacher, run_synth

    if teacher == "auto":
        teacher = pick_teacher()
        console.print(f"Teacher: [bold]{teacher}[/bold] (picked for this Mac's memory)")
    out = out or Path("data") / ("preferences.jsonl" if fmt == "preference" else "train.jsonl")
    if out.exists():
        console.print(f"[red]{out} already exists[/red] — pass -o to write elsewhere.")
        raise typer.Exit(1)

    stats = run_synth(
        n=n, out_path=out, fmt=fmt, teacher=teacher,
        seed_task=seed, source=source,
        max_tokens=max_tokens, temperature=temperature,
        tools_path=tools, think=not no_think,
    )
    console.print(
        f"\nWrote [bold]{stats['records']}[/bold] examples to [bold]{stats['out']}[/bold] "
        f"({stats['teacher_calls']} teacher calls)"
    )
    if stats["records"] < stats["requested"]:
        console.print(
            f"[yellow]Stopped at {stats['records']}/{stats['requested']} — "
            "try a larger --teacher, higher --max-tokens, or more source material.[/yellow]"
        )
    console.print(
        "Review the data before training — spot-check a dozen examples, then: "
        "[bold]troy data validate " + str(out) + "[/bold] and [bold]troy train[/bold]."
    )


mesh_app = typer.Typer(
    name="mesh",
    help="Distribute data synthesis across devices on your LAN.",
    no_args_is_help=True,
)
app.add_typer(mesh_app)


@mesh_app.command("serve")
def mesh_serve(
    source: Optional[Path] = typer.Option(
        None, "--from", help="Ground examples in a file or folder of docs/code."
    ),
    seed: Optional[str] = typer.Option(
        None, "--seed", help='Task description, e.g. "customer support bot for Acme".'
    ),
    n: int = typer.Option(100, "--n", help="Number of examples to generate."),
    fmt: str = typer.Option(
        "chat", "--format", "-f",
        help="Output format: chat (SFT), preference (DPO/ORPO), or tools (tool calling).",
    ),
    tools: Optional[Path] = typer.Option(
        None, "--tools",
        help="For -f tools: JSON file with the tool schemas (OpenAI function format).",
    ),
    no_think: bool = typer.Option(
        False, "--no-think",
        help="For -f tools: omit the <think> reasoning traces.",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o",
        help="Output file (default: data/train.jsonl or data/preferences.jsonl).",
    ),
    max_tokens: int = typer.Option(2048, help="Max tokens per teacher call."),
    temperature: float = typer.Option(0.8),
    host: str = typer.Option("0.0.0.0", help="Interface to bind."),
    port: int = typer.Option(8765),
    token: Optional[str] = typer.Option(
        None, help="Shared worker token (default: generated at startup)."
    ),
    lease_timeout: float = typer.Option(
        300.0, help="Seconds before an unanswered work item is requeued."
    ),
    linger: float = typer.Option(
        30.0, help="Seconds to wait for in-flight results after the target is hit."
    ),
) -> None:
    """Coordinate a mesh: serve synth work to iPhones and Macs on your LAN.

    Workers run the teacher model; this machine only mints prompts and
    validates results, so it can be any Mac (the model never loads here).
    """
    if source is None and seed is None:
        console.print(
            '[red]Give the workers something to work from:[/red] '
            '--from ./docs and/or --seed "task description".'
        )
        raise typer.Exit(1)
    if fmt not in ("chat", "preference", "tools"):
        console.print("[red]--format must be `chat`, `preference`, or `tools`.[/red]")
        raise typer.Exit(1)
    if fmt == "tools":
        if tools is None or not tools.exists():
            console.print(
                "[red]-f tools needs --tools schemas.json[/red] — a JSON list of "
                "OpenAI-style function specs the assistant can call."
            )
            raise typer.Exit(1)
        if seed is None:
            console.print(
                '[red]-f tools needs --seed[/red] — it becomes the system prompt.'
            )
            raise typer.Exit(1)
    out = out or Path("data") / ("preferences.jsonl" if fmt == "preference" else "train.jsonl")
    if out.exists():
        console.print(f"[red]{out} already exists[/red] — pass -o to write elsewhere.")
        raise typer.Exit(1)

    import secrets

    from rich.panel import Panel

    from .mesh import MeshState, lan_ip, run_mesh_serve
    from .synth import prepare_synth

    token = token or secrets.token_urlsafe(16)
    state = MeshState(
        prepare_synth(fmt, seed, source, tools), n, out,
        max_tokens=max_tokens, temperature=temperature,
        lease_timeout=lease_timeout, think=not no_think,
    )
    join_url = f"http://{lan_ip()}:{port}"
    console.print(Panel.fit(
        f"Join from a Mac:   [bold]troy mesh join {join_url} --token {token}[/bold]\n"
        f"Join from iPhone:  TroyWorker app → {join_url} + token [bold]{token}[/bold]",
        title="troy mesh coordinator",
    ))

    try:
        stats = run_mesh_serve(state, host, port, token, linger=linger)
    except OSError as e:
        state.close()
        if out.exists() and out.stat().st_size == 0:
            out.unlink()  # nothing was written; don't block a relaunch
        console.print(f"[red]Can't bind {host}:{port}[/red] ({e.strerror}) — "
                      "pass --port to use a different one.")
        raise typer.Exit(1)
    console.print(
        f"\nWrote [bold]{stats['records']}[/bold] examples to [bold]{stats['out']}[/bold] "
        f"across {len(stats['workers'])} worker(s)"
    )
    if stats["records"] < stats["target"]:
        console.print(
            f"[yellow]Stopped at {stats['records']}/{stats['target']}.[/yellow]"
        )
    console.print(
        "Review the data before training — spot-check a dozen examples, then: "
        f"[bold]troy data validate {out}[/bold] and [bold]troy train[/bold]."
    )


@mesh_app.command("join")
def mesh_join(
    url: str = typer.Argument(..., help="Coordinator URL, e.g. http://192.168.1.5:8765"),
    token: str = typer.Option(..., help="Token printed by `troy mesh serve`."),
    model: str = typer.Option(
        "auto", help="Teacher model to run here (auto = sized to this Mac's memory)."
    ),
    name: Optional[str] = typer.Option(
        None, help="Worker name shown on the coordinator (default: hostname)."
    ),
    batch: int = typer.Option(2, help="Work items to lease per request."),
) -> None:
    """Join a mesh as a worker: run the teacher here, send results back."""
    _require_apple_silicon()

    import socket

    from .mesh import run_mesh_join
    from .synth import pick_teacher

    if model == "auto":
        model = pick_teacher()
        console.print(f"Teacher: [bold]{model}[/bold] (picked for this Mac's memory)")
    stats = run_mesh_join(url, token, model, name or socket.gethostname(), batch=batch)
    console.print(
        f"\nDone: completed [bold]{stats['completed']}[/bold] work item(s) "
        f"({stats.get('records', '?')}/{stats.get('target', '?')} mesh total)."
    )


if __name__ == "__main__":
    app()

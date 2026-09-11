"""OpenAI-compatible local API server (wraps the mlx-lm HTTP server)."""

from __future__ import annotations

import subprocess
import sys
from typing import Optional


def run_serve(
    model: str,
    adapter_path: Optional[str],
    host: str,
    port: int,
    max_tokens: int,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--max-tokens",
        str(max_tokens),
    ]
    if adapter_path:
        cmd += ["--adapter-path", adapter_path]

    print(f"Serving {model}" + (f" + adapter {adapter_path}" if adapter_path else ""))
    print(f"OpenAI-compatible API at http://{host}:{port}/v1")
    print(
        "Try it:\n"
        f'  curl http://{host}:{port}/v1/chat/completions \\\n'
        '    -H "Content-Type: application/json" \\\n'
        '    -d \'{"messages": [{"role": "user", "content": "Hello"}]}\'\n'
    )
    try:
        raise SystemExit(subprocess.run(cmd).returncode)
    except KeyboardInterrupt:
        print("\nServer stopped.")

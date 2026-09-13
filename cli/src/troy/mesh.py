"""Distribute data synthesis across devices on your LAN.

`troy mesh serve` runs a coordinator: it mints teacher prompts (the same ones
`troy data synth` uses), leases them to workers over plain HTTP, and validates
whatever raw text comes back with the exact same parsing as local synth —
mesh output is indistinguishable from `troy data synth` output.

Workers pull work, run the teacher locally, and POST raw completions back:
`troy mesh join` on any Mac, or the TroyWorker iPhone app
(examples/ios/TroyWorker).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .synth import SynthSpec, ingest_raw, mint_prompt

MAX_ATTEMPTS = 3
RETRY_SECONDS = 5


@dataclass
class WorkItem:
    id: str
    prompt: str
    want: int
    attempts: int = 0
    leased_at: Optional[float] = None
    worker: Optional[str] = None


@dataclass
class WorkerStats:
    completed: int = 0
    duplicates: int = 0
    last_seen: float = 0.0


class MeshState:
    """All coordinator state; every public method is thread-safe."""

    def __init__(
        self,
        spec: SynthSpec,
        n: int,
        out_path: Path,
        max_tokens: int,
        temperature: float,
        lease_timeout: float = 300.0,
        think: bool = True,
    ):
        self.spec = spec
        self.n = n
        self.out_path = out_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.lease_timeout = lease_timeout
        self.think = think

        self._lock = threading.Lock()
        self._seen: Set[str] = set()
        self._available: List[WorkItem] = []
        self._leased: Dict[str, WorkItem] = {}
        self._records = 0
        self._minted = 0
        self._dropped = 0
        self._empty_results = 0
        self._workers: Dict[str, WorkerStats] = {}
        self._started = time.time()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(out_path, "w")

    # -- internals (call with lock held) --------------------------------

    def _reap(self, now: float) -> None:
        expired = [
            i for i in self._leased.values()
            if i.leased_at is not None and now - i.leased_at > self.lease_timeout
        ]
        for item in expired:
            del self._leased[item.id]
            item.leased_at = item.worker = None
            item.attempts += 1
            if item.attempts < MAX_ATTEMPTS:
                self._available.insert(0, item)  # retries jump the queue
            else:
                self._dropped += 1  # a fresh prompt gets minted instead

    def _top_up(self) -> None:
        # keep ~2x the remaining need outstanding, so late/duplicate results
        # never starve the queue
        remaining = max(self.n - self._records, 0)
        outstanding = (
            sum(i.want for i in self._available)
            + sum(i.want for i in self._leased.values())
        )
        while outstanding < 2 * remaining:
            want = min(self.spec.pairs_per_call, remaining)
            item = WorkItem(
                id=f"w-{self._minted:06d}",
                prompt=mint_prompt(self.spec, self._minted, want),
                want=want,
            )
            self._minted += 1
            self._available.append(item)
            outstanding += want

    # -- worker API -----------------------------------------------------

    def lease(
        self, worker: str, max_items: int, now: Optional[float] = None
    ) -> Tuple[List[WorkItem], bool]:
        """Lease up to max_items; returns (items, done)."""
        now = time.time() if now is None else now
        with self._lock:
            self._workers.setdefault(worker, WorkerStats()).last_seen = now
            if self._records >= self.n:
                return [], True
            self._reap(now)
            self._top_up()
            items = self._available[: max(max_items, 0)]
            del self._available[: len(items)]
            for item in items:
                item.leased_at = now
                item.worker = worker
                self._leased[item.id] = item
            return items, False

    def ingest(
        self, worker: str, item_id: str, raw: str, now: Optional[float] = None
    ) -> Dict[str, Any]:
        """Validate one raw completion; returns accept/duplicate counts."""
        now = time.time() if now is None else now
        with self._lock:
            stats = self._workers.setdefault(worker, WorkerStats())
            stats.last_seen = now
            self._leased.pop(item_id, None)  # unknown/expired ids ingest anyway
            records, parsed = ingest_raw(raw, self.spec, self.think, self._seen)
            if not parsed:
                self._empty_results += 1
            room = max(self.n - self._records, 0)
            for r in records[:room]:
                self._fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            self._fh.flush()
            accepted = min(len(records), room)
            self._records += accepted
            stats.completed += accepted
            stats.duplicates += parsed - len(records)
            return {
                "accepted": accepted,
                "duplicates": parsed - len(records),
                "records": self._records,
                "target": self.n,
                "done": self._records >= self.n,
            }

    @property
    def done(self) -> bool:
        with self._lock:
            return self._records >= self.n

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "records": self._records,
                "target": self.n,
                "format": self.spec.fmt,
                "queue": {
                    "available": len(self._available),
                    "leased": len(self._leased),
                    "minted": self._minted,
                    "dropped": self._dropped,
                },
                "empty_results": self._empty_results,
                "workers": {
                    name: {
                        "completed": s.completed,
                        "duplicates": s.duplicates,
                        "last_seen": s.last_seen,
                    }
                    for name, s in self._workers.items()
                },
                "out": str(self.out_path),
                "elapsed_seconds": round(time.time() - self._started, 1),
            }

    def close(self) -> None:
        with self._lock:
            self._fh.close()


class MeshHandler(BaseHTTPRequestHandler):
    """GET /v1/work, POST /v1/results, GET /v1/status."""

    server: "MeshServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        pass  # the live table is the log

    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self) -> bool:
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {self.server.token}"

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if not self._authed():
            return self._send(401, {"error": "bad or missing bearer token"})
        state = self.server.state
        path, _, query = self.path.partition("?")
        if path == "/v1/status":
            return self._send(200, state.snapshot())
        if path == "/v1/work":
            max_items = 1
            for part in query.split("&"):
                if part.startswith("max="):
                    try:
                        max_items = max(1, min(int(part[4:]), 8))
                    except ValueError:
                        pass
            worker = self.headers.get("X-Worker", self.client_address[0])
            items, done = state.lease(worker, max_items)
            return self._send(200, {
                "done": done,
                "retry_seconds": RETRY_SECONDS,
                "items": [
                    {
                        "id": i.id,
                        "prompt": i.prompt,
                        "max_tokens": state.max_tokens,
                        "temperature": state.temperature,
                        "lease_seconds": state.lease_timeout,
                    }
                    for i in items
                ],
            })
        return self._send(404, {"error": f"no route {path}"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authed():
            return self._send(401, {"error": "bad or missing bearer token"})
        if self.path.partition("?")[0] != "/v1/results":
            return self._send(404, {"error": f"no route {self.path}"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            worker = str(payload.get("worker") or self.client_address[0])
            results = payload["results"]
            assert isinstance(results, list)
        except Exception:
            return self._send(400, {"error": "body must be JSON: "
                                    '{"worker": ..., "results": [{"id", "raw"}]}'})
        reply: Dict[str, Any] = {}
        for r in results:
            reply = self.server.state.ingest(worker, str(r.get("id")), str(r.get("raw", "")))
        return self._send(200, reply or self.server.state.snapshot())


class MeshServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr: Tuple[str, int], state: MeshState, token: str):
        super().__init__(addr, MeshHandler)
        self.state = state
        self.token = token


def lan_ip() -> str:
    """Best-effort LAN IP via the UDP-connect trick (no packets sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def run_mesh_serve(
    state: MeshState, host: str, port: int, token: str, linger: float = 30.0
) -> Dict[str, Any]:
    """Serve until the target is reached (plus a linger) or Ctrl-C."""
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table

    console = Console()
    server = MeshServer((host, port), state, token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def render() -> Table:
        snap = state.snapshot()
        elapsed = max(snap["elapsed_seconds"], 1)
        table = Table(title=f"troy mesh — {snap['records']}/{snap['target']} records")
        table.add_column("worker")
        table.add_column("records", justify="right")
        table.add_column("dupes", justify="right")
        table.add_column("last seen", justify="right")
        for name, w in snap["workers"].items():
            table.add_row(
                name, str(w["completed"]), str(w["duplicates"]),
                f"{time.time() - w['last_seen']:.0f}s ago",
            )
        q = snap["queue"]
        table.caption = (
            f"{snap['records'] / elapsed * 60:.1f} rec/min · "
            f"queue {q['available']} free / {q['leased']} leased · "
            f"{snap['empty_results']} empty results · {snap['out']}"
        )
        return table

    done_at: Optional[float] = None
    try:
        with Live(render(), console=console, refresh_per_second=2) as live:
            while True:
                time.sleep(0.5)
                live.update(render())
                if state.done:
                    done_at = done_at or time.time()
                    if not state.snapshot()["queue"]["leased"]:
                        break  # nothing in flight, no need to linger
                    if time.time() - done_at > linger:
                        break
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        state.close()
    return state.snapshot()


# -- Mac worker ---------------------------------------------------------


def _request(url: str, token: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import urllib.request

    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def run_mesh_join(
    url: str, token: str, model: str, worker_name: str, batch: int = 2
) -> Dict[str, Any]:
    """Pull work from a coordinator and generate with a local teacher."""
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    base = url.rstrip("/")
    print(f"Loading teacher {model} ...")
    lm, tokenizer = load(model)
    completed = 0
    backoff = 1.0
    last: Dict[str, Any] = {}

    while True:
        try:
            work = _request(f"{base}/v1/work?max={batch}", token)
            backoff = 1.0
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  coordinator unreachable ({e}); retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        if work.get("done"):
            break
        items = work.get("items", [])
        if not items:
            time.sleep(work.get("retry_seconds", RETRY_SECONDS))
            continue
        for item in items:
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                add_generation_prompt=True,
                return_dict=False,
            )
            raw = generate(
                lm,
                tokenizer,
                templated,
                max_tokens=item.get("max_tokens", 2048),
                sampler=make_sampler(temp=item.get("temperature", 0.8)),
            )
            last = _request(
                f"{base}/v1/results",
                token,
                {"worker": worker_name, "results": [{"id": item["id"], "raw": raw}]},
            )
            completed += 1
            print(
                f"  {item['id']}: +{last.get('accepted', 0)} "
                f"({last.get('records', '?')}/{last.get('target', '?')} total)"
            )
        if last.get("done"):
            break

    return {"completed": completed, **last}

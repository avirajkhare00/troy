"""Interactive chat with a trained model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load


def run_chat(
    model_path: str,
    adapter_path: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    prompt: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    system: Optional[str] = None,
) -> None:
    print(f"Loading {model_path} ...")
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    sampler = make_sampler(temp=temperature)
    seed_history = [{"role": "system", "content": system}] if system else []
    history = list(seed_history)

    def respond(user_text: str) -> None:
        history.append({"role": "user", "content": user_text})
        templated = tokenizer.apply_chat_template(
            history, tools=tools, add_generation_prompt=True, return_dict=False
        )
        reply = ""
        for response in stream_generate(
            model, tokenizer, templated, max_tokens=max_tokens, sampler=sampler
        ):
            print(response.text, end="", flush=True)
            reply += response.text
        print()
        history.append({"role": "assistant", "content": reply})

    if prompt is not None:  # one-shot mode
        respond(prompt)
        return

    print("Chat started. Type /exit to quit, /clear to reset history.\n")
    while True:
        try:
            user_text = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/clear":
            history[:] = list(seed_history)
            print("(history cleared)")
            continue
        respond(user_text)

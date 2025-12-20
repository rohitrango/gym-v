"""
Minimal gym-v evaluator (OpenAI-compatible `POST /v1/chat/completions`).

Required env vars:
- OPENAI_BASE_URL
- OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import urllib.request
from typing import Any

SYSTEM_PROMPT = """You are a game player. You need to explore and figure out the game rules based on the current situation, then make a move.
You must follow the following requirements:
- You must give the current action directly.
- You must ensure the output action is valid in the game, and avoid causing environment errors."""


def _env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"Missing env var: {name}")
    return v


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/chat/completions"


def _img_data_url_png(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> str:
    req = urllib.request.Request(
        _chat_url(base_url),
        data=json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.load(resp)
    # Be tolerant to minor OpenAI-compatible variations.
    choice0 = (data.get("choices") or [{}])[0]
    msg = choice0.get("message") or {}
    content: Any = msg.get("content")
    if content is None:
        content = msg.get("text") or choice0.get("text") or ""
    if isinstance(content, list):
        # e.g. [{type:"text", text:"..."}]
        parts = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        content = "".join(parts)
    if not isinstance(content, str):
        content = "" if content is None else str(content)

    content = content.strip()
    if content == "":
        print(
            f"[warn] empty model output (choices[0] keys={list(choice0.keys())})",
            file=sys.stderr,
        )
    return content


def _messages(*, env_desc: str, obs_text: str | None, obs_img, include_image: bool):
    user = []
    if env_desc.strip():
        user.append(f"Environment:\n{env_desc.strip()}")
    if obs_text:
        user.append(f"Observation:\n{str(obs_text).strip()}")
    user.append("Action:")
    user_str = "\n\n".join(user).strip()

    if include_image and obs_img is not None:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": _img_data_url_png(obs_img)},
                    },
                    {"type": "text", "text": "\n" + user_str},
                ],
            },
        ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_str},
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="TextArena/Sokoban-v0")
    ap.add_argument("--model", default="claude-sonnet-4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--no-image", action="store_true")
    args = ap.parse_args()

    import gym_v
    import gym_v.envs  # noqa: F401  # register built-in envs (TextArena/* etc)

    base_url = _env("OPENAI_BASE_URL")
    api_key = _env("OPENAI_API_KEY")
    model = args.model

    env = gym_v.make(args.env_id)
    obs, _ = env.reset(seed=args.seed)

    total_reward = 0.0
    for t in range(args.max_steps):
        action = _chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=_messages(
                env_desc=getattr(env, "description", "") or "",
                obs_text=getattr(obs, "text", None),
                obs_img=getattr(obs, "image", None),
                include_image=(not args.no_image),
            ),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
        )

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        print(
            f"[step={t}] reward={float(reward)} terminated={terminated} truncated={truncated}\n"
            f"[step={t}] action={action!r}\n"
        )
        if terminated or truncated:
            break

    print(f"[done] env_id={args.env_id} total_reward={total_reward}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

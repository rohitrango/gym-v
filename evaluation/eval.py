"""
Minimal gym-v evaluator (OpenAI-compatible `POST /v1/chat/completions`).

Required env vars:
- OPENAI_BASE_URL
- OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
import sys
import time
from typing import Any
import urllib.error
import urllib.request

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


def _run_one_episode(
    *,
    episode_idx: int,
    env_id: str,
    seed: int | None,
    max_steps: int,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    include_image: bool,
):
    import gym_v

    env = gym_v.make(env_id)
    obs, _ = env.reset(seed=seed)

    final_reward = 0.0
    steps = 0
    last_action = ""
    trajectory: list[dict[str, Any]] = []

    try:
        for t in range(max_steps):
            steps = t + 1
            last_action = _chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=_messages(
                    env_desc=getattr(env, "description", "") or "",
                    obs_text=getattr(obs, "text", None),
                    obs_img=getattr(obs, "image", None),
                    include_image=include_image,
                ),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )

            obs, reward, terminated, truncated, _ = env.step(last_action)
            final_reward += float(reward)
            trajectory.append(
                {
                    "t": t,
                    "action": last_action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )

            if terminated or truncated:
                return {
                    "_episode": episode_idx,
                    "seed": seed,
                    "steps": steps,
                    "final_reward": final_reward,
                    "trajectory": trajectory,
                }

        return {
            "_episode": episode_idx,
            "seed": seed,
            "steps": steps,
            "final_reward": final_reward,
            "trajectory": trajectory,
        }
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        return {
            "_episode": episode_idx,
            "seed": seed,
            "steps": steps,
            "final_reward": final_reward,
            "trajectory": trajectory,
            "error": f"{type(e).__name__}: {e}",
        }
    except Exception as e:
        return {
            "_episode": episode_idx,
            "seed": seed,
            "steps": steps,
            "final_reward": final_reward,
            "trajectory": trajectory,
            "error": f"{type(e).__name__}: {e}",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="TextArena/Sokoban-v0")
    ap.add_argument("--model", default="claude-sonnet-4")
    ap.add_argument("--num-episodes", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    import gym_v
    import gym_v.envs  # noqa: F401  # register built-in envs (TextArena/* etc)

    base_url = _env("OPENAI_BASE_URL")
    api_key = _env("OPENAI_API_KEY")
    model = args.model

    num_episodes = int(args.num_episodes)
    concurrency = max(1, int(args.concurrency))
    include_image = not bool(args.no_image)

    print(
        f"[run] env_id={args.env_id} model={model} num_episodes={num_episodes} concurrency={concurrency}"
    )

    started_at = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = []
        for i in range(num_episodes):
            seed_i = None if args.seed is None else int(args.seed) + i
            futures.append(
                ex.submit(
                    _run_one_episode,
                    episode_idx=i,
                    env_id=args.env_id,
                    seed=seed_i,
                    max_steps=int(args.max_steps),
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    temperature=float(args.temperature),
                    max_tokens=int(args.max_tokens),
                    timeout_s=float(args.timeout_s),
                    include_image=include_image,
                )
            )

        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if "error" in r:
                print(
                    f"[episode={r['_episode']}] seed={r['seed']} steps={r['steps']} final_reward={r['final_reward']} error={r['error']}"
                )
            else:
                print(
                    f"[episode={r['_episode']}] seed={r['seed']} steps={r['steps']} final_reward={r['final_reward']}"
                )

    avg = sum(float(x.get("final_reward", 0.0)) for x in results) / max(1, len(results))
    episodes_out = []
    for r in sorted(results, key=lambda x: int(x.get("_episode", 0))):
        episodes_out.append(
            {
                "seed": r.get("seed"),
                "steps": r.get("steps"),
                "final_reward": r.get("final_reward", 0.0),
                "trajectory": r.get("trajectory", []),
                **({"error": r["error"]} if "error" in r else {}),
            }
        )
    record = {
        "config": {
            "env_id": args.env_id,
            "model": model,
            "num_episodes": num_episodes,
            "concurrency": concurrency,
            "seed": args.seed,
            "max_steps": int(args.max_steps),
            "timeout_s": float(args.timeout_s),
            "temperature": float(args.temperature),
            "max_tokens": int(args.max_tokens),
            "include_image": bool(include_image),
        },
        "episodes": episodes_out,
        "avg_final_reward": avg,
        "elapsed_s": time.time() - started_at,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"[done] episodes={len(results)} avg_final_reward={avg} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

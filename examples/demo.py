"""Interactive gym-v environment viewer."""

from __future__ import annotations

import ast
import shutil
import sys
import tkinter as tk
from textwrap import dedent
from typing import Any

import click
from PIL import ImageTk
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

import gym_v
from gym_v.logger import get_logger

logger = get_logger()


def parse_env_args(env_args: tuple[str, ...]) -> dict[str, Any]:
    """Parse key=value arguments."""
    if not env_args:
        return {}

    kwargs = {}
    for arg in env_args:
        if "=" not in arg:
            continue
        key, raw_value = arg.split("=", 1)

        try:
            value = ast.literal_eval(raw_value.strip())
        except (ValueError, SyntaxError):
            value = raw_value.strip()

        kwargs[key.strip()] = value
    return kwargs


@click.command()
@click.option("--id", "env_id", default="TextArena/Sokoban-v0", show_default=True)
@click.option("--env-args", "env_args", multiple=True)
def main(env_id: str, env_args: tuple[str, ...]):
    root = tk.Tk()
    root.title(f"gym-v: {env_id}")
    root.attributes("-topmost", True)

    image_label = tk.Label(root)
    image_label.pack()

    session = PromptSession(history=InMemoryHistory())

    kwargs = parse_env_args(env_args)
    try:
        env = gym_v.make(env_id, **kwargs)
    except Exception as e:
        logger.error(f"Failed to create environment({env_id}): {e}")
        sys.exit(1)

    is_game_over = False

    logger.info(f"Environment {env_id} created.")
    logger.info("Controls: Type command and Enter.")
    logger.info("  - 'reset' or 'r: Reset environment")
    logger.info("  - 'quit' or 'q': Exit")

    logger.info(f"\nEnv Description: {env.description}")

    obs, info = env.reset()
    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    logger.info(
        dedent(f"""
        {f" Step info(step_count={env._current_episode_steps}) ".center(width, "=")}
        observation (text): {obs.text}
        info: {info}
        {"=" * width}""")
    )

    running = True
    while running:
        try:
            tk_image = ImageTk.PhotoImage(obs.image)

            image_label.configure(image=tk_image)
            image_label.image = tk_image
            root.update()

        except tk.TclError:
            break

        prompt = "[Game Over] Type 'r' to reset >>> " if is_game_over else ">>> "

        try:
            action = session.prompt(prompt)
        except (EOFError, KeyboardInterrupt):
            break

        if not action:
            continue

        action_lower = action.lower()

        if action_lower in ("quit", "q"):
            running = False
            break

        elif action_lower in ("reset", "r"):
            obs, _ = env.reset()
            is_game_over = False
            logger.info(f"Environment({env_id}) Reset")
            continue

        if is_game_over:
            logger.info("Game over. Type 'r' to reset.")
            continue

        obs, reward, terminated, truncated, info = env.step(action)

        width = shutil.get_terminal_size(fallback=(80, 20)).columns
        logger.info(
            dedent(f"""
            {f" Step info(step_count={env._current_episode_steps}) ".center(width, "=")}
            observation (text): {obs.text}
            reward: {reward}
            terminated: {terminated}
            truncated: {truncated}
            info: {info}
            {"=" * width}""")
        )

        if terminated or truncated:
            is_game_over = True
            logger.info("Game over. Type 'r' to reset.")

    try:
        root.destroy()
    except tk.TclError:
        pass
    env.close()


if __name__ == "__main__":
    main()

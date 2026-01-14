"""Interactive gym-v environment viewer."""

from __future__ import annotations

import ast
import shutil
import sys
from textwrap import dedent
import tkinter as tk
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
@click.option("--agent", "agent_id", default="agent_0", help="The agent to control")
def main(env_id: str, env_args: tuple[str, ...], agent_id: str):
    root = tk.Tk()
    root.title(f"gym-v: {env_id} ({agent_id})")
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
    logger.info(f"Controlling Agent: {agent_id}")
    logger.info("Controls: Type command and Enter.")
    logger.info("  - 'reset' or 'r: Reset environment")
    logger.info("  - 'quit' or 'q': Exit")

    logger.info(f"\nEnv Description: {env.description}")

    # Handle Multi-Agent Reset
    obs_dict, info_dict = env.reset()
    if agent_id not in obs_dict:
        logger.error(
            f"Agent {agent_id} not found in observation dict: {obs_dict.keys()}"
        )
        sys.exit(1)

    obs = obs_dict[agent_id]
    info = info_dict.get(agent_id, {})

    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    logger.info(
        dedent(f"""
        {f" Step info(step_count={env.get_wrapper_attr('_current_episode_steps')}) ".center(width, "=")}
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

        prompt = (
            "[Game Over] Type 'r' to reset >>> "
            if is_game_over
            else f"[{agent_id}] >>> "
        )

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
            obs_dict, info_dict = env.reset()
            obs = obs_dict[agent_id]
            is_game_over = False
            logger.info(f"Environment({env_id}) Reset")
            continue

        if is_game_over:
            logger.info("Game over. Type 'r' to reset.")
            continue

        # Handle Multi-Agent Step
        action_dict = {agent_id: action}
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(
            action_dict
        )

        obs = obs_dict[agent_id]
        reward = reward_dict.get(agent_id, 0)
        terminated = terminated_dict.get(agent_id, False)
        truncated = truncated_dict.get(agent_id, False)
        env_done = terminated_dict.get("__all__", False) or truncated_dict.get(
            "__all__", False
        )
        info = info_dict.get(agent_id, {})

        width = shutil.get_terminal_size(fallback=(80, 20)).columns
        logger.info(
            dedent(f"""
            {f" Step info(step_count={env.get_wrapper_attr('_current_episode_steps')}) ".center(width, "=")}
            observation (text): {obs.text}
            reward: {reward}
            terminated: {terminated}
            truncated: {truncated}
            info: {info}
            {"=" * width}""")
        )

        if env_done:
            is_game_over = True
            logger.info("Game over (Env terminated/truncated). Type 'r' to reset.")

    try:
        root.destroy()
    except tk.TclError:
        pass
    env.close()


if __name__ == "__main__":
    main()

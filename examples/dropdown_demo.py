"""Tkinter gym-v environment viewer with a combobox and editable parameters."""

from __future__ import annotations

import argparse
import ast
import copy
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        dest="groups",
        help=(
            "Only show environments in this top-level group, such as Puzzles or "
            "Logic. Can be passed more than once."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parse_args()
    raise SystemExit


from PIL import Image, ImageTk

import gym_v
import gym_v.envs  # noqa: F401 - registers built-in environments

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WATCH_PATHS = (PROJECT_ROOT / "gym_v", PROJECT_ROOT / "examples")
WATCH_SUFFIXES = {
    ".py",
    ".json",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
}
WATCH_INTERVAL_MS = 750
RELOAD_DEBOUNCE_SECONDS = 0.5


def value_to_text(value: Any) -> str:
    """Render a default value as editable text."""
    if isinstance(value, dict):
        return json.dumps(value, indent=2, sort_keys=True)
    return repr(value)


def parse_value(raw_value: str) -> Any:
    """Parse a parameter value from a text entry."""
    value = raw_value.strip()
    if value == "":
        return None

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def first_observation(obs_dict: dict[str, Any]) -> tuple[str | None, Any | None]:
    """Return the first active agent observation."""
    if not obs_dict:
        return None, None
    agent_id = next(iter(obs_dict))
    return agent_id, obs_dict[agent_id]


def env_group(env_id: str) -> str:
    """Return the leading group for an environment id."""
    return env_id.split("/", 1)[0] if "/" in env_id else ""


def matching_env_ids(groups: set[str] | None = None) -> list[str]:
    """Return registered environment ids, optionally filtered by group."""
    env_ids = sorted(gym_v.registry)
    if not groups:
        return env_ids

    normalized_groups = {group.casefold() for group in groups}
    return [
        env_id
        for env_id in env_ids
        if env_group(env_id).casefold() in normalized_groups
    ]


def source_snapshot() -> dict[str, tuple[int, int]]:
    """Capture source file mtimes and sizes for cheap change detection."""
    snapshot: dict[str, tuple[int, int]] = {}

    for watch_path in WATCH_PATHS:
        if not watch_path.exists():
            continue
        for path in watch_path.rglob("*"):
            if (
                not path.is_file()
                or path.suffix.lower() not in WATCH_SUFFIXES
                or "__pycache__" in path.parts
            ):
                continue

            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[str(path)] = (stat.st_mtime_ns, stat.st_size)

    return snapshot


def reload_gym_v() -> None:
    """Reload local gym-v modules so changed environment code is used."""
    global gym_v

    importlib.invalidate_caches()
    for module_name in sorted(
        [name for name in sys.modules if name == "gym_v" or name.startswith("gym_v.")]
    ):
        del sys.modules[module_name]

    gym_v = importlib.import_module("gym_v")
    importlib.import_module("gym_v.envs")


class ParameterPanel(ttk.Frame):
    """Scrollable parameter editor generated from a selected environment spec."""

    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self._variables: dict[str, tk.StringVar] = {}
        self._text_widgets: dict[str, tk.Text] = {}

        self.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(self, highlightthickness=0)
        self._scrollbar = ttk.Scrollbar(
            self, orient=tk.VERTICAL, command=self._canvas.yview
        )
        self._body = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window(
            (0, 0), window=self._body, anchor=tk.NW
        )

        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self._scrollbar.grid(row=0, column=1, sticky=tk.NS)

        self._body.bind("<Configure>", self._sync_scroll_region)
        self._canvas.bind("<Configure>", self._sync_canvas_width)

    def set_parameters(self, kwargs: dict[str, Any]) -> None:
        for child in self._body.winfo_children():
            child.destroy()

        self._variables.clear()
        self._text_widgets.clear()

        if not kwargs:
            ttk.Label(self._body, text="No registered parameters.").grid(
                row=0, column=0, sticky=tk.W, padx=8, pady=8
            )
            return

        self._body.columnconfigure(1, weight=1)
        for row, (name, value) in enumerate(kwargs.items()):
            ttk.Label(self._body, text=name).grid(
                row=row, column=0, sticky=tk.NW, padx=(8, 12), pady=6
            )

            if isinstance(value, dict):
                text = tk.Text(self._body, height=5, width=34, wrap=tk.NONE)
                text.insert("1.0", value_to_text(value))
                text.grid(row=row, column=1, sticky=tk.EW, padx=(0, 8), pady=4)
                self._text_widgets[name] = text
            else:
                variable = tk.StringVar(value=value_to_text(value))
                entry = ttk.Entry(self._body, textvariable=variable)
                entry.grid(row=row, column=1, sticky=tk.EW, padx=(0, 8), pady=4)
                self._variables[name] = variable

    def get_parameters(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}

        for name, variable in self._variables.items():
            kwargs[name] = parse_value(variable.get())

        for name, text in self._text_widgets.items():
            kwargs[name] = parse_value(text.get("1.0", tk.END))

        return kwargs

    def _sync_scroll_region(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox(tk.ALL))

    def _sync_canvas_width(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)


class EnvironmentViewer:
    """Dropdown-based viewer for registered visual environments."""

    def __init__(self, root: tk.Tk, *, groups: set[str] | None = None):
        self.root = root
        self.root.title("gym-v environment viewer")
        self.root.geometry("1100x760")
        self.groups = groups

        self.env = None
        self.current_agent_id: str | None = None
        self.current_obs = None
        self.env_done = False
        self.tk_image: ImageTk.PhotoImage | None = None
        self._source_snapshot = source_snapshot()
        self._pending_reload_at: float | None = None

        self.env_ids = matching_env_ids(self.groups)
        if not self.env_ids:
            available = ", ".join(
                sorted({env_group(env_id) for env_id in gym_v.registry})
            )
            filter_text = ", ".join(sorted(self.groups or []))
            raise RuntimeError(
                f"No environments are registered for group filter: {filter_text}. "
                f"Available groups: {available}"
            )

        self.selected_env = tk.StringVar(value=self.env_ids[0])
        self.status = tk.StringVar(value="")
        self.action = tk.StringVar(value="")

        self._build_layout()
        self._select_env(load_env=True)
        self.root.after(WATCH_INTERVAL_MS, self._poll_source_changes)

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self.root, padding=12)
        sidebar.grid(row=0, column=0, sticky=tk.NS)
        sidebar.rowconfigure(3, weight=1)

        ttk.Label(sidebar, text="Environment").grid(row=0, column=0, sticky=tk.W)
        self.env_combo = ttk.Combobox(
            sidebar,
            textvariable=self.selected_env,
            values=self.env_ids,
            state="readonly",
            width=38,
        )
        self.env_combo.grid(row=1, column=0, sticky=tk.EW, pady=(4, 12))
        self.env_combo.bind("<<ComboboxSelected>>", lambda _event: self._select_env())

        button_row = ttk.Frame(sidebar)
        button_row.grid(row=2, column=0, sticky=tk.EW, pady=(0, 8))
        button_row.columnconfigure((0, 1), weight=1)

        ttk.Button(button_row, text="Apply", command=self._load_selected_env).grid(
            row=0, column=0, sticky=tk.EW, padx=(0, 4)
        )
        ttk.Button(button_row, text="Defaults", command=self._reset_parameters).grid(
            row=0, column=1, sticky=tk.EW, padx=(4, 0)
        )

        self.parameter_panel = ParameterPanel(sidebar)
        self.parameter_panel.grid(row=3, column=0, sticky=tk.NSEW)

        main = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        main.grid(row=0, column=1, sticky=tk.NSEW)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(main, anchor=tk.CENTER)
        self.image_label.grid(row=0, column=0, sticky=tk.NSEW)

        controls = ttk.Frame(main)
        controls.grid(row=1, column=0, sticky=tk.EW, pady=(12, 0))
        controls.columnconfigure(1, weight=1)

        ttk.Button(controls, text="Reset", command=self._reset_env).grid(
            row=0, column=0, sticky=tk.W
        )
        action_entry = ttk.Entry(controls, textvariable=self.action)
        action_entry.grid(row=0, column=1, sticky=tk.EW, padx=8)
        action_entry.bind("<Return>", lambda _event: self._step_env())
        ttk.Button(controls, text="Step", command=self._step_env).grid(
            row=0, column=2, sticky=tk.E
        )

        self.text_output = scrolledtext.ScrolledText(
            main, height=6, wrap=tk.WORD, state=tk.DISABLED
        )
        self.text_output.grid(row=2, column=0, sticky=tk.EW, pady=(12, 0))

        ttk.Label(main, text="Observation text").grid(
            row=3, column=0, sticky=tk.W, pady=(12, 0)
        )
        self.obs_text_output = scrolledtext.ScrolledText(
            main, height=8, wrap=tk.WORD, state=tk.DISABLED
        )
        self.obs_text_output.grid(row=4, column=0, sticky=tk.EW, pady=(4, 0))

        ttk.Label(main, text="Input prompt").grid(
            row=5, column=0, sticky=tk.W, pady=(12, 0)
        )
        self.prompt_output = scrolledtext.ScrolledText(
            main, height=6, wrap=tk.WORD, state=tk.DISABLED
        )
        self.prompt_output.grid(row=6, column=0, sticky=tk.EW, pady=(4, 0))

        ttk.Label(main, textvariable=self.status).grid(
            row=7, column=0, sticky=tk.W, pady=(6, 0)
        )

    def _select_env(self, *, load_env: bool = True) -> None:
        self._close_env()
        self._reset_parameters()
        self._clear_output()
        self._set_image(None)
        self.status.set("Edit parameters and click Apply.")
        if load_env:
            self._load_selected_env()

    def _reset_parameters(self) -> None:
        spec = gym_v.registry[self.selected_env.get()]
        self.parameter_panel.set_parameters(copy.deepcopy(spec.kwargs))

    def _load_selected_env(self, *, show_error: bool = True) -> bool:
        env_id = self.selected_env.get()
        kwargs = self.parameter_panel.get_parameters()

        self._close_env()
        try:
            self.env = gym_v.make(env_id, **kwargs)
            self.status.set(f"Loaded {env_id}")
            return self._reset_env(show_error=show_error)
        except Exception as exc:
            self.env = None
            self.current_agent_id = None
            self.current_obs = None
            self._set_image(None)
            self.status.set(f"Failed to load {env_id}")
            if show_error:
                messagebox.showerror("Failed to load environment", str(exc))
            else:
                self._set_text(f"Failed to load {env_id}\n\n{exc}\n")
            return False

    def _reset_env(self, *, show_error: bool = True) -> bool:
        if self.env is None:
            return False

        try:
            obs_dict, info_dict = self.env.reset()
        except Exception as exc:
            if show_error:
                messagebox.showerror("Reset failed", str(exc))
            else:
                self._set_text(f"Reset failed\n\n{exc}\n")
            return False

        self.env_done = False
        self.current_agent_id, self.current_obs = first_observation(obs_dict)
        self.action.set("")
        self._show_current_observation(info=info_dict.get(self.current_agent_id, {}))
        return True

    def _step_env(self) -> None:
        if self.env is None or self.current_agent_id is None:
            return
        if self.env_done:
            self.status.set("Environment is done. Reset to continue.")
            return

        action = self.action.get()
        if not action:
            return

        try:
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = (
                self.env.step({self.current_agent_id: action})
            )
        except Exception as exc:
            messagebox.showerror("Step failed", str(exc))
            return

        self.action.set("")
        done = terminated_dict.get("__all__", False) or truncated_dict.get(
            "__all__", False
        )
        if done:
            self.env_done = True
            self.status.set("Environment is done.")
            self._append_text("Environment is done.\n")
            self._append_text(f"Rewards: {reward_dict}\n")
            self._append_text(f"Info: {info_dict}\n")
            return

        self.current_agent_id, self.current_obs = first_observation(obs_dict)
        info = info_dict.get(self.current_agent_id, {})
        reward = reward_dict.get(self.current_agent_id, 0.0)
        self._show_current_observation(reward=reward, info=info)

    def _show_current_observation(
        self, *, reward: float | None = None, info: dict[str, Any] | None = None
    ) -> None:
        if self.current_obs is None:
            self._set_image(None)
            self._set_text("No observation returned.\n")
            self._set_obs_text("")
            self._set_prompt_text("")
            return

        self._set_image(getattr(self.current_obs, "image", None))
        metadata = getattr(self.current_obs, "metadata", {})
        text_prompt = metadata.get("text_prompt") if isinstance(metadata, dict) else None

        lines = [f"Current player: {self.current_agent_id}"]
        if reward is not None:
            lines.append(f"Reward: {reward}")
        if info:
            lines.append(f"Info: {info}")

        self._set_text("\n".join(lines) + "\n")
        obs_text = getattr(self.current_obs, "text", None)
        self._set_obs_text(str(obs_text) if obs_text is not None else "")
        self._set_prompt_text(str(text_prompt) if text_prompt else "")

    def _set_image(self, image: Image.Image | list[Image.Image] | None) -> None:
        if isinstance(image, list):
            image = image[0] if image else None

        if image is None:
            self.tk_image = None
            self.image_label.configure(image="", text="No image")
            return

        display = image.copy()
        display.thumbnail((780, 520), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(display)
        self.image_label.configure(image=self.tk_image, text="")

    def _set_text(self, text: str) -> None:
        self.text_output.configure(state=tk.NORMAL)
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, text)
        self.text_output.configure(state=tk.DISABLED)

    def _set_obs_text(self, text: str) -> None:
        self.obs_text_output.configure(state=tk.NORMAL)
        self.obs_text_output.delete("1.0", tk.END)
        self.obs_text_output.insert(tk.END, text)
        self.obs_text_output.configure(state=tk.DISABLED)

    def _set_prompt_text(self, text: str) -> None:
        self.prompt_output.configure(state=tk.NORMAL)
        self.prompt_output.delete("1.0", tk.END)
        self.prompt_output.insert(tk.END, text)
        self.prompt_output.configure(state=tk.DISABLED)

    def _append_text(self, text: str) -> None:
        self.text_output.configure(state=tk.NORMAL)
        self.text_output.insert(tk.END, text)
        self.text_output.see(tk.END)
        self.text_output.configure(state=tk.DISABLED)

    def _clear_output(self) -> None:
        self._set_text("")
        self._set_obs_text("")
        self._set_prompt_text("")

    def _close_env(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None
        self.current_agent_id = None
        self.current_obs = None
        self.env_done = False

    def _poll_source_changes(self) -> None:
        try:
            current_snapshot = source_snapshot()
            now = time.monotonic()

            if current_snapshot != self._source_snapshot:
                self._source_snapshot = current_snapshot
                self._pending_reload_at = now + RELOAD_DEBOUNCE_SECONDS
                self.status.set("Source changed. Waiting for edits to settle...")

            if self._pending_reload_at is not None and now >= self._pending_reload_at:
                self._pending_reload_at = None
                self._reload_after_source_change()
        finally:
            self.root.after(WATCH_INTERVAL_MS, self._poll_source_changes)

    def _reload_after_source_change(self) -> None:
        selected_env = self.selected_env.get()
        current_kwargs = self.parameter_panel.get_parameters()

        self.status.set("Reloading gym-v source...")
        self._close_env()

        try:
            reload_gym_v()
            self._refresh_env_ids(selected_env)
            if self.selected_env.get() == selected_env:
                self.parameter_panel.set_parameters(current_kwargs)
            self._load_selected_env(show_error=False)
            self.status.set(f"Reloaded {self.selected_env.get()} after source change")
        except Exception as exc:
            self.env = None
            self.current_agent_id = None
            self.current_obs = None
            self._set_image(None)
            self.status.set("Reload failed")
            self._set_text(f"Reload failed\n\n{exc}\n")

    def _refresh_env_ids(self, preferred_env_id: str) -> None:
        self.env_ids = matching_env_ids(self.groups)
        if not self.env_ids:
            raise RuntimeError("No environments are registered after reload.")

        self.env_combo.configure(values=self.env_ids)
        if preferred_env_id in self.env_ids:
            self.selected_env.set(preferred_env_id)
        else:
            self.selected_env.set(self.env_ids[0])
            self._reset_parameters()


def main() -> None:
    args = parse_args()
    groups = set(args.groups) if args.groups else None

    root = tk.Tk()
    EnvironmentViewer(root, groups=groups)
    root.mainloop()


if __name__ == "__main__":
    main()

"""Parametric Curve perception environment."""

from __future__ import annotations

from collections.abc import Callable
import io
import json
import logging
import random
from textwrap import dedent
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

from gym_v import Env, Observation, get_logger

logger = get_logger()

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


class ParametricCurveEnv(Env):
    # Meta: source=Perception, category=perception, turn=single
    """Parametric Curve perception environment.

    The agent must perceive a parametric curve (x(t), y(t)) and
    extract the mathematical expressions that generate it.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (640, 480),
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._current_x_expr: str | None = None
        self._current_y_expr: str | None = None
        self._current_x_func: Callable | None = None
        self._current_y_func: Callable | None = None
        self._current_t_range: tuple[float, float] | None = None
        self._current_curve_type: str | None = None
        self._current_image: Image.Image | None = None

        self._line_colors = [
            "#FF0000",
            "#0000FF",
            "#00AA00",
            "#FF6600",
            "#9900CC",
            "#00CCCC",
            "#CC0066",
            "#006699",
        ]

    @property
    def description(self) -> str:
        return dedent("""
            You are given a parametric curve defined by x(t) and y(t).
            Your task is to identify the parametric equations.

            The curve could be:
            - Circle: x = r*cos(t), y = r*sin(t)
            - Ellipse: x = a*cos(t), y = b*sin(t)
            - Lissajous: x = A*sin(a*t), y = B*sin(b*t + c)
            - Cycloid: x = r*(t - sin(t)), y = r*(1 - cos(t))
            - Cardioid (parametric form)
            - Spiral: x = t*cos(t), y = t*sin(t)

            Output the expressions using Python syntax.

            Example: {"x": "cos(t)", "y": "sin(t)", "t_range": [0, 6.28], "type": "circle"}
            Example: {"x": "sin(2*t)", "y": "sin(3*t)", "t_range": [0, 6.28], "type": "lissajous"}

            Output ONLY the JSON string.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self._generate_new_problem()

        obs = Observation(
            image=self._current_image,
            text=None,
            metadata={
                "curve_type": self._current_curve_type,
            },
        )

        info = {
            "oracle_answer": json.dumps(
                {
                    "x": self._current_x_expr,
                    "y": self._current_y_expr,
                    "t_range": list(self._current_t_range),
                    "type": self._current_curve_type,
                }
            ),
        }

        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]
        reward = self._compute_reward(action_str)
        info = {
            "oracle_answer": json.dumps(
                {
                    "x": self._current_x_expr,
                    "y": self._current_y_expr,
                    "t_range": list(self._current_t_range),
                    "type": self._current_curve_type,
                }
            ),
        }

        obs = Observation(image=self._current_image, text=None)

        return (
            {agent_id: obs for agent_id in self._agent_ids},
            {agent_id: reward for agent_id in self._agent_ids},
            {
                **{agent_id: True for agent_id in self._agent_ids},
                "__all__": True,
            },
            {
                **{agent_id: False for agent_id in self._agent_ids},
                "__all__": False,
            },
            {agent_id: info for agent_id in self._agent_ids},
        )

    def _compute_reward(self, action: str) -> float:
        """Compute reward by comparing action with oracle answer."""
        try:
            parsed_action = json.loads(action)
            oracle = {
                "x": self._current_x_expr,
                "y": self._current_y_expr,
                "t_range": list(self._current_t_range),
                "type": self._current_curve_type,
            }
            if parsed_action == oracle:
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    def render(self) -> Image.Image:
        return self._current_image

    def _generate_new_problem(self):
        """Generate a random parametric curve and render it."""
        curve_types = [
            "circle",
            "ellipse",
            "lissajous",
            "cycloid",
            "spiral",
            "cardioid",
            "astroid",
            "epicycloid",
        ]
        self._current_curve_type = random.choice(curve_types)

        if self._current_curve_type == "circle":
            self._generate_circle()
        elif self._current_curve_type == "ellipse":
            self._generate_ellipse()
        elif self._current_curve_type == "lissajous":
            self._generate_lissajous()
        elif self._current_curve_type == "cycloid":
            self._generate_cycloid()
        elif self._current_curve_type == "spiral":
            self._generate_spiral()
        elif self._current_curve_type == "cardioid":
            self._generate_cardioid()
        elif self._current_curve_type == "astroid":
            self._generate_astroid()
        elif self._current_curve_type == "epicycloid":
            self._generate_epicycloid()

        self._current_image = self._render_curve()

    def _generate_circle(self):
        """Generate circle: x = r*cos(t), y = r*sin(t)"""
        r = random.choice([1, 2, 3])

        if r == 1:
            self._current_x_expr = "cos(t)"
            self._current_y_expr = "sin(t)"
        else:
            self._current_x_expr = f"{r}*cos(t)"
            self._current_y_expr = f"{r}*sin(t)"

        self._current_x_func = lambda t, r=r: r * np.cos(t)
        self._current_y_func = lambda t, r=r: r * np.sin(t)
        self._current_t_range = (0, 2 * np.pi)

    def _generate_ellipse(self):
        """Generate ellipse: x = a*cos(t), y = b*sin(t)"""
        a = random.choice([1, 2, 3])
        b = random.choice([1, 2, 3])
        while a == b:
            b = random.choice([1, 2, 3])

        if a == 1:
            self._current_x_expr = "cos(t)"
        else:
            self._current_x_expr = f"{a}*cos(t)"

        if b == 1:
            self._current_y_expr = "sin(t)"
        else:
            self._current_y_expr = f"{b}*sin(t)"

        self._current_x_func = lambda t, a=a: a * np.cos(t)
        self._current_y_func = lambda t, b=b: b * np.sin(t)
        self._current_t_range = (0, 2 * np.pi)

    def _generate_lissajous(self):
        """Generate Lissajous curve: x = sin(a*t), y = sin(b*t)"""
        a = random.choice([1, 2, 3])
        b = random.choice([2, 3, 4, 5])
        while a == b:
            b = random.choice([2, 3, 4, 5])

        if a == 1:
            self._current_x_expr = "sin(t)"
        else:
            self._current_x_expr = f"sin({a}*t)"

        if b == 1:
            self._current_y_expr = "sin(t)"
        else:
            self._current_y_expr = f"sin({b}*t)"

        self._current_x_func = lambda t, a=a: np.sin(a * t)
        self._current_y_func = lambda t, b=b: np.sin(b * t)
        self._current_t_range = (0, 2 * np.pi)

    def _generate_cycloid(self):
        """Generate cycloid: x = r*(t - sin(t)), y = r*(1 - cos(t))"""
        r = random.choice([1, 2])

        if r == 1:
            self._current_x_expr = "t - sin(t)"
            self._current_y_expr = "1 - cos(t)"
        else:
            self._current_x_expr = f"{r}*(t - sin(t))"
            self._current_y_expr = f"{r}*(1 - cos(t))"

        self._current_x_func = lambda t, r=r: r * (t - np.sin(t))
        self._current_y_func = lambda t, r=r: r * (1 - np.cos(t))
        self._current_t_range = (0, 4 * np.pi)

    def _generate_spiral(self):
        """Generate spiral: x = t*cos(t), y = t*sin(t)"""
        a = random.choice([0.5, 1])

        if a == 1:
            self._current_x_expr = "t*cos(t)"
            self._current_y_expr = "t*sin(t)"
        else:
            self._current_x_expr = f"{a}*t*cos(t)"
            self._current_y_expr = f"{a}*t*sin(t)"

        self._current_x_func = lambda t, a=a: a * t * np.cos(t)
        self._current_y_func = lambda t, a=a: a * t * np.sin(t)
        self._current_t_range = (0, 4 * np.pi)

    def _generate_cardioid(self):
        """Generate cardioid: x = 2*cos(t) - cos(2*t), y = 2*sin(t) - sin(2*t)"""
        self._current_x_expr = "2*cos(t) - cos(2*t)"
        self._current_y_expr = "2*sin(t) - sin(2*t)"

        self._current_x_func = lambda t: 2 * np.cos(t) - np.cos(2 * t)
        self._current_y_func = lambda t: 2 * np.sin(t) - np.sin(2 * t)
        self._current_t_range = (0, 2 * np.pi)

    def _generate_astroid(self):
        """Generate astroid: x = cos^3(t), y = sin^3(t)"""
        a = random.choice([1, 2])

        if a == 1:
            self._current_x_expr = "cos(t)**3"
            self._current_y_expr = "sin(t)**3"
        else:
            self._current_x_expr = f"{a}*cos(t)**3"
            self._current_y_expr = f"{a}*sin(t)**3"

        self._current_x_func = lambda t, a=a: a * np.cos(t) ** 3
        self._current_y_func = lambda t, a=a: a * np.sin(t) ** 3
        self._current_t_range = (0, 2 * np.pi)

    def _generate_epicycloid(self):
        """Generate epicycloid with small integer ratio."""
        R = 2  # Outer circle radius
        r = random.choice([1])  # Inner circle radius

        self._current_x_expr = f"({R}+{r})*cos(t) - {r}*cos(({R}+{r})*t/{r})"
        self._current_y_expr = f"({R}+{r})*sin(t) - {r}*sin(({R}+{r})*t/{r})"

        # Simplify expression
        self._current_x_expr = "3*cos(t) - cos(3*t)"
        self._current_y_expr = "3*sin(t) - sin(3*t)"

        self._current_x_func = lambda t: 3 * np.cos(t) - np.cos(3 * t)
        self._current_y_func = lambda t: 3 * np.sin(t) - np.sin(3 * t)
        self._current_t_range = (0, 2 * np.pi)

    def _render_curve(self) -> Image.Image:
        """Render the parametric curve as a PIL Image."""
        fig, ax = plt.subplots(
            figsize=(self.img_size[0] / 100, self.img_size[1] / 100), dpi=100
        )

        # Generate t values
        t = np.linspace(self._current_t_range[0], self._current_t_range[1], 1000)

        # Compute x and y values
        x = self._current_x_func(t)
        y = self._current_y_func(t)

        # Choose line style
        color = random.choice(self._line_colors)
        linewidth = random.uniform(2, 3)

        ax.plot(x, y, color=color, linewidth=linewidth)

        # Add grid
        if random.random() < 0.7:
            ax.grid(True, linestyle="--", alpha=0.5)

        # Add axes through origin
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.axvline(x=0, color="black", linewidth=0.8)

        # Set labels
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)

        # Add title
        if random.random() < 0.7:
            ax.set_title(
                "Parametric Curve: (x(t), y(t))", fontsize=14, fontweight="bold"
            )

        ax.set_aspect("equal")
        plt.tight_layout()

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close(fig)

        # Optional noise
        if random.random() < 0.1:
            img = self._add_noise(img)

        return img

    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Add subtle noise or blur to the image."""
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        return img

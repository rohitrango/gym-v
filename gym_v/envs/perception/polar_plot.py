"""Polar Plot perception environment."""

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


class PerceptionPolarPlotEnv(Env):
    """Polar Plot perception environment.

    The agent must perceive a polar plot r = f(theta) and
    extract the mathematical expression that generates it.
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
        self._current_expression: str | None = None
        self._current_func: Callable | None = None
        self._current_func_type: str | None = None
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
            You are given a polar plot of r = f(theta).
            Your task is to identify the polar equation.

            The function could be:
            - Circle: r = a (constant)
            - Rose: r = a*cos(n*theta) or r = a*sin(n*theta)
            - Spiral: r = a*theta (Archimedean)
            - Cardioid: r = a*(1 + cos(theta))
            - Lemniscate: r^2 = a*cos(2*theta)
            - Limaçon: r = a + b*cos(theta)

            Output the expression using theta as the angle variable.

            Example: {"expression": "2*cos(3*theta)", "type": "rose", "petals": 3}
            Example: {"expression": "1 + cos(theta)", "type": "cardioid"}

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
                "func_type": self._current_func_type,
            },
        )

        info = {
            "oracle_answer": json.dumps(self._current_answer),
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
            "oracle_answer": json.dumps(self._current_answer),
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
            if parsed_action == self._current_answer:
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    def render(self) -> Image.Image:
        return self._current_image

    def _generate_new_problem(self):
        """Generate a random polar function and render its plot."""
        func_types = ["circle", "rose_cos", "rose_sin", "spiral", "cardioid", "limacon"]
        self._current_func_type = random.choice(func_types)

        if self._current_func_type == "circle":
            self._generate_circle()
        elif self._current_func_type == "rose_cos":
            self._generate_rose_cos()
        elif self._current_func_type == "rose_sin":
            self._generate_rose_sin()
        elif self._current_func_type == "spiral":
            self._generate_spiral()
        elif self._current_func_type == "cardioid":
            self._generate_cardioid()
        elif self._current_func_type == "limacon":
            self._generate_limacon()

        self._current_image = self._render_polar()

    def _generate_circle(self):
        """Generate circle: r = a"""
        a = random.choice([1, 2, 3])
        self._current_expression = str(a)
        self._current_func = lambda theta, a=a: np.full_like(theta, a)
        self._current_answer = {"expression": str(a), "type": "circle"}
        self._current_func_type = "circle"

    def _generate_rose_cos(self):
        """Generate rose curve: r = a*cos(n*theta)"""
        a = random.choice([1, 2, 3])
        n = random.choice([2, 3, 4, 5])

        if a == 1:
            if n == 1:
                self._current_expression = "cos(theta)"
            else:
                self._current_expression = f"cos({n}*theta)"
        else:
            if n == 1:
                self._current_expression = f"{a}*cos(theta)"
            else:
                self._current_expression = f"{a}*cos({n}*theta)"

        self._current_func = lambda theta, a=a, n=n: a * np.cos(n * theta)
        self._current_answer = {
            "expression": self._current_expression,
            "type": "rose",
            "petals": n if n % 2 == 1 else 2 * n,
        }
        self._current_func_type = "rose"

    def _generate_rose_sin(self):
        """Generate rose curve: r = a*sin(n*theta)"""
        a = random.choice([1, 2, 3])
        n = random.choice([2, 3, 4, 5])

        if a == 1:
            if n == 1:
                self._current_expression = "sin(theta)"
            else:
                self._current_expression = f"sin({n}*theta)"
        else:
            if n == 1:
                self._current_expression = f"{a}*sin(theta)"
            else:
                self._current_expression = f"{a}*sin({n}*theta)"

        self._current_func = lambda theta, a=a, n=n: a * np.sin(n * theta)
        self._current_answer = {
            "expression": self._current_expression,
            "type": "rose",
            "petals": n if n % 2 == 1 else 2 * n,
        }
        self._current_func_type = "rose"

    def _generate_spiral(self):
        """Generate Archimedean spiral: r = a*theta"""
        a = random.choice([0.5, 1, 2])

        if a == 1:
            self._current_expression = "theta"
        else:
            self._current_expression = f"{a}*theta"

        self._current_func = lambda theta, a=a: a * theta
        self._current_answer = {
            "expression": self._current_expression,
            "type": "spiral",
        }
        self._current_func_type = "spiral"

    def _generate_cardioid(self):
        """Generate cardioid: r = a*(1 + cos(theta))"""
        a = random.choice([1, 2])

        if a == 1:
            self._current_expression = "1 + cos(theta)"
        else:
            self._current_expression = f"{a}*(1 + cos(theta))"

        self._current_func = lambda theta, a=a: a * (1 + np.cos(theta))
        self._current_answer = {
            "expression": self._current_expression,
            "type": "cardioid",
        }

    def _generate_limacon(self):
        """Generate limaçon: r = a + b*cos(theta)"""
        a = random.choice([1, 2, 3])
        b = random.choice([1, 2])

        if b == 1:
            self._current_expression = f"{a} + cos(theta)"
        else:
            self._current_expression = f"{a} + {b}*cos(theta)"

        self._current_func = lambda theta, a=a, b=b: a + b * np.cos(theta)
        self._current_answer = {
            "expression": self._current_expression,
            "type": "limacon",
        }

    def _render_polar(self) -> Image.Image:
        """Render the polar plot as a PIL Image."""
        fig, ax = plt.subplots(
            figsize=(self.img_size[0] / 100, self.img_size[1] / 100),
            dpi=100,
            subplot_kw={"projection": "polar"},
        )

        # Generate theta values
        if self._current_func_type == "spiral":
            theta = np.linspace(0, 4 * np.pi, 1000)
        else:
            theta = np.linspace(0, 2 * np.pi, 1000)

        # Compute r values
        r = self._current_func(theta)

        # Handle negative r values for rose curves (reflect through origin)
        r_plot = np.abs(r)

        # Choose line style
        color = random.choice(self._line_colors)
        linewidth = random.uniform(2, 3)

        ax.plot(theta, r_plot, color=color, linewidth=linewidth)

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.6)

        # Add title
        if random.random() < 0.7:
            ax.set_title("Polar Plot: r = f(θ)", fontsize=14, fontweight="bold", pad=15)

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

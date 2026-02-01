"""Contour Plot perception environment."""

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


class PerceptionContourPlotEnv(Env):
    """Contour Plot perception environment.

    The agent must perceive a contour plot of z = f(x, y) and
    extract the mathematical expression that generates it.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (640, 480),
        xy_range: tuple[float, float] = (-3, 3),
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.xy_range = xy_range
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._current_expression: str | None = None
        self._current_func: Callable | None = None
        self._current_func_type: str | None = None
        self._current_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent("""
            You are given a contour plot of a function z = f(x, y).
            Your task is to identify the function expression.

            The function could be:
            - Quadratic: a*x^2 + b*y^2 (elliptic paraboloid)
            - Saddle: a*x^2 - b*y^2 (hyperbolic paraboloid)
            - Linear: a*x + b*y
            - Circular: x^2 + y^2 (circles)
            - Mixed: a*x*y + b*x + c*y

            Output the expression using Python syntax (** for power).

            Example: {"expression": "x**2 + y**2", "type": "circular"}
            Example: {"expression": "x**2 - y**2", "type": "saddle"}

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
                "xy_range": self.xy_range,
            },
        )

        info = {
            "oracle_answer": json.dumps(
                {
                    "expression": self._current_expression,
                    "type": self._current_func_type,
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
                    "expression": self._current_expression,
                    "type": self._current_func_type,
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
                "expression": self._current_expression,
                "type": self._current_func_type,
            }
            if parsed_action == oracle:
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    def render(self) -> Image.Image:
        return self._current_image

    def _generate_new_problem(self):
        """Generate a random 2D function and render its contour plot."""
        func_types = ["circular", "elliptic", "saddle", "linear", "mixed", "gaussian"]
        self._current_func_type = random.choice(func_types)

        if self._current_func_type == "circular":
            self._generate_circular()
        elif self._current_func_type == "elliptic":
            self._generate_elliptic()
        elif self._current_func_type == "saddle":
            self._generate_saddle()
        elif self._current_func_type == "linear":
            self._generate_linear()
        elif self._current_func_type == "mixed":
            self._generate_mixed()
        elif self._current_func_type == "gaussian":
            self._generate_gaussian()

        self._current_image = self._render_contour()

    def _generate_circular(self):
        """Generate circular function: x^2 + y^2"""
        a = random.choice([1, 2])
        if a == 1:
            self._current_expression = "x**2 + y**2"
        else:
            self._current_expression = f"{a}*(x**2 + y**2)"
        self._current_func = lambda x, y, a=a: a * (x**2 + y**2)

    def _generate_elliptic(self):
        """Generate elliptic function: a*x^2 + b*y^2"""
        a = random.choice([1, 2, 3])
        b = random.choice([1, 2, 3])
        while a == b:
            b = random.choice([1, 2, 3])

        if a == 1:
            expr_x = "x**2"
        else:
            expr_x = f"{a}*x**2"

        if b == 1:
            expr_y = "y**2"
        else:
            expr_y = f"{b}*y**2"

        self._current_expression = f"{expr_x} + {expr_y}"
        self._current_func = lambda x, y, a=a, b=b: a * x**2 + b * y**2

    def _generate_saddle(self):
        """Generate saddle function: a*x^2 - b*y^2"""
        a = random.choice([1, 2])
        b = random.choice([1, 2])

        if a == 1:
            expr_x = "x**2"
        else:
            expr_x = f"{a}*x**2"

        if b == 1:
            expr_y = "y**2"
        else:
            expr_y = f"{b}*y**2"

        self._current_expression = f"{expr_x} - {expr_y}"
        self._current_func = lambda x, y, a=a, b=b: a * x**2 - b * y**2

    def _generate_linear(self):
        """Generate linear function: a*x + b*y"""
        a = random.choice([-2, -1, 1, 2])
        b = random.choice([-2, -1, 1, 2])

        parts = []
        if a == 1:
            parts.append("x")
        elif a == -1:
            parts.append("-x")
        else:
            parts.append(f"{a}*x")

        if b == 1:
            parts.append("+ y")
        elif b == -1:
            parts.append("- y")
        elif b > 0:
            parts.append(f"+ {b}*y")
        else:
            parts.append(f"- {abs(b)}*y")

        self._current_expression = " ".join(parts)
        self._current_func = lambda x, y, a=a, b=b: a * x + b * y

    def _generate_mixed(self):
        """Generate mixed function: a*x*y + b"""
        a = random.choice([1, 2, -1, -2])
        b = random.choice([0, 1, -1])

        if a == 1:
            expr = "x*y"
        elif a == -1:
            expr = "-x*y"
        else:
            expr = f"{a}*x*y"

        if b > 0:
            self._current_expression = f"{expr} + {b}"
        elif b < 0:
            self._current_expression = f"{expr} - {abs(b)}"
        else:
            self._current_expression = expr

        self._current_func = lambda x, y, a=a, b=b: a * x * y + b

    def _generate_gaussian(self):
        """Generate Gaussian-like function: exp(-(x^2 + y^2))"""
        a = random.choice([1, 2])
        if a == 1:
            self._current_expression = "exp(-(x**2 + y**2))"
        else:
            self._current_expression = f"{a}*exp(-(x**2 + y**2))"
        self._current_func = lambda x, y, a=a: a * np.exp(-(x**2 + y**2))

    def _render_contour(self) -> Image.Image:
        """Render the contour plot as a PIL Image."""
        fig, ax = plt.subplots(
            figsize=(self.img_size[0] / 100, self.img_size[1] / 100), dpi=100
        )

        # Generate grid
        x = np.linspace(self.xy_range[0], self.xy_range[1], 200)
        y = np.linspace(self.xy_range[0], self.xy_range[1], 200)
        X, Y = np.meshgrid(x, y)

        # Compute z values
        with np.errstate(all="ignore"):
            Z = self._current_func(X, Y)
            Z = np.clip(Z, -100, 100)

        # Choose colormap
        cmaps = ["viridis", "plasma", "coolwarm", "RdYlBu", "PiYG", "PRGn"]
        cmap = random.choice(cmaps)

        # Number of contour levels
        num_levels = random.randint(10, 20)

        # Draw contour
        plot_type = random.choice(["contour", "contourf", "both"])

        if plot_type == "contour":
            cs = ax.contour(X, Y, Z, levels=num_levels, cmap=cmap)
            if random.random() < 0.7:
                ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
        elif plot_type == "contourf":
            ax.contourf(X, Y, Z, levels=num_levels, cmap=cmap, alpha=0.8)
            if random.random() < 0.5:
                ax.contour(
                    X,
                    Y,
                    Z,
                    levels=num_levels,
                    colors="black",
                    linewidths=0.5,
                    alpha=0.5,
                )
        else:
            ax.contourf(X, Y, Z, levels=num_levels, cmap=cmap, alpha=0.7)
            cs = ax.contour(X, Y, Z, levels=num_levels, colors="black", linewidths=0.8)
            if random.random() < 0.6:
                ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

        # Add colorbar
        if random.random() < 0.7:
            plt.colorbar(ax.collections[0], ax=ax, label="z")

        # Add labels
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)

        # Add title
        if random.random() < 0.7:
            ax.set_title("Contour Plot: z = f(x, y)", fontsize=14, fontweight="bold")

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

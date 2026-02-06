"""Function Graph perception environment."""

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


class FunctionGraphEnv(Env):
    # Meta: source=Perception, category=perception, turn=single
    """Function Graph perception environment.

    The agent must perceive a function graph f(x) and extract
    the mathematical expression that generates it.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (640, 480),
        x_range: tuple[float, float] = (-5, 5),
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.x_range = x_range
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
            You are given a graph of a mathematical function f(x).
            Your task is to identify the function expression.

            The function could be:
            - Polynomial: a*x^n + b*x^(n-1) + ... + c
            - Trigonometric: a*sin(b*x + c), a*cos(b*x + c)
            - Exponential: a*exp(b*x) or a*b^x
            - Logarithmic: a*log(x + b)
            - Linear: a*x + b
            - Quadratic: a*x^2 + b*x + c

            Output the expression using Python syntax (** for power, math functions).

            Example: {"expression": "2*x**2 + 3*x - 1", "type": "polynomial"}
            Example: {"expression": "3*sin(2*x)", "type": "trigonometric"}

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
                "x_range": self.x_range,
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
        """Generate a random function and render its graph."""
        func_types = [
            "linear",
            "quadratic",
            "cubic",
            "polynomial",
            "sin",
            "cos",
            "exponential",
            "logarithmic",
        ]
        self._current_func_type = random.choice(func_types)

        if self._current_func_type == "linear":
            self._generate_linear()
        elif self._current_func_type == "quadratic":
            self._generate_quadratic()
        elif self._current_func_type == "cubic":
            self._generate_cubic()
        elif self._current_func_type == "polynomial":
            self._generate_polynomial()
        elif self._current_func_type == "sin":
            self._generate_sin()
        elif self._current_func_type == "cos":
            self._generate_cos()
        elif self._current_func_type == "exponential":
            self._generate_exponential()
        elif self._current_func_type == "logarithmic":
            self._generate_logarithmic()

        self._current_image = self._render_function()

    def _generate_linear(self):
        """Generate linear function: a*x + b"""
        a = random.choice([-3, -2, -1, 1, 2, 3])
        b = random.randint(-5, 5)

        if b == 0:
            self._current_expression = f"{a}*x"
        elif b > 0:
            self._current_expression = f"{a}*x + {b}"
        else:
            self._current_expression = f"{a}*x - {abs(b)}"

        self._current_func = lambda x, a=a, b=b: a * x + b

    def _generate_quadratic(self):
        """Generate quadratic function: a*x^2 + b*x + c"""
        a = random.choice([-2, -1, 1, 2])
        b = random.randint(-4, 4)
        c = random.randint(-5, 5)

        parts = [f"{a}*x**2"]
        if b != 0:
            if b > 0:
                parts.append(f"+ {b}*x")
            else:
                parts.append(f"- {abs(b)}*x")
        if c != 0:
            if c > 0:
                parts.append(f"+ {c}")
            else:
                parts.append(f"- {abs(c)}")

        self._current_expression = " ".join(parts)
        self._current_func = lambda x, a=a, b=b, c=c: a * x**2 + b * x + c

    def _generate_cubic(self):
        """Generate cubic function: a*x^3 + b*x^2 + c*x + d"""
        a = random.choice([-1, 1])
        b = random.randint(-2, 2)
        c = random.randint(-3, 3)
        d = random.randint(-3, 3)

        parts = [f"{a}*x**3"]
        if b != 0:
            if b > 0:
                parts.append(f"+ {b}*x**2")
            else:
                parts.append(f"- {abs(b)}*x**2")
        if c != 0:
            if c > 0:
                parts.append(f"+ {c}*x")
            else:
                parts.append(f"- {abs(c)}*x")
        if d != 0:
            if d > 0:
                parts.append(f"+ {d}")
            else:
                parts.append(f"- {abs(d)}")

        self._current_expression = " ".join(parts)
        self._current_func = (
            lambda x, a=a, b=b, c=c, d=d: a * x**3 + b * x**2 + c * x + d
        )

    def _generate_polynomial(self):
        """Generate higher degree polynomial."""
        degree = random.randint(2, 4)
        coeffs = [random.randint(-2, 2) for _ in range(degree + 1)]
        coeffs[0] = random.choice([-1, 1])  # Leading coefficient non-zero

        parts = []
        for i, coef in enumerate(coeffs):
            power = degree - i
            if coef == 0:
                continue
            if power == 0:
                if coef > 0:
                    parts.append(f"+ {coef}")
                else:
                    parts.append(f"- {abs(coef)}")
            elif power == 1:
                if len(parts) == 0:
                    parts.append(f"{coef}*x")
                elif coef > 0:
                    parts.append(f"+ {coef}*x")
                else:
                    parts.append(f"- {abs(coef)}*x")
            else:
                if len(parts) == 0:
                    parts.append(f"{coef}*x**{power}")
                elif coef > 0:
                    parts.append(f"+ {coef}*x**{power}")
                else:
                    parts.append(f"- {abs(coef)}*x**{power}")

        self._current_expression = " ".join(parts) if parts else "0"
        self._current_func = lambda x, coeffs=coeffs, degree=degree: sum(
            c * x ** (degree - i) for i, c in enumerate(coeffs)
        )
        self._current_func_type = "polynomial"

    def _generate_sin(self):
        """Generate sin function: a*sin(b*x + c)"""
        a = random.choice([-2, -1, 1, 2, 3])
        b = random.choice([1, 2, 3])
        c = random.choice([0, np.pi / 4, np.pi / 2, np.pi])

        if c == 0:
            if b == 1:
                self._current_expression = f"{a}*sin(x)"
            else:
                self._current_expression = f"{a}*sin({b}*x)"
        else:
            c_str = {0: "0", np.pi / 4: "pi/4", np.pi / 2: "pi/2", np.pi: "pi"}.get(
                c, str(round(c, 2))
            )
            if b == 1:
                self._current_expression = f"{a}*sin(x + {c_str})"
            else:
                self._current_expression = f"{a}*sin({b}*x + {c_str})"

        self._current_func = lambda x, a=a, b=b, c=c: a * np.sin(b * x + c)
        self._current_func_type = "trigonometric"

    def _generate_cos(self):
        """Generate cos function: a*cos(b*x + c)"""
        a = random.choice([-2, -1, 1, 2, 3])
        b = random.choice([1, 2, 3])
        c = random.choice([0, np.pi / 4, np.pi / 2])

        if c == 0:
            if b == 1:
                self._current_expression = f"{a}*cos(x)"
            else:
                self._current_expression = f"{a}*cos({b}*x)"
        else:
            c_str = {0: "0", np.pi / 4: "pi/4", np.pi / 2: "pi/2"}.get(
                c, str(round(c, 2))
            )
            if b == 1:
                self._current_expression = f"{a}*cos(x + {c_str})"
            else:
                self._current_expression = f"{a}*cos({b}*x + {c_str})"

        self._current_func = lambda x, a=a, b=b, c=c: a * np.cos(b * x + c)
        self._current_func_type = "trigonometric"

    def _generate_exponential(self):
        """Generate exponential function: a*exp(b*x)"""
        a = random.choice([1, 2])
        b = random.choice([-1, -0.5, 0.5, 1])

        if a == 1:
            if b == 1:
                self._current_expression = "exp(x)"
            elif b == -1:
                self._current_expression = "exp(-x)"
            else:
                self._current_expression = f"exp({b}*x)"
        else:
            if b == 1:
                self._current_expression = f"{a}*exp(x)"
            elif b == -1:
                self._current_expression = f"{a}*exp(-x)"
            else:
                self._current_expression = f"{a}*exp({b}*x)"

        self._current_func = lambda x, a=a, b=b: a * np.exp(b * x)

    def _generate_logarithmic(self):
        """Generate logarithmic function: a*log(x + b)"""
        a = random.choice([1, 2, -1])
        b = random.choice([0, 1, 2])  # Shift to keep domain positive

        if b == 0:
            if a == 1:
                self._current_expression = "log(x)"
            else:
                self._current_expression = f"{a}*log(x)"
        else:
            if a == 1:
                self._current_expression = f"log(x + {b})"
            else:
                self._current_expression = f"{a}*log(x + {b})"

        self._current_func = lambda x, a=a, b=b: a * np.log(np.maximum(x + b, 1e-10))

    def _render_function(self) -> Image.Image:
        """Render the function as a PIL Image."""
        fig, ax = plt.subplots(
            figsize=(self.img_size[0] / 100, self.img_size[1] / 100), dpi=100
        )

        # Generate x values
        x = np.linspace(self.x_range[0], self.x_range[1], 500)

        # Compute y values with error handling
        with np.errstate(all="ignore"):
            y = self._current_func(x)
            # Clip extreme values for better visualization
            y = np.clip(y, -50, 50)
            # Handle NaN and Inf
            mask = np.isfinite(y)
            x_plot = x[mask]
            y_plot = y[mask]

        # Choose line style
        color = random.choice(self._line_colors)
        linewidth = random.uniform(2, 3.5)
        linestyle = random.choice(["-", "-", "-", "--"])  # Mostly solid

        ax.plot(x_plot, y_plot, color=color, linewidth=linewidth, linestyle=linestyle)

        # Add grid
        if random.random() < 0.8:
            ax.grid(True, linestyle="--", alpha=0.5)

        # Add axes through origin
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.axvline(x=0, color="black", linewidth=0.8)

        # Set axis labels
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y = f(x)", fontsize=12)

        # Add title
        if random.random() < 0.7:
            ax.set_title("Function Graph", fontsize=14, fontweight="bold")

        # Adjust y limits based on function values
        y_min, y_max = np.min(y_plot), np.max(y_plot)
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        plt.tight_layout()

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close(fig)

        # Optional noise
        if random.random() < 0.15:
            img = self._add_noise(img)

        return img

    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Add subtle noise or blur to the image."""
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        return img

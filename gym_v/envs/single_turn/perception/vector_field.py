"""Vector Field perception environment."""

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


class VectorFieldEnv(Env):
    # Meta: source=Perception, category=perception, turn=single
    """Vector Field perception environment.

    The agent must perceive a 2D vector field visualization and
    extract the mathematical expressions for Fx and Fy.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (640, 480),
        xy_range: tuple[float, float] = (-3, 3),
        grid_density: int = 15,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.xy_range = xy_range
        self.grid_density = grid_density
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._current_fx_expr: str | None = None
        self._current_fy_expr: str | None = None
        self._current_fx: Callable | None = None
        self._current_fy: Callable | None = None
        self._current_field_type: str | None = None
        self._current_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent("""
            You are given a 2D vector field visualization F(x, y) = (Fx, Fy).
            Your task is to identify the expressions for Fx and Fy components.

            The vector field could be:
            - Rotation: F = (-y, x) or variations
            - Radial: F = (x, y) pointing outward
            - Source/Sink: F = (x, y) / r or variations
            - Constant: F = (a, b)
            - Linear combinations

            Output the expressions using Python syntax.

            Example: {"Fx": "-y", "Fy": "x", "type": "rotation"}
            Example: {"Fx": "x", "Fy": "y", "type": "radial"}

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
                "field_type": self._current_field_type,
                "xy_range": self.xy_range,
            },
        )

        info = {
            "oracle_answer": json.dumps(
                {
                    "Fx": self._current_fx_expr,
                    "Fy": self._current_fy_expr,
                    "type": self._current_field_type,
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
                    "Fx": self._current_fx_expr,
                    "Fy": self._current_fy_expr,
                    "type": self._current_field_type,
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
                "Fx": self._current_fx_expr,
                "Fy": self._current_fy_expr,
                "type": self._current_field_type,
            }
            if parsed_action == oracle:
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    def render(self) -> Image.Image:
        return self._current_image

    def _generate_new_problem(self):
        """Generate a random vector field and render it."""
        field_types = [
            "rotation_ccw",
            "rotation_cw",
            "radial_out",
            "radial_in",
            "constant",
            "shear",
            "saddle",
            "vortex",
        ]
        self._current_field_type = random.choice(field_types)

        if self._current_field_type == "rotation_ccw":
            self._generate_rotation_ccw()
        elif self._current_field_type == "rotation_cw":
            self._generate_rotation_cw()
        elif self._current_field_type == "radial_out":
            self._generate_radial_out()
        elif self._current_field_type == "radial_in":
            self._generate_radial_in()
        elif self._current_field_type == "constant":
            self._generate_constant()
        elif self._current_field_type == "shear":
            self._generate_shear()
        elif self._current_field_type == "saddle":
            self._generate_saddle()
        elif self._current_field_type == "vortex":
            self._generate_vortex()

        self._current_image = self._render_vector_field()

    def _generate_rotation_ccw(self):
        """Generate counter-clockwise rotation: F = (-y, x)"""
        a = random.choice([1, 2])
        if a == 1:
            self._current_fx_expr = "-y"
            self._current_fy_expr = "x"
        else:
            self._current_fx_expr = f"-{a}*y"
            self._current_fy_expr = f"{a}*x"

        self._current_fx = lambda x, y, a=a: -a * y
        self._current_fy = lambda x, y, a=a: a * x
        self._current_field_type = "rotation"

    def _generate_rotation_cw(self):
        """Generate clockwise rotation: F = (y, -x)"""
        a = random.choice([1, 2])
        if a == 1:
            self._current_fx_expr = "y"
            self._current_fy_expr = "-x"
        else:
            self._current_fx_expr = f"{a}*y"
            self._current_fy_expr = f"-{a}*x"

        self._current_fx = lambda x, y, a=a: a * y
        self._current_fy = lambda x, y, a=a: -a * x
        self._current_field_type = "rotation"

    def _generate_radial_out(self):
        """Generate outward radial field: F = (x, y)"""
        a = random.choice([1, 2])
        if a == 1:
            self._current_fx_expr = "x"
            self._current_fy_expr = "y"
        else:
            self._current_fx_expr = f"{a}*x"
            self._current_fy_expr = f"{a}*y"

        self._current_fx = lambda x, y, a=a: a * x
        self._current_fy = lambda x, y, a=a: a * y
        self._current_field_type = "radial"

    def _generate_radial_in(self):
        """Generate inward radial field: F = (-x, -y)"""
        a = random.choice([1, 2])
        if a == 1:
            self._current_fx_expr = "-x"
            self._current_fy_expr = "-y"
        else:
            self._current_fx_expr = f"-{a}*x"
            self._current_fy_expr = f"-{a}*y"

        self._current_fx = lambda x, y, a=a: -a * x
        self._current_fy = lambda x, y, a=a: -a * y
        self._current_field_type = "radial"

    def _generate_constant(self):
        """Generate constant field: F = (a, b)"""
        a = random.choice([-2, -1, 1, 2])
        b = random.choice([-2, -1, 1, 2])

        self._current_fx_expr = str(a)
        self._current_fy_expr = str(b)

        self._current_fx = lambda x, y, a=a: np.full_like(x, a)
        self._current_fy = lambda x, y, b=b: np.full_like(y, b)

    def _generate_shear(self):
        """Generate shear field: F = (y, 0) or (0, x)"""
        if random.random() < 0.5:
            a = random.choice([1, 2])
            if a == 1:
                self._current_fx_expr = "y"
            else:
                self._current_fx_expr = f"{a}*y"
            self._current_fy_expr = "0"
            self._current_fx = lambda x, y, a=a: a * y
            self._current_fy = lambda x, y: np.zeros_like(y)
        else:
            a = random.choice([1, 2])
            self._current_fx_expr = "0"
            if a == 1:
                self._current_fy_expr = "x"
            else:
                self._current_fy_expr = f"{a}*x"
            self._current_fx = lambda x, y: np.zeros_like(x)
            self._current_fy = lambda x, y, a=a: a * x

    def _generate_saddle(self):
        """Generate saddle field: F = (x, -y)"""
        a = random.choice([1, 2])
        if a == 1:
            self._current_fx_expr = "x"
            self._current_fy_expr = "-y"
        else:
            self._current_fx_expr = f"{a}*x"
            self._current_fy_expr = f"-{a}*y"

        self._current_fx = lambda x, y, a=a: a * x
        self._current_fy = lambda x, y, a=a: -a * y

    def _generate_vortex(self):
        """Generate vortex field: F = (-y, x) / (x^2 + y^2)"""
        self._current_fx_expr = "-y / (x**2 + y**2)"
        self._current_fy_expr = "x / (x**2 + y**2)"

        def fx(x, y):
            r2 = x**2 + y**2
            r2 = np.where(r2 < 0.01, 0.01, r2)
            return -y / r2

        def fy(x, y):
            r2 = x**2 + y**2
            r2 = np.where(r2 < 0.01, 0.01, r2)
            return x / r2

        self._current_fx = fx
        self._current_fy = fy

    def _render_vector_field(self) -> Image.Image:
        """Render the vector field as a PIL Image."""
        fig, ax = plt.subplots(
            figsize=(self.img_size[0] / 100, self.img_size[1] / 100), dpi=100
        )

        # Generate grid
        x = np.linspace(self.xy_range[0], self.xy_range[1], self.grid_density)
        y = np.linspace(self.xy_range[0], self.xy_range[1], self.grid_density)
        X, Y = np.meshgrid(x, y)

        # Compute vector components
        U = self._current_fx(X, Y)
        V = self._current_fy(X, Y)

        # Normalize for better visualization
        magnitude = np.sqrt(U**2 + V**2)
        magnitude = np.where(magnitude < 0.001, 0.001, magnitude)

        # Choose visualization style
        style = random.choice(["quiver", "quiver_colored", "streamplot"])

        if style == "quiver":
            color = random.choice(["#0000FF", "#FF0000", "#008800", "#660099"])
            ax.quiver(X, Y, U, V, color=color, scale=random.uniform(20, 40))
        elif style == "quiver_colored":
            ax.quiver(
                X, Y, U, V, magnitude, cmap="viridis", scale=random.uniform(20, 40)
            )
            plt.colorbar(ax.collections[0], ax=ax, label="Magnitude")
        else:
            color = random.choice(["#0000FF", "#FF0000", "#008800"])
            # Use full grid for streamplot
            x_fine = np.linspace(self.xy_range[0], self.xy_range[1], 50)
            y_fine = np.linspace(self.xy_range[0], self.xy_range[1], 50)
            X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
            U_fine = self._current_fx(X_fine, Y_fine)
            V_fine = self._current_fy(X_fine, Y_fine)
            ax.streamplot(X_fine, Y_fine, U_fine, V_fine, color=color, density=1.2)

        # Add grid
        if random.random() < 0.7:
            ax.grid(True, linestyle="--", alpha=0.4)

        # Add axes through origin
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.axvline(x=0, color="black", linewidth=0.8)

        # Set labels
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)

        # Add title
        if random.random() < 0.7:
            ax.set_title(
                "Vector Field: F(x, y) = (Fx, Fy)", fontsize=14, fontweight="bold"
            )

        ax.set_xlim(self.xy_range)
        ax.set_ylim(self.xy_range)
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

"""Hamiltonian Path Existence environment for gym-v."""

from __future__ import annotations

from importlib import resources
import math
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEHamiltonianPathExistenceEnv(Env):
    """RLVE Hamiltonian Path Existence as a single-turn visual environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a **directed graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The graph contains the following directed edges. Each edge is represented as a tuple `(s, t)`, meaning there is a directed edge **from vertex `s` to vertex `t`**:
{edges}

Please find a path `p_1, p_2, ..., p_{N}` such that the path **visits every vertex exactly once** (revisiting vertices is NOT allowed).

Output Format:
Your final answer should be a single line containing the path in order: `p_1, p_2, ..., p_{N}`, separated by **spaces**.
Example: `0 2 1` (do **NOT** include the backticks or quotes); this means the path starts at vertex 0, then goes to vertex 2, and finally to vertex 1 (assuming 3 vertices in total)."""

    def __init__(
        self,
        max_n: int = 10,
        edge_density: float = 0.3,
        node_radius: int = 18,
        image_size: int = 700,
        padding: int = 60,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._edge_density = edge_density
        self._node_radius = node_radius
        self._image_size = image_size
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._N: int | None = None
        self._edges: list[tuple[int, int]] | None = None
        self._oracle_answer: str | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Hamiltonian Path Existence Problem:

            Given a directed graph with N vertices, find a path that visits every
            vertex exactly once. This is called a Hamiltonian path.

            In the image:
            - Vertices are numbered and shown as circles
            - Directed edges are shown as arrows between vertices
            - Find a path that visits all vertices exactly once

            Output format: A single line containing the path vertices in order,
            separated by spaces. Example: "0 2 1" means the path goes from vertex
            0 to vertex 2 to vertex 1.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return a text representation of the current graph state."""
        if self._N is None or self._edges is None:
            return ""
        lines = [f"Directed graph with {self._N} vertices (0 to {self._N - 1})."]
        lines.append("Directed edges (from -> to):")
        for s, t in self._edges:
            lines.append(f"  ({s}, {t})")
        return "\n".join(lines)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._generate()
        self._prompt = self._prompt_generate()
        self._last_image = self.render()

        state_text = self._get_state_text()
        obs = Observation(
            image=self._last_image,
            text=None,
            metadata={
                "state_text": state_text,
                "text_prompt": self._prompt,
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
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

        reward = float(self._score_answer(action_str))

        state_text = self._get_state_text()
        obs = Observation(
            image=self._last_image,
            text=None,
            metadata={
                "state_text": state_text,
                "text_prompt": self._prompt,
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
        }

        terminated = True
        truncated = False

        return (
            {agent_id: obs for agent_id in self._agent_ids},
            {agent_id: reward for agent_id in self._agent_ids},
            {
                **{agent_id: terminated for agent_id in self._agent_ids},
                "__all__": terminated,
            },
            {
                **{agent_id: truncated for agent_id in self._agent_ids},
                "__all__": truncated,
            },
            {agent_id: info for agent_id in self._agent_ids},
        )

    def _generate(self) -> None:
        """Generate problem instance with guaranteed Hamiltonian path."""
        N = int(self.np_random.integers(3, self._max_n + 1))
        self._N = N

        edges = []

        # Construct a valid Hamiltonian path
        constructed_path = list(range(N))
        self.np_random.shuffle(constructed_path)
        self._oracle_answer = " ".join(map(str, constructed_path))

        # Add edges for the constructed path
        for s, t in zip(constructed_path, constructed_path[1:], strict=False):
            edges.append((s, t))

        # Add additional random edges to reach target density
        num_edges = int(self._edge_density * N * (N - 1))
        if len(edges) < num_edges:
            remaining_edges = list(
                set((s, t) for s in range(N) for t in range(N) if s != t) - set(edges)
            )
            additional_count = min(len(remaining_edges), num_edges - len(edges))
            additional_edges = list(
                self.np_random.choice(
                    len(remaining_edges), size=additional_count, replace=False
                )
            )
            edges += [remaining_edges[i] for i in additional_edges]

        self.np_random.shuffle(edges)
        self._edges = edges

    def _prompt_generate(self) -> str:
        """Generate text prompt."""
        N = self._N
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges="\n".join(f"({s}, {t})" for s, t in self._edges),
        )

    def _process(self, answer: str | None) -> list[int] | None:
        """Process answer string into list of vertices."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            answer_array = list(map(int, answer.split()))
            return answer_array
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score answer using RLVE scoring logic."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0

        path = processed_result
        if len(path) != self._N:
            return 0.0
        if set(path) != set(range(self._N)):
            return 0.0

        edges = set(map(tuple, self._edges))
        existing = sum(
            int((s, t) in edges) for s, t in zip(path, path[1:], strict=False)
        )

        # Using "(existing/all)^beta" strategy with beta=5.0
        return ((existing / (self._N - 1)) ** 5.0) if self._N > 1 else 1.0

    def render(self) -> Image.Image:
        """Render directed graph with arrows."""
        if self._N is None:
            return Image.new(
                "RGB", (self._image_size, self._image_size), (245, 248, 250)
            )

        N = self._N
        edges = self._edges

        # Soft background color
        img = Image.new("RGB", (self._image_size, self._image_size), (245, 248, 250))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 18)
            title_font = ImageFont.truetype(str(font_path), 24)
        else:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()

        # Add title
        title = "Hamiltonian Path Existence"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, 20),
            title,
            fill=(50, 50, 70),
            font=title_font,
        )

        # Calculate circular layout positions
        center_x = self._image_size // 2
        center_y = self._image_size // 2 + 20
        radius = (self._image_size - 2 * self._padding - 60) // 2

        positions = []
        for i in range(N):
            angle = 2 * math.pi * i / N - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y))

        # Draw directed edges with arrows
        for u, v in edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]

            # Calculate arrow endpoint (stop at node boundary)
            dx, dy = x2 - x1, y2 - y1
            length = math.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx / length, dy / length
                arrow_end_x = x2 - dx * (self._node_radius + 5)
                arrow_end_y = y2 - dy * (self._node_radius + 5)

                # Draw edge line
                draw.line(
                    [(x1, y1), (arrow_end_x, arrow_end_y)],
                    fill=(120, 140, 160),
                    width=2,
                )

                # Draw arrowhead
                arrow_size = 10
                angle = math.atan2(dy, dx)
                arrow_left = (
                    arrow_end_x - arrow_size * math.cos(angle - math.pi / 6),
                    arrow_end_y - arrow_size * math.sin(angle - math.pi / 6),
                )
                arrow_right = (
                    arrow_end_x - arrow_size * math.cos(angle + math.pi / 6),
                    arrow_end_y - arrow_size * math.sin(angle + math.pi / 6),
                )
                draw.polygon(
                    [(arrow_end_x, arrow_end_y), arrow_left, arrow_right],
                    fill=(120, 140, 160),
                )

        # Draw nodes
        for i, (x, y) in enumerate(positions):
            # Draw shadow effect
            shadow_offset = 4
            draw.ellipse(
                [
                    x - self._node_radius + shadow_offset,
                    y - self._node_radius + shadow_offset,
                    x + self._node_radius + shadow_offset,
                    y + self._node_radius + shadow_offset,
                ],
                fill=(180, 180, 180, 100),
                outline=None,
            )

            # Draw main node
            draw.ellipse(
                [
                    x - self._node_radius,
                    y - self._node_radius,
                    x + self._node_radius,
                    y + self._node_radius,
                ],
                fill=(70, 130, 220),  # Blue color
                outline=(40, 80, 150),  # Darker blue border
                width=3,
            )

            # Draw node number
            text = str(i)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x - tw // 2, y - th // 2), text, fill=(255, 255, 255), font=font)

        # Add graph info
        info_text = f"Vertices: {N}  |  Directed Edges: {len(edges)}"
        bbox = draw.textbbox((0, 0), info_text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, self._image_size - 35),
            info_text,
            fill=(100, 100, 120),
            font=font,
        )

        return img

"""Longest Path environment for gym-v (self-contained)."""

from __future__ import annotations

import math
from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVELongestPathEnv(Env):
    """RLVE Longest Path as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a **directed graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following directed edges. Each edge is represented as a tuple `(s, t, w)`, meaning there is a directed edge **from vertex `s` to vertex `t` with weight `w`** :
{edges}

Your task is to find a path `p1, p2, ..., pk` such that:
- **No vertex appears more than once** in the path.
- Try your best to **maximize** the total weight of the path (i.e., the sum of all edge weights used).

**Output Format:** Your final answer should be a single line containing the path in order: `p1 p2 ... pk`, separated by **spaces**.
Example: `0 1 {N_minus_1}` (do **NOT** include the backticks or quotes); this means the path (k = 3) goes from `0` to `1` to `{N_minus_1}`."""

    def __init__(
        self,
        max_n: int = 10,
        edge_density: float = 0.5,
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
        self._edges: list[tuple[int, int, int]] | None = None
        self._reference_answer: str | None = None
        self._gold_weight: int | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Longest Simple Path Problem (Maximum Weight):

            Given a directed graph with N vertices and weighted edges, find a simple path
            (no vertex appears more than once) that maximizes the total weight.

            In the image:
            - Vertices are numbered and shown as circles
            - Directed edges are shown as arrows with weights labeled
            - Find a simple path that maximizes the sum of edge weights

            Output format: A single line containing the path as vertex indices
            separated by spaces. Example: "0 1 2 3" means the path goes from
            vertex 0 to 1, then 1 to 2, then 2 to 3.
            """
        ).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._generate()
        self._prompt = self._prompt_generate()
        self._last_image = self.render()

        obs = Observation(
            image=self._last_image,
            text=self._prompt,
            metadata={
                "rlve_prompt": self._prompt,
                "rlve_reference_answer": str(self._reference_answer),
            },
        )
        info = {
            "reference_answer": str(self._reference_answer),
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

        obs = Observation(
            image=self._last_image,
            text=None,
            metadata={
                "rlve_prompt": self._prompt,
                "rlve_reference_answer": str(self._reference_answer),
            },
        )
        info = {
            "reference_answer": str(self._reference_answer),
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
        """Generate problem instance - ported from RLVE."""
        N = int(self.np_random.integers(3, self._max_n + 1))
        self._N = N

        # Generate edges based on edge density
        num_edges = int(self._edge_density * N * (N - 1))
        all_pairs = [(s, t) for s in range(N) for t in range(N) if s != t]
        selected_indices = self.np_random.choice(
            len(all_pairs), size=min(num_edges, len(all_pairs)), replace=False
        )
        edges = [
            (all_pairs[idx][0], all_pairs[idx][1], int(self.np_random.integers(1, N + 1)))
            for idx in selected_indices
        ]
        self.np_random.shuffle(edges)
        self._edges = edges

        # Build adjacency list
        adjacent = [[] for _ in range(N)]
        for s, t, w in edges:
            adjacent[s].append((t, w))

        # Use dynamic programming to find the longest path
        # dp(s, visited) = maximum weight starting from s with visited bitmask
        self._gold_weight = 0
        dpF = {}
        parent = {}  # Track the best next node for path reconstruction

        def dp(s: int, visited: int) -> int:
            if visited == (1 << N) - 1:
                return 0
            if (s, visited) in dpF:
                return dpF[(s, visited)]
            ans = 0
            best_t = -1
            for t, w in adjacent[s]:
                if visited & (1 << t) == 0:
                    val = dp(t, visited | (1 << t)) + w
                    if val > ans:
                        ans = val
                        best_t = t
            dpF[(s, visited)] = ans
            if best_t != -1:
                parent[(s, visited)] = best_t
            return ans

        best_start = 0
        for s in range(N):
            weight = dp(s, 1 << s)
            if weight > self._gold_weight:
                self._gold_weight = weight
                best_start = s

        # Reconstruct path from best start
        path = [best_start]
        visited = 1 << best_start
        current = best_start
        while (current, visited) in parent:
            next_node = parent[(current, visited)]
            path.append(next_node)
            visited |= (1 << next_node)
            current = next_node

        self._reference_answer = " ".join(map(str, path))

    def _prompt_generate(self) -> str:
        """Generate text prompt."""
        N = self._N
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges="\n".join(f"({s}, {t}, {w})" for s, t, w in self._edges),
        )

    def _process(self, answer: str | None) -> list[int] | None:
        """Process answer string."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            answer_array = list(map(int, answer.split()))
            if not answer_array:
                return None
            return answer_array
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score answer - ported from RLVE."""
        processed_result = self._process(answer)
        if processed_result is None:
            return -1.0

        path = processed_result
        N = self._N

        # Check if all vertices are in range
        if not all(0 <= vertex < N for vertex in path):
            return -0.5

        # Check if path is simple (no vertex appears more than once)
        if len(path) != len(set(path)):
            return -0.5

        # Calculate path weight
        edge2weight = {(s, t): w for s, t, w in self._edges}
        answer_weight = 0
        for s, t in zip(path, path[1:]):
            if (s, t) not in edge2weight:
                return -0.5
            answer_weight += edge2weight[(s, t)]

        gold = self._gold_weight
        assert answer_weight <= gold and gold > 0, (
            f"answer_weight ({answer_weight}) should be <= gold ({gold}) and gold > 0"
        )

        # Return reward based on quality (answer/gold)^5
        return (answer_weight / gold) ** 5

    def render(self) -> Image.Image:
        """Render directed graph with NetworkX-style layout."""
        if self._N is None:
            return Image.new(
                "RGB", (self._image_size, self._image_size), (245, 248, 250)
            )

        N = self._N
        edges = self._edges

        img = Image.new("RGB", (self._image_size, self._image_size), (245, 248, 250))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 18)
            small_font = ImageFont.truetype(str(font_path), 14)
            title_font = ImageFont.truetype(str(font_path), 24)
        else:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            title_font = ImageFont.load_default()

        # Draw title
        title = "Longest Simple Path (Max Weight)"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, 20),
            title,
            fill=(50, 50, 70),
            font=title_font,
        )

        # Calculate node positions (circular layout)
        center_x = self._image_size // 2
        center_y = self._image_size // 2 + 20
        radius = (self._image_size - 2 * self._padding - 60) // 2

        positions = []
        for i in range(N):
            angle = 2 * math.pi * i / N - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y))

        # Draw edges with arrows and weights
        for s, t, w in edges:
            x1, y1 = positions[s]
            x2, y2 = positions[t]

            # Calculate arrow endpoint (stop at node boundary)
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx / length, dy / length
                arrow_end_x = x2 - dx * (self._node_radius + 5)
                arrow_end_y = y2 - dy * (self._node_radius + 5)

                # Draw edge line
                draw.line(
                    [(x1, y1), (arrow_end_x, arrow_end_y)],
                    fill=(100, 100, 100),
                    width=2,
                )

                # Draw arrowhead
                arrow_size = 10
                angle = np.arctan2(dy, dx)
                arrow_left = (
                    arrow_end_x - arrow_size * np.cos(angle - np.pi / 6),
                    arrow_end_y - arrow_size * np.sin(angle - np.pi / 6),
                )
                arrow_right = (
                    arrow_end_x - arrow_size * np.cos(angle + np.pi / 6),
                    arrow_end_y - arrow_size * np.sin(angle + np.pi / 6),
                )
                draw.polygon(
                    [(arrow_end_x, arrow_end_y), arrow_left, arrow_right],
                    fill=(100, 100, 100),
                )

                # Draw edge weight
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                weight_text = str(w)
                bbox = draw.textbbox((0, 0), weight_text, font=small_font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

                # Draw white background for weight text
                draw.rectangle(
                    [
                        mid_x - tw / 2 - 3,
                        mid_y - th / 2 - 3,
                        mid_x + tw / 2 + 3,
                        mid_y + th / 2 + 3,
                    ],
                    fill=(255, 255, 255),
                    outline=(150, 150, 150),
                )
                draw.text(
                    (mid_x - tw / 2, mid_y - th / 2),
                    weight_text,
                    fill=(0, 0, 0),
                    font=small_font,
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
                fill=(70, 130, 220),  # Blue nodes
                outline=(40, 80, 160),
                width=3,
            )

            # Draw node label
            text = str(i)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x - tw // 2, y - th // 2), text, fill=(255, 255, 255), font=font)

        # Add graph info
        info_text = f"Vertices: {N}  |  Edges: {len(edges)}"
        bbox = draw.textbbox((0, 0), info_text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, self._image_size - 35),
            info_text,
            fill=(100, 100, 120),
            font=font,
        )

        return img

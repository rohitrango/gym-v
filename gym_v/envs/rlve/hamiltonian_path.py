"""Hamiltonian Path environment for gym-v (self-contained)."""

from __future__ import annotations

import heapq
import math
from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEHamiltonianPathEnv(Env):
    """RLVE Hamiltonian Path as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a **directed graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following directed edges. Each edge is represented as a tuple `(s, t, w)`, meaning there is a directed edge **from vertex `s` to vertex `t` with weight `w`**:
{edges}

Your task is to find a path `p1, p2, ..., pk` such that:
- The path **visits every vertex at least once** (revisiting vertices is allowed).
- Your goal is to **minimize the total weight** of the path. The total weight is the sum of the weights of all edges used in the path.

Output Format:
Your final answer should be a single line containing the path in order: `p1, p2, ..., pk`, separated by **spaces**.
Example: `0 1 0 2` (do **NOT** include the backticks or quotes); this means the path starts at vertex 0, goes to 1, returns to 0, and then to 2 — thus visiting all three vertices at least once (assuming 3 vertices in total)."""

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
        self._reference_answer_weight: int | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Hamiltonian Path Problem (Minimum Weight):

            Given a directed graph with N vertices and weighted edges, find a path
            that visits every vertex at least once while minimizing the total weight
            of the path. Vertices can be revisited if needed.

            In the image:
            - Vertices are numbered and shown as circles
            - Directed edges are shown as arrows with weights labeled
            - Find a path that visits all vertices with minimum total edge weight

            Output format: A single line containing the path as vertex indices
            separated by spaces. Example: "0 1 2 3" means start at vertex 0,
            then go to 1, then 2, then 3.
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
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
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
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
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
        N = int(self.np_random.integers(2, self._max_n + 1))
        self._N = N

        edges = []

        # Construct a valid Hamiltonian path
        constructed_path = list(range(N))
        self.np_random.shuffle(constructed_path)

        # Initialize reference answer with constructed path
        self._reference_answer = " ".join(map(str, constructed_path))
        self._reference_answer_weight = 0

        for s, t in zip(constructed_path, constructed_path[1:]):
            w = int(self.np_random.integers(1, N + 1))
            edges.append((s, t, w))
            self._reference_answer_weight += w

        # Add additional edges based on edge density
        num_edges = int(self._edge_density * N * (N - 1))
        if len(edges) < num_edges:
            existing_pairs = {(s, t) for s, t, w in edges}
            all_pairs = [(s, t) for s in range(N) for t in range(N) if s != t]
            remaining_edges = [
                pair for pair in all_pairs if pair not in existing_pairs
            ]
            num_to_add = min(len(remaining_edges), num_edges - len(edges))
            if num_to_add > 0:
                selected_indices = self.np_random.choice(
                    len(remaining_edges), size=num_to_add, replace=False
                )
                for idx in selected_indices:
                    s, t = remaining_edges[idx]
                    edges.append((s, t, int(self.np_random.integers(1, max(1, N // 2) + 1))))

        self.np_random.shuffle(edges)
        self._edges = edges

        # Use Dijkstra with bitmask to find optimal Hamiltonian path
        adjacent = [[] for s in range(N)]
        for s, t, w in edges:
            adjacent[s].append((t, w))

        priority_queue = [(0, (1 << start, start)) for start in range(N)]
        visited_states = set()
        dist = {(1 << start, start): 0 for start in range(N)}
        prev = {(1 << start, start): (0, -1) for start in range(N)}

        while priority_queue:
            current_dist, (visited, s) = heapq.heappop(priority_queue)

            if visited == (1 << N) - 1:
                if current_dist < self._reference_answer_weight:
                    self._reference_answer_weight = current_dist

                    path = []
                    while True:
                        if visited == 0:
                            break
                        path.append(s)
                        visited, s = prev[(visited, s)]
                    path.reverse()
                    self._reference_answer = " ".join(map(str, path))
                break

            if (visited, s) in visited_states:
                continue
            visited_states.add((visited, s))

            for t, w in adjacent[s]:
                new_visited = visited | (1 << t)
                new_dist = current_dist + w
                if dist.get((new_visited, t), self._reference_answer_weight) > new_dist:
                    dist[(new_visited, t)] = new_dist
                    prev[(new_visited, t)] = (visited, s)
                    heapq.heappush(priority_queue, (new_dist, (new_visited, t)))

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
            return answer_array
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score answer."""
        processed_result = self._process(answer)
        if processed_result is None:
            return -1.0

        path = processed_result
        N = self._N

        # Check if all vertices are in range
        for vertex in path:
            if not (0 <= vertex < N):
                return -0.5

        # Check if all vertices are visited
        if len(set(path)) != N:
            return -0.5

        # Calculate path weight
        edge2weight = {(s, t): w for s, t, w in self._edges}
        answer_weight = 0
        for s, t in zip(path, path[1:]):
            if (s, t) not in edge2weight:
                return -0.5
            answer_weight += edge2weight[(s, t)]

        # Return reward based on quality
        return ((self._reference_answer_weight / answer_weight) ** 5)

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
        title = "Hamiltonian Path (Min Weight)"
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

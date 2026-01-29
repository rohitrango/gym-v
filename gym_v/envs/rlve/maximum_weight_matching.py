"""Maximum Weight Matching environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
import math
from textwrap import dedent
from typing import Any

import networkx as nx
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEMaximumWeightMatchingEnv(Env):
    """RLVE Maximum Weight Matching as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning an undirected edge **connecting vertex `u` to vertex `v` with weight `w`**:
{edges}

Your task is to select a subset of edges `S = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)]` such that:
- Each selected edge must exist in the graph.
- **Each vertex appears in at most one edge** in the set `S` — in other words, no two edges in `S` share a vertex.
- Your goal is to **maximize** the total weight of the selected edges `w_1 + w_2 + ... + w_k`.

**Output Format:**
Your final answer should be a single line containing the endpoints of the selected edges in order: `u_1 v_1 u_2 v_2 ... u_k v_k`, separated by **spaces**.
Example: `0 1 3 4` (do **NOT** include the backticks or quotes); this means k = 2 edges are selected: `(0, 1, w_1)` and `(3, 4, w_2)`."""

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
        self._oracle_answer: str | None = None
        self._gold_weight: int | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Maximum Weight Matching Problem:

            Given an undirected graph with N vertices and weighted edges, find a matching
            (subset of edges where no two edges share a vertex) that maximizes the total weight.

            In the image:
            - Vertices are numbered and shown as circles
            - Undirected edges are shown as lines with weights labeled
            - Find a matching that maximizes the sum of edge weights

            Output format: A single line containing the endpoints of selected edges:
            "u1 v1 u2 v2 ... uk vk" (space-separated). Example: "0 1 3 4" means
            select edges (0, 1) and (3, 4).
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the current state."""
        return self._prompt

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
            text=state_text,
            metadata={
                "text_prompt": f"{state_text}\n\n{self.description}",
            },
        )
        info = {
            "oracle_answer": str(self._oracle_answer),
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
            text=state_text,
            metadata={
                "text_prompt": f"{state_text}\n\n{self.description}",
            },
        )
        info = {
            "oracle_answer": str(self._oracle_answer),
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

        # Generate edges based on edge density
        num_edges = int(self._edge_density * N * (N - 1) / 2)
        # Ensure at least 1 edge when possible
        all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
        if len(all_pairs) > 0:
            num_edges = max(1, min(num_edges, len(all_pairs)))

        # Sample edges without replacement
        selected_indices = self.np_random.choice(
            len(all_pairs), size=num_edges, replace=False
        )
        edges = [
            (
                all_pairs[idx][0],
                all_pairs[idx][1],
                int(self.np_random.integers(1, N + 1)),
            )
            for idx in selected_indices
        ]
        self.np_random.shuffle(edges)
        self._edges = edges

        # Build NetworkX graph and compute maximum weight matching
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        matching = nx.max_weight_matching(G, maxcardinality=False)

        # Convert matching set to list for consistent ordering
        matching_list = sorted(matching)

        # Generate reference answer (empty string if no matching)
        if matching_list:
            self._oracle_answer = " ".join(f"{u} {v}" for u, v in matching_list)
        else:
            # Empty matching - represent as empty string
            # This can happen if graph has no edges
            self._oracle_answer = ""

        # Compute gold weight
        edge2weight = {(u, v): w for u, v, w in edges}
        self._gold_weight = sum(
            edge2weight[(min(u, v), max(u, v))] for u, v in matching_list
        )

    def _prompt_generate(self) -> str:
        """Generate text prompt."""
        N = self._N
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges="\n".join(f"({u}, {v}, {w})" for u, v, w in self._edges),
        )

    def _process(self, answer: str | None) -> list[int] | None:
        """Process answer string."""
        if answer is None:
            return None
        answer = answer.strip()
        if not answer:  # Empty string
            return None
        try:
            answer_array = list(map(int, answer.split()))
            return answer_array
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score answer - ported from RLVE."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        matches = processed_result
        N = self._N

        # Check if the number of integers is even
        if len(matches) % 2 != 0:
            return 0.0
        matches = [(matches[i], matches[i + 1]) for i in range(0, len(matches), 2)]

        # Check if each vertex appears at most once (valid matching)
        all_vertices = [u for u, v in matches] + [v for u, v in matches]
        if len(set(all_vertices)) != len(all_vertices):
            return 0.0
        edge2weight = {(u, v): w for u, v, w in self._edges}
        answer_weight = 0
        for u, v in matches:
            u, v = min(u, v), max(u, v)
            if (u, v) not in edge2weight:
                return 0.0
            answer_weight += edge2weight[(u, v)]

        gold = self._gold_weight

        # Special case: if gold is 0, the only valid answer is empty matching
        if gold == 0:
            return 1.0 if answer_weight == 0 else -0.5

        assert answer_weight <= gold, (
            f"answer_weight ({answer_weight}) should be <= gold ({gold})"
        )

        # Return reward based on quality (answer/gold)^5
        return (answer_weight / gold) ** 5

    def render(self) -> Image.Image:
        """Render undirected weighted graph with clear edge labels."""
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
        title = "Maximum Weight Matching"
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

        # Draw edges with weights
        for u, v, w in edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]

            # Draw edge line
            draw.line([(x1, y1), (x2, y2)], fill=(100, 100, 100), width=2)

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

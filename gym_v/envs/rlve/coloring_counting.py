"""Coloring Counting environment for gym-v (self-contained)."""

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


class RLVEColoringCountingEnv(Env):
    """RLVE Coloring Counting as a single-turn environment.

    This environment counts valid graph colorings where each vertex has
    a maximum allowed color constraint.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The graph contains the following undirected edges:
{edges}

You are also given an array `R` of length {N}, where `R[u]` denotes the **maximum allowed color** for vertex `u`:
{R}

A coloring assigns an integer `C[u]` to each vertex `u`, satisfying the following conditions:
- `0 <= C[u] <= R[u]` for all vertices `u`
- For every edge `(u, v)`, `C[u] ≠ C[v]` (i.e., adjacent vertices must have different colors)

The **value** of a valid coloring is the number of **distinct colors used** (i.e., the count of unique values among `C[0], C[1], ..., C[{N_minus_1}]`). Please compute the **total value of all valid colorings**.

**Output Format:** Your final answer should be a single integer — the **sum of values** over all valid colorings of the graph."""

    def __init__(
        self,
        max_n: int = 8,
        edge_density: float = 0.5,
        node_radius: int = 20,
        image_size: int = 800,
        padding: int = 60,
        num_players: int = 1,
        rewarding_strategy: str = "(min/max)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 10.0,
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

        # Reward parameters
        self._rewarding_strategy = rewarding_strategy
        self._rewarding_weight = rewarding_weight
        self._rewarding_beta = rewarding_beta

        # Environment state
        self._N: int | None = None
        self._edges: list[tuple[int, int]] | None = None
        self._R: tuple[int, ...] | None = None
        self._graph: nx.Graph | None = None
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Coloring Counting Problem:

            Given an undirected graph with N vertices, where each vertex u has a
            maximum allowed color R[u], count the total value of all valid colorings.

            A valid coloring assigns colors C[u] (0 <= C[u] <= R[u]) to each vertex
            such that adjacent vertices have different colors. The value of a coloring
            is the number of distinct colors used.

            In the image:
            - Vertices are numbered and shown as circles
            - Each vertex label shows: vertex_id (max_color)
            - Edges are shown as lines connecting vertices
            - The graph structure determines valid colorings

            Output format: A single integer representing the sum of values over all
            valid colorings of the graph.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the graph coloring problem."""
        if self._N is None or self._edges is None or self._R is None:
            return ""
        lines = [
            f"N = {self._N} vertices",
            "",
            "Edges:",
        ]
        for u, v in self._edges:
            lines.append(f"  ({u}, {v})")
        lines.append("")
        lines.append(f"R (max colors): {' '.join(map(str, self._R))}")
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
        info = {"oracle_answer": str(self._oracle_answer)}

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
        # Generate N between 2 and max_n
        N = self.np_random.integers(2, self._max_n + 1)
        self._N = N

        # Generate edges based on edge density
        num_edges = int(self._edge_density * N * (N - 1) / 2)
        all_possible_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]

        # Sample edges without replacement
        edge_indices = self.np_random.choice(
            len(all_possible_edges), size=num_edges, replace=False
        )
        edges = [all_possible_edges[i] for i in edge_indices]
        self.np_random.shuffle(edges)
        self._edges = edges

        # Calculate degree for each vertex
        Deg = [0] * N
        for u, v in edges:
            Deg[u] += 1
            Deg[v] += 1

        # Generate R[u] = random value between Deg[u] and 2*Deg[u]
        R = tuple(
            self.np_random.integers(Deg[u], 2 * Deg[u] + 1) if Deg[u] > 0 else 0
            for u in range(N)
        )
        self._R = R

        # Build graph for visualization
        self._graph = nx.Graph()
        self._graph.add_nodes_from(range(N))
        self._graph.add_edges_from(edges)

        # Compute reference answer using DP algorithm from RLVE
        # Sort nodes by R values
        nodes = list(enumerate(R))
        nodes.sort(key=lambda x: x[1])
        sorted_R = [r for _, r in nodes]
        orig_to_sorted = [0] * N
        for new_idx, (orig_idx, _) in enumerate(nodes):
            orig_to_sorted[orig_idx] = new_idx

        # Build adjacency matrix for sorted nodes
        G = [[False] * N for _ in range(N)]
        for u, v in edges:
            u_sorted = orig_to_sorted[u]
            v_sorted = orig_to_sorted[v]
            G[u_sorted][v_sorted] = G[v_sorted][u_sorted] = True

        # Check which subsets can be colored the same
        total_S = 1 << N
        Can = [True] * total_S
        for S in range(total_S):
            for u in range(N):
                if not (S >> u) & 1:
                    continue
                for v in range(u + 1, N):
                    if (S >> v) & 1 and G[u][v]:
                        Can[S] = False
                        break
                if not Can[S]:
                    break

        # DP to count colorings
        F = [[0] * (N + 1) for _ in range(total_S)]
        F[total_S - 1][0] = 1

        for S in range(total_S - 1, 0, -1):
            # Find minimum index in S
            Min = -1
            for i in range(N):
                if (S >> i) & 1:
                    Min = i
                    break

            if Min == -1:
                continue

            max_k = min(sorted_R[Min], N - 1)
            for k in range(max_k + 1):
                ways = F[S][k]
                if ways == 0:
                    continue
                W = S & ~(1 << Min)
                T = W
                while True:
                    if Can[T | (1 << Min)]:
                        new_S = W & ~T
                        F[new_S][k + 1] += ways * (sorted_R[Min] + 1 - k)
                    if T == 0:
                        break
                    T = (T - 1) & W

        self._oracle_answer = sum(F[0][k] * k for k in range(1, N + 1))
        assert self._oracle_answer > 0

    def _prompt_generate(self) -> str:
        """Generate text prompt."""
        return self.prompt_template.format(
            N=self._N,
            N_minus_1=self._N - 1,
            edges="\n".join(f"({u}, {v})" for u, v in self._edges),
            R="\n".join(f"R[{u}]={Ru}" for u, Ru in enumerate(self._R)),
        )

    def _process(self, answer: str | None) -> int | None:
        """Process answer string to extract integer."""
        if answer is not None:
            answer = answer.strip()
            try:
                int_answer = int(answer)
                return int_answer
            except ValueError:
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score answer using RLVE scoring strategy."""
        processed_result = self._process(answer)
        if processed_result is not None:
            if processed_result <= 0:
                return 0.0

            if self._rewarding_strategy == "(min/max)^beta":
                a, b = self._oracle_answer, processed_result
                return self._rewarding_weight * (
                    (min(a, b) / max(a, b)) ** self._rewarding_beta
                )
            elif self._rewarding_strategy == "gold=answer":
                return self._rewarding_weight * float(
                    processed_result == self._oracle_answer
                )
            else:
                raise NotImplementedError(
                    f"Unknown rewarding strategy: {self._rewarding_strategy}"
                )
        else:
            return 0.0

    def render(self) -> Image.Image:
        """Render graph with node labels showing max colors."""
        if self._graph is None or self._N is None or self._R is None:
            raise ValueError("Must call reset() before render()")

        G = self._graph
        N = self._N
        R = self._R

        # Create image
        img = Image.new("RGB", (self._image_size, self._image_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Load fonts
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 18)
            small_font = ImageFont.truetype(str(font_path), 14)
        else:
            font = ImageFont.load_default()
            small_font = font

        # Compute layout with fixed seed for consistency
        if N > 1:
            pos = nx.spring_layout(G, seed=42, k=1 / math.sqrt(N))
        else:
            pos = {0: (0.5, 0.5)}

        # Scale positions to image coordinates with margins
        margin = self._padding
        width = self._image_size - 2 * margin
        height = self._image_size - 2 * margin

        scaled_pos = {}
        for node, (x, y) in pos.items():
            scaled_pos[node] = (
                margin + (x + 0.5) * width,
                margin + (0.5 - y) * height,  # Flip y-axis
            )

        # Draw edges
        for u, v in G.edges():
            x1, y1 = scaled_pos[u]
            x2, y2 = scaled_pos[v]
            draw.line([(x1, y1), (x2, y2)], fill=(100, 100, 100), width=2)

        # Draw nodes with labels showing vertex and max color
        node_radius = self._node_radius
        for node in G.nodes():
            x, y = scaled_pos[node]

            # Draw node circle
            draw.ellipse(
                [
                    x - node_radius,
                    y - node_radius,
                    x + node_radius,
                    y + node_radius,
                ],
                fill=(135, 206, 250),  # Light blue
                outline=(0, 0, 0),
                width=2,
            )

            # Draw node label: "id(max_color)"
            label = f"{node}({R[node]})"
            bbox = draw.textbbox((0, 0), label, font=small_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x - tw / 2, y - th / 2), label, fill=(0, 0, 0), font=small_font)

        # Add title
        title = f"Graph with {N} vertices - Coloring Counting Problem"
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_w = title_bbox[2] - title_bbox[0]
        draw.text(
            ((self._image_size - title_w) / 2, 20),
            title,
            fill=(0, 0, 0),
            font=font,
        )

        # Add legend
        legend_y = self._image_size - 40
        legend_text = "Label format: vertex_id(max_color)"
        draw.text((margin, legend_y), legend_text, fill=(50, 50, 50), font=small_font)

        return img

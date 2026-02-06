"""Minimum Directed Spanning Tree environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
import math
from textwrap import dedent
from typing import Any

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class MinimumDirectedSpanningTreeEnv(Env):
    # Meta: source=RLVE, category=graphs, turn=single
    """RLVE Minimum Directed Spanning Tree (Arborescence) as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a **directed graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following directed edges. Each edge is represented as a tuple `(s, t, w)`, meaning a directed edge **from vertex `s` to vertex `t` with weight `w`**:
{edges}

Your task is to select a subset of edges `T = [(s_1, t_1, w_1), (s_2, t_2, w_2), ..., (s_k, t_k, w_k)]` such that:
- k = {N} - 1 = {N_minus_1} (i.e., you select exactly {N_minus_1} edges).
- The selected edges form a **spanning arborescence rooted at vertex {root}** — meaning:
  - All vertices are reachable from vertex `{root}`.
  - Each vertex other than `{root}` has exactly one incoming edge.
  - The selected edges form no cycles.
- Your goal is to **minimize** the total weight of the selected edges: `w_1 + w_2 + ... + w_k`.

**Output Format:**
Your final answer should be a single line containing the endpoints of the selected edges in order: `s_1 t_1 s_2 t_2 ... s_k t_k`, separated by **spaces**.
Example: `0 1 0 2 2 3` (do **NOT** include the backticks or quotes); this means the arborescence includes edges `(0, 1)`, `(0, 2)`, and `(2, 3)` (assuming 4 vertices in total and root = 0)."""

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
        self._root: int | None = None
        self._oracle_answer: str | None = None
        self._gold_answer: int | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Minimum Directed Spanning Tree (Arborescence) Problem:

            Given a directed graph with N vertices and weighted directed edges,
            find a minimum-weight spanning arborescence rooted at a specified vertex.
            A spanning arborescence is a directed tree where all vertices are reachable
            from the root, and each vertex (except the root) has exactly one incoming edge.

            In the image:
            - Vertices are numbered and shown as circles
            - Directed edges are shown as arrows with weights labeled
            - The root vertex is highlighted in gold
            - Find N-1 edges forming a minimum-weight arborescence from the root

            **Output Format:**
            Your final answer should be a single line containing the endpoints of the
            selected edges in order: `s_1 t_1 s_2 t_2 ... s_k t_k`, separated by
            **spaces**.
            Example: `0 1 0 2 2 3` (do **NOT** include the backticks or quotes); this
            means the arborescence includes edges `(0, 1)`, `(0, 2)`, and `(2, 3)`
            (assuming 4 vertices in total and root = 0).
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
        """Generate problem instance - ported from RLVE."""
        N = int(self.np_random.integers(6, self._max_n + 1))
        self._N = N

        # Keep generating until we get a valid instance
        while True:
            edges = []

            # Create a permutation to build a valid spanning arborescence
            permutations = list(range(N))
            self.np_random.shuffle(permutations)

            # Build initial arborescence
            for index, vertex in enumerate(permutations):
                if index == 0:
                    continue
                t = vertex
                s = permutations[self.np_random.integers(0, index)]
                weight = int(
                    self.np_random.integers(
                        1, max(2, int(self._edge_density * N * (N - 1)) + 1)
                    )
                )
                edges.append((s, t, weight))

            root = permutations[0]
            self._root = root

            # Add additional edges based on edge density
            num_edges = int(self._edge_density * N * (N - 1))
            if len(edges) < num_edges:
                # Get all possible edges not already in the graph
                existing_pairs = {(s, t) for s, t, w in edges}
                all_pairs = [(s, t) for s in range(N) for t in range(N) if s != t]
                remaining_edges = [
                    pair for pair in all_pairs if pair not in existing_pairs
                ]

                # Add random edges
                num_to_add = min(len(remaining_edges), num_edges - len(edges))
                if num_to_add > 0:
                    selected_indices = self.np_random.choice(
                        len(remaining_edges), size=num_to_add, replace=False
                    )
                    for idx in selected_indices:
                        s, t = remaining_edges[idx]
                        weight = int(
                            self.np_random.integers(
                                1, max(2, int(self._edge_density * N * (N - 1)) + 1)
                            )
                        )
                        edges.append((s, t, weight))

            self.np_random.shuffle(edges)

            # Verify uniqueness
            edge_pairs = [(s, t) for s, t, w in edges]
            if len(edge_pairs) != len(set(edge_pairs)):
                continue

            # Verify all edges are valid
            valid = True
            for s, t, _w in edges:
                if not (0 <= s < N and 0 <= t < N and s != t):
                    valid = False
                    break
            if not valid:
                continue

            # Try to find minimum spanning arborescence using NetworkX
            try:
                G = nx.DiGraph()
                # Add a virtual root node to use NetworkX's algorithm
                G.add_weighted_edges_from(edges + [(N, root, 0)])
                msa = nx.minimum_spanning_arborescence(G)

                # Extract edges excluding the virtual root edge
                msa_edges = [(s, t) for s, t in msa.edges() if (s, t) != (N, root)]
                self._oracle_answer = " ".join(f"{s} {t}" for s, t in msa_edges)

                # Calculate gold answer (minimum total weight)
                self._gold_answer = sum(
                    msa[s][t]["weight"] for s, t in msa.edges() if s != N
                )

                if self._gold_answer > 0:
                    self._edges = edges
                    break
            except Exception:
                # NetworkX might fail on some graphs
                continue

    def _prompt_generate(self) -> str:
        """Generate text prompt."""
        N = self._N
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges="\n".join(f"({s}, {t}, {w})" for s, t, w in self._edges),
            root=self._root,
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
        """Score answer - ported from RLVE."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        msa = processed_result
        N = self._N

        # Check if answer has correct format (even number of integers)
        if len(msa) % 2 != 0:
            return 0.0
        msa_edges = [(msa[i], msa[i + 1]) for i in range(0, len(msa), 2)]

        # Check if we have exactly N-1 edges
        if len(msa_edges) != N - 1:
            return 0.0
        vertices_in_msa = set(s for s, t in msa_edges) | set(t for s, t in msa_edges)
        if vertices_in_msa != set(range(N)):
            return 0.0
        adjacent_list = [[] for _ in range(N)]
        for s, t in msa_edges:
            if not (0 <= s < N and 0 <= t < N):
                return 0.0
            if s == t:
                return 0.0
            adjacent_list[s].append(t)

        # DFS to check if it's a valid tree from root
        visited = [False] * N

        def dfs(vertex: int) -> bool:
            for neighbor in adjacent_list[vertex]:
                if visited[neighbor]:
                    return False  # Cycle detected
                visited[neighbor] = True
                if not dfs(neighbor):
                    return False
            return True

        visited[self._root] = True
        if not dfs(self._root):
            return 0.0
        if not all(visited):
            return 0.0
        G = nx.DiGraph()
        G.add_nodes_from(range(N + 1))
        G.add_edges_from(msa_edges + [(N, self._root)])
        if not nx.is_arborescence(G):
            return 0.0
        edges_dict = {(s, t): w for s, t, w in self._edges}
        answer_weight = 0
        for s, t in msa_edges:
            if (s, t) not in edges_dict:
                return 0.0
            answer_weight += edges_dict[(s, t)]

        # Verify answer weight is at least as good as gold
        if answer_weight < self._gold_answer:
            # This shouldn't happen, but handle gracefully
            return 0.0
        return (self._gold_answer / answer_weight) ** 5

    def render(self) -> Image.Image:
        """Render directed graph with arrows and edge weights."""
        if self._N is None:
            return Image.new(
                "RGB", (self._image_size, self._image_size), (245, 248, 250)
            )

        N = self._N
        edges = self._edges
        root = self._root

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
        title = "Minimum Directed Spanning Tree"
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
            # Determine node color (root is gold, others are blue)
            if i == root:
                node_color = (255, 215, 0)  # Gold
                node_outline = (200, 160, 0)  # Darker gold
            else:
                node_color = (70, 130, 220)  # Blue
                node_outline = (40, 80, 160)  # Darker blue

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
                fill=node_color,
                outline=node_outline,
                width=3,
            )

            # Draw node label
            text = str(i)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            # Use black text for gold root, white text for blue nodes
            text_color = (0, 0, 0) if i == root else (255, 255, 255)
            draw.text((x - tw // 2, y - th // 2), text, fill=text_color, font=font)

        # Add graph info
        info_text = f"Vertices: {N}  |  Edges: {len(edges)}  |  Root: {root}"
        bbox = draw.textbbox((0, 0), info_text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, self._image_size - 35),
            info_text,
            fill=(100, 100, 120),
            font=font,
        )

        return img

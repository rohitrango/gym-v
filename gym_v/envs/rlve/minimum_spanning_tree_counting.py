"""Minimum Spanning Tree Counting environment for gym-v (self-contained)."""

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


class RLVEMinimumSpanningTreeCountingEnv(Env):
    """RLVE Minimum Spanning Tree Counting as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning an undirected edge **connecting vertex u to vertex v with weight w**:
{edges}

Consider a subset of edges `T = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)]` such that:
- k = {N_minus_1} (i.e., you select exactly {N_minus_1} edges),
- The selected edges form a **spanning tree** — that is, they connect all {N} vertices without forming any cycles,
- The total weight `w_1 + w_2 + ... + w_k` is **minimized** among all such spanning trees (so it is called a minimum spanning tree).

Please compute **the number of such minimum spanning trees** modulo {MOD}."""

    def __init__(
        self,
        max_n: int = 10,
        edge_ratio: float = 2.0,
        max_mod: int = 10000,
        weight_range_divisor: int = 10,
        node_radius: int = 18,
        image_size: int = 700,
        padding: int = 60,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._edge_ratio = edge_ratio
        self._max_mod = max_mod
        self._weight_range_divisor = weight_range_divisor
        self._node_radius = node_radius
        self._image_size = image_size
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._N: int | None = None
        self._edges: list[tuple[int, int, int]] | None = None
        self._MOD: int | None = None
        self._reference_answer: int | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Minimum Spanning Tree Counting Problem:

            Given an undirected weighted graph with N vertices, count the number
            of different minimum spanning trees (MSTs) modulo a given number.

            A spanning tree is a subset of N-1 edges that connects all vertices
            without forming cycles. A minimum spanning tree has the smallest
            possible total edge weight among all spanning trees.

            In the image:
            - Vertices are numbered and shown as circles
            - Edges are shown as lines with weights labeled
            - The graph is undirected (edges work both ways)

            Output format: A single integer - the count of minimum spanning trees
            modulo MOD. Example: "5" (do NOT include quotes).
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

        weight_range = max(1, int(self._edge_ratio * N / self._weight_range_divisor)) + 1

        edges = []

        # Generate spanning tree to ensure connectivity
        permutations = list(range(N))
        self.np_random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, permutations[self.np_random.integers(0, index)]
            u, v = min(u, v), max(u, v)
            edges.append((u, v, int(self.np_random.integers(1, weight_range + 1))))

        # Add additional edges based on edge ratio
        num_edges = int(self._edge_ratio * N)
        if len(edges) < num_edges:
            remaining_edges = list(
                set((u, v) for u in range(N) for v in range(u + 1, N))
                - set((u, v) for u, v, w in edges)
            )
            if remaining_edges:
                num_to_add = min(len(remaining_edges), num_edges - len(edges))
                selected = self.np_random.choice(
                    len(remaining_edges), size=num_to_add, replace=False
                )
                for idx in selected:
                    u, v = remaining_edges[idx]
                    edges.append((u, v, int(self.np_random.integers(1, weight_range + 1))))

        self.np_random.shuffle(edges)
        self._edges = edges

        self._MOD = int(self.np_random.integers(2, self._max_mod + 1))

        # Count MSTs using matrix-tree theorem for each weight class
        self._reference_answer = self._count_mst()

    def _count_mst(self) -> int:
        """Count minimum spanning trees using matrix-tree theorem."""
        N = self._N
        edges = self._edges
        MOD = self._MOD
        M = len(edges)

        def find(parent, x):
            if parent[x] != x:
                parent[x] = find(parent, parent[x])
            return parent[x]

        def union(parent, a, b):
            ra = find(parent, a)
            rb = find(parent, b)
            if ra != rb:
                parent[rb] = ra

        def det_mod(mat, mod):
            """Compute determinant modulo mod using gcd-based elimination."""
            n = len(mat)
            f = 1
            tp = 1

            for i in range(n):
                for j in range(n):
                    mat[i][j] %= mod

            for i in range(n):
                for j in range(i + 1, n):
                    a = mat[i][i]
                    b = mat[j][i]
                    while b:
                        t = a // b
                        a, b = b, a - t * b
                        for k in range(i, n):
                            mat[i][k] = (mat[i][k] - t * mat[j][k]) % mod
                        for k in range(i, n):
                            mat[i][k], mat[j][k] = mat[j][k], mat[i][k]
                        f = -f
                if mat[i][i] % mod == 0:
                    return 0
                tp = tp * (mat[i][i] % mod) % mod

            res = f * tp % mod
            return res if res >= 0 else res + mod

        # Sort edges by weight
        sorted_edges = sorted(edges, key=lambda x: x[2])

        parent = list(range(N))
        ans = 1
        i = 0

        while i < M:
            w = sorted_edges[i][2]
            j = i
            while j < M and sorted_edges[j][2] == w:
                j += 1
            group = sorted_edges[i:j]

            # Build multigraph on current DSU components
            adj_count = {}
            nodes = set()
            for u, v, _ in group:
                ru = find(parent, u)
                rv = find(parent, v)
                if ru != rv:
                    nodes.add(ru)
                    nodes.add(rv)
                    adj_count[(ru, rv)] = adj_count.get((ru, rv), 0) + 1
                    adj_count[(rv, ru)] = adj_count.get((rv, ru), 0) + 1

            # Find connected components in this subgraph
            visited = set()
            for u in nodes:
                if u in visited:
                    continue
                stack = [u]
                comp = []
                visited.add(u)
                while stack:
                    x = stack.pop()
                    comp.append(x)
                    for (a, b), cnt in adj_count.items():
                        if a == x and b not in visited:
                            visited.add(b)
                            stack.append(b)

                t = len(comp)
                if t > 1:
                    m = t - 1
                    mat = [[0] * m for _ in range(m)]
                    for xi in range(m):
                        ni = comp[xi]
                        deg = 0
                        for nj in comp:
                            deg += adj_count.get((ni, nj), 0)
                        deg %= MOD
                        mat[xi][xi] = deg
                        for yj in range(m):
                            if xi != yj:
                                nj = comp[yj]
                                mat[xi][yj] = (-adj_count.get((ni, nj), 0)) % MOD

                    ans = ans * det_mod(mat, MOD) % MOD

            # Unite DSU by all useful edges in this group
            for u, v, _ in group:
                ru = find(parent, u)
                rv = find(parent, v)
                if ru != rv:
                    union(parent, ru, rv)

            i = j

        # Check if graph is connected
        roots = {find(parent, x) for x in range(N)}
        if len(roots) != 1:
            return 0
        else:
            return ans

    def _prompt_generate(self) -> str:
        """Generate text prompt."""
        N = self._N
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges="\n".join(f"({u}, {v}, {w})" for u, v, w in self._edges),
            MOD=self._MOD,
        )

    def _process(self, answer: str | None) -> int | None:
        """Process answer string."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            int_answer = int(answer)
            return int_answer
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score answer - ported from RLVE."""
        processed_result = self._process(answer)
        if processed_result is None:
            return -1.0

        if not (0 <= processed_result < self._MOD):
            return -0.5

        if processed_result == self._reference_answer:
            return 1.0
        else:
            return 0.0

    def render(self) -> Image.Image:
        """Render undirected weighted graph."""
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
        title = "Minimum Spanning Tree Counting"
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

        # Draw edges (undirected) with weights
        for u, v, w in edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]

            # Draw edge line
            draw.line([(x1, y1), (x2, y2)], fill=(100, 100, 100), width=2)

            # Draw edge weight at midpoint
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
                fill=(135, 206, 250),  # Light blue for undirected graph
                outline=(70, 130, 180),
                width=3,
            )

            # Draw node label
            text = str(i)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x - tw // 2, y - th // 2), text, fill=(0, 0, 0), font=font)

        # Add graph info
        info_text = f"Vertices: {N}  |  Edges: {len(edges)}  |  MOD: {self._MOD}"
        bbox = draw.textbbox((0, 0), info_text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, self._image_size - 35),
            info_text,
            fill=(100, 100, 120),
            font=font,
        )

        return img

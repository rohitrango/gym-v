"""Patrol environment for gym-v (self-contained)."""

from __future__ import annotations

from collections import deque
from importlib import resources
import math
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEPatrolEnv(Env):
    """RLVE Patrol as a single-turn environment.

    Given a tree with N vertices, you can add K edges. Find the minimum number
    of edges to traverse in a path starting and ending at vertex 1 that visits
    each original edge at least once and each added edge exactly once.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from `1` to `{N}`. It contains the following {N_minus_1} undirected edges:
{edges}

You are allowed to add {K} arbitrary edges to the tree. Each added edge can connect any two existing vertices (including possibly the same vertex); it is allowed to be a duplicate of an existing edge. After adding these {K} edges, you must start at vertex `1` (and also end at vertex `1`) and traverse a path that:
- Visits each **original edge at least once**, and
- Visits each **added edge exactly once**.

Please output the **minimum total number of edges traversed** (of course, edges that are traversed multiple times should be counted multiple times) in such a path."""

    def __init__(
        self,
        max_n: int = 10,
        node_radius: int = 22,
        image_size: int = 800,
        padding: int = 60,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._node_radius = node_radius
        self._image_size = image_size
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._N: int | None = None
        self._K: int | None = None
        self._edges: list[tuple[int, int]] | None = None
        self._reference_answer: int | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Patrol Problem:

            Given a tree (connected undirected graph with no cycles) with N vertices,
            you can add K arbitrary edges. Find the minimum total number of edges
            traversed in a path that:
            - Starts and ends at vertex 1
            - Visits each original edge at least once
            - Visits each added edge exactly once

            In the image:
            - The tree is shown with vertices as numbered circles
            - Original edges are shown as solid blue lines
            - The patrol route is visualized as a path on the tree
            - Vertex 1 (start/end point) is highlighted in green
            - Other vertices are shown in light blue
            - The legend shows the constraints and parameters

            Output format: A single integer representing the minimum number of edges traversed.
            Example: "15"
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
        """Generate patrol problem instance using self.np_random."""
        N = int(self.np_random.integers(4, self._max_n + 1))
        self._N = N

        # Generate tree edges using random permutation
        edges = []
        permutations = list(range(N))
        self.np_random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, int(self.np_random.choice(permutations[:index]))
            u, v = min(u, v), max(u, v)
            edges.append((u + 1, v + 1))  # Convert to 1-based indexing
        self.np_random.shuffle(edges)

        for u, v in edges:
            assert 1 <= u < v <= N
        assert len(edges) == len(set(edges)) == N - 1

        self._edges = edges

        K = int(self.np_random.integers(1, 3))
        self._K = K

        # Build adjacency list for the tree
        adj = [[] for _ in range(N + 1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # BFS to find farthest node and distance from a start node
        def bfs(start: int, record_parent: bool = False):
            dist = [-1] * (N + 1)
            parent = [0] * (N + 1)
            q = deque([start])
            dist[start] = 0
            far_node = start
            maxd = 0
            while q:
                x = q.popleft()
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        q.append(y)
                        if dist[y] > maxd:
                            maxd = dist[y]
                            far_node = y
            if record_parent:
                return far_node, maxd, parent, dist
            return far_node, maxd

        # First BFS from node 1 to find one end of the diameter
        u, _ = bfs(1)
        # Second BFS from u to find the other end, and record parents
        v, L1, parent, _ = bfs(u, record_parent=True)

        # Case K = 1: formula is 2*(N-1) - L1 + 1
        if K == 1:
            result = 2 * (N - 1) - L1 + 1
            self._reference_answer = result
            return

        # For K = 2: mark the nodes on the diameter path
        on_path = [False] * (N + 1)
        node = v
        while node != 0:
            on_path[node] = True
            node = parent[node]

        # Prepare for DP to compute L2 (weighted diameter with diameter edges weight -1)
        d = [0] * (N + 1)
        L2 = [0]

        def dfs(x: int, p: int) -> None:
            for y in adj[x]:
                if y == p:
                    continue
                dfs(y, x)
                # weight = -1 if edge is on the original diameter, else +1
                w = -1 if on_path[x] and on_path[y] else 1
                # update the maximum combination across two branches
                L2[0] = max(L2[0], d[x] + d[y] + w)
                # update the best single branch length
                d[x] = max(d[x], d[y] + w)

        # Run DP from root = 1
        dfs(1, 0)

        # Final answer for K = 2: 2*N - L1 - L2
        result = 2 * N - L1 - L2[0]
        self._reference_answer = result

    def _prompt_generate(self) -> str:
        N = self._N
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            K=self._K,
            edges="\n".join(f"{u} {v}" for u, v in self._edges),
        )

    def _process(self, answer: str | None) -> int | None:
        """Process the answer string into an integer."""
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
        """Score the answer based on correctness.

        Returns:
            -1.0: wrong format (not an integer)
             0.0: wrong answer
            +1.0: correct answer
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if processed_result == self._reference_answer:
                return 1.0
            else:
                return 0.0
        else:
            return -1.0

    def render(self) -> Image.Image:
        """Render the patrol tree as an image with path visualization."""
        if self._N is None:
            return Image.new(
                "RGB", (self._image_size, self._image_size), (240, 245, 248)
            )

        N = self._N
        edges = self._edges

        # Create image with soft blue-gray background
        img = Image.new("RGB", (self._image_size, self._image_size), (240, 245, 248))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 20)
            title_font = ImageFont.truetype(str(font_path), 28)
            info_font = ImageFont.truetype(str(font_path), 18)
            legend_font = ImageFont.truetype(str(font_path), 16)
        else:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()
            legend_font = ImageFont.load_default()

        # Add title
        title = "Patrol Route Problem"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, 20),
            title,
            fill=(50, 50, 70),
            font=title_font,
        )

        # Compute circular layout
        center_x = self._image_size // 2
        center_y = (self._image_size // 2) + 20
        radius = (self._image_size - 2 * self._padding - 150) // 2

        positions = []
        for i in range(N):
            angle = 2 * math.pi * i / N - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y))

        # Draw edges with gradient for visual appeal
        for u, v in edges:
            x1, y1 = positions[u - 1]  # Convert back to 0-based
            x2, y2 = positions[v - 1]

            # Draw edge line in blue
            draw.line([(x1, y1), (x2, y2)], fill=(70, 130, 180), width=3)

        # Draw nodes
        for i in range(N):
            x, y = positions[i]
            vertex_num = i + 1  # 1-based indexing

            # Check if this is vertex 1 (start/end point)
            is_start = vertex_num == 1

            # Draw node shadow
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

            # Draw node with color based on whether it's the start point
            if is_start:
                # Green for start/end vertex
                node_fill = (144, 238, 144)
                node_outline = (34, 139, 34)
            else:
                # Light blue for other vertices
                node_fill = (135, 206, 250)
                node_outline = (70, 130, 180)

            draw.ellipse(
                [
                    x - self._node_radius,
                    y - self._node_radius,
                    x + self._node_radius,
                    y + self._node_radius,
                ],
                fill=node_fill,
                outline=node_outline,
                width=3,
            )

            # Draw node label
            text = str(vertex_num)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x - tw // 2, y - th // 2),
                text,
                fill=(0, 0, 0) if is_start else (255, 255, 255),
                font=font,
            )

        # Add legend box at bottom
        legend_y = self._image_size - 140
        legend_x = 40
        legend_width = self._image_size - 80
        legend_height = 120

        # Draw legend background
        draw.rectangle(
            [legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
            fill=(255, 255, 255),
            outline=(100, 100, 120),
            width=2,
        )

        # Legend title
        legend_title = "Problem Constraints:"
        draw.text(
            (legend_x + 15, legend_y + 10),
            legend_title,
            fill=(50, 50, 70),
            font=info_font,
        )

        # Legend items
        legend_items = [
            f"• Tree with {N} vertices and {N - 1} edges",
            f"• Can add {self._K} arbitrary edge(s) to the tree",
            f"• Must visit all original edges at least once",
            f"• Must visit each added edge exactly once",
            f"• Path starts and ends at vertex 1 (green)",
        ]

        y_offset = legend_y + 35
        for item in legend_items:
            draw.text(
                (legend_x + 15, y_offset),
                item,
                fill=(80, 80, 100),
                font=legend_font,
            )
            y_offset += 20

        # Add info text at top
        info_text = f"N = {N}  |  K = {self._K}  |  Edges = {len(edges)}  |  Min Path Length = {self._reference_answer}"
        bbox = draw.textbbox((0, 0), info_text, font=info_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, 60),
            info_text,
            fill=(100, 100, 120),
            font=info_font,
        )

        return img

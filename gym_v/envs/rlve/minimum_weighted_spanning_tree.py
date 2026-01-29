"""Minimum Weighted Spanning Tree environment for gym-v (self-contained)."""

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


class RLVEMinimumWeightedSpanningTreeEnv(Env):
    """RLVE Minimum Weighted Spanning Tree as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning an undirected edge **connecting vertex `u` to vertex `v` with weight `w`**:
{edges}

Your task is to select a subset of edges `T = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)]` such that:
- `k = {N} - 1 = {N_minus_1}` (i.e., you select exactly {N_minus_1} edges).
- The selected edges form a **spanning tree** — that is, they connect all {N} vertices without forming any cycles.
- You choose one vertex as the **root**. Then, every non-root vertex has exactly one incoming edge in the tree.

The cost of your scheme (the edge subset and chosen root) is defined as follows:
- For each vertex `t ≠ root`, suppose `(s, t, w)` is the single incoming edge on the path from the root to `t`, and the number of edges from the root to `t` is `K`.
- The cost of this edge is `w × K`.
- The total cost is the sum of such edge costs for all `t ≠ root`.

Your goal is to **minimize the total cost** as defined above.

**Output Format:**
Output a single line containing the root and the endpoints of the selected edges in order: `root u_1 v_1 u_2 v_2 ... u_k v_k`, separated by **spaces** . Example: `0 0 1 1 2 1 3` (do **NOT** include the backticks or quotes); this means the root is `0`, and the selected edges are `(0, 1)`, `(1, 2)`, and `(1, 3)` (assuming 4 vertices in total)."""

    def __init__(
        self,
        max_n: int = 10,
        edge_density: float = 0.5,
        node_radius: int = 18,
        image_size: int = 700,
        padding: int = 60,
        num_players: int = 1,
        rewarding_strategy: str = "(gold/answer)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 5.0,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
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

        self._rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        self._N: int | None = None
        self._edges: list[tuple[int, int, int]] | None = None
        self._oracle_answer: str | None = None
        self._gold_answer: int | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Minimum Weighted Spanning Tree Problem:

            Given an undirected graph with N vertices and weighted edges, find a spanning
            tree and root vertex that minimizes the total weighted depth cost. The cost
            of a tree with root is the sum over all non-root vertices of (edge_weight × depth),
            where depth is the number of edges from the root to that vertex.

            In the image:
            - Vertices are numbered and shown as circles
            - Undirected edges are shown as lines with weights labeled on them
            - All edges have equal visual weight (none are highlighted)
            - Find a root and N-1 edges forming a spanning tree that minimizes total weighted depth

            Output format: A single line starting with the root vertex, followed by the
            selected edges as pairs of vertex indices separated by spaces.
            Example: "0 0 1 1 2 1 3" means root=0 and edges (0,1), (1,2), (1,3) are selected.
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
            text=state_text,
            metadata={
                "text_prompt": f"{state_text}\n\n{self.description}",
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
        N = int(self.np_random.integers(3, self._max_n + 1))
        self._N = N

        edges = []

        # Create a permutation to build a valid spanning tree
        permutations = list(range(N))
        self.np_random.shuffle(permutations)

        # Build initial spanning tree
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u = vertex
            v = permutations[self.np_random.integers(0, index)]
            u, v = min(u, v), max(u, v)
            w = int(self.np_random.integers(0, N + 1))
            edges.append((u, v, w))

        # Add additional edges based on edge density
        num_edges = int(self._edge_density * N * (N - 1) / 2)
        if len(edges) < num_edges:
            remaining_edges = list(
                set((u, v) for u in range(N) for v in range(u + 1, N))
                - set((u, v) for u, v, w in edges)
            )
            num_to_add = min(len(remaining_edges), num_edges - len(edges))
            if num_to_add > 0:
                selected = self.np_random.choice(
                    len(remaining_edges), size=num_to_add, replace=False
                )
                for idx in selected:
                    u, v = remaining_edges[idx]
                    w = int(self.np_random.integers(0, N + 1))
                    edges.append((u, v, w))

        self.np_random.shuffle(edges)

        # Verify edges are valid and unique
        for u, v, w in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, w in edges))

        self._edges = edges

        # Calculate gold answer using dynamic programming from RLVE
        total_length = sum(w for u, v, w in edges)
        INF = total_length * N + 1

        # Build adjacency matrix A
        A = [[INF] * N for _ in range(N)]
        for x, y, v in edges:
            if v < A[x][y]:
                A[x][y] = A[y][x] = v

        S = (1 << N) - 1

        # Precompute low-bit index
        lg = [0] * (S + 1)
        for i in range(N):
            lg[1 << i] = i

        # f[i][j] = min cost to attach subset j (disjoint from i) to i by exactly |j| edges
        f = [dict() for _ in range(S + 1)]

        # FIX: make f[0][j] = 0 for all j
        f[0] = {j: 0 for j in range(S + 1)}

        # Base case: attaching an empty set costs 0
        for i in range(1, S + 1):
            f[i][0] = 0

        ne = [0] * (S + 1)
        # Build f table
        for i in range(1, S + 1):
            s = S ^ i
            prev = 0
            j = s
            # build reverse linked list of submasks of s
            while j:
                ne[j] = prev
                prev = j
                j = (j - 1) & s

            # traverse that linked list
            j = prev
            while j:
                x = lg[j & -j]
                # find cheapest edge from x into i
                best = INF
                tmp = i
                while tmp:
                    yb = tmp & -tmp
                    y = lg[yb]
                    if A[x][y] < best:
                        best = A[x][y]
                    tmp ^= yb

                without_low = j ^ (j & -j)
                f[i][j] = f[i][without_low] + best
                j = ne[j]

        # g[l][i] = min cost to excavate exactly the set i using l roads
        g = [[INF] * (S + 1) for _ in range(N + 1)]
        # with 0 roads, only singletons are free
        for i in range(N):
            g[0][1 << i] = 0

        # build g
        for l in range(1, N + 1):
            for i in range(1, S + 1):
                j = i
                while j:
                    prev_set = i ^ j
                    cost = g[l - 1][prev_set] + f[prev_set][j] * l
                    if cost < g[l][i]:
                        g[l][i] = cost
                    j = (j - 1) & i

        # answer is min over all l
        ans = min(g[l][S] for l in range(N + 1))
        self._gold_answer = ans

        # For reference answer, we need to construct a valid solution
        # (doesn't need to be optimal, just valid)
        # Use simple BFS from vertex 0
        root = 0
        mst_edges = []
        visited = [False] * N
        visited[root] = True
        edge_dict = {}
        for u, v, w in edges:
            if u not in edge_dict:
                edge_dict[u] = []
            if v not in edge_dict:
                edge_dict[v] = []
            edge_dict[u].append((v, w))
            edge_dict[v].append((u, w))

        queue = [root]
        while queue:
            u = queue.pop(0)
            if u in edge_dict:
                for v, w in edge_dict[u]:
                    if not visited[v]:
                        visited[v] = True
                        mst_edges.append((u, v))
                        queue.append(v)

        self._oracle_answer = f"{root} " + " ".join(f"{u} {v}" for u, v in mst_edges)

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
            return 0.0

        N = self._N
        root = processed_result[0]

        if not (0 <= root < N):
            return 0.0

        mst = processed_result[1:]
        if len(mst) % 2 != 0:
            return 0.0

        mst = [(mst[i], mst[i + 1]) for i in range(0, len(mst), 2)]

        if len(mst) != N - 1:
            return 0.0

        vertices_in_mst = set(u for u, v in mst) | set(v for u, v in mst)
        if vertices_in_mst != set(range(N)):
            return 0.0

        # Build edge weight lookup
        edge2weight = {}
        for u, v, w in self._edges:
            edge2weight[(min(u, v), max(u, v))] = w

        # Check if all edges are valid
        subgraph = nx.Graph()
        for u, v in mst:
            u_norm, v_norm = min(u, v), max(u, v)
            if (u_norm, v_norm) not in edge2weight:
                return 0.0
            subgraph.add_edge(u, v)

        if not nx.is_connected(subgraph):
            return 0.0

        if not nx.is_tree(subgraph):
            return 0.0

        # Calculate answer weight using DFS
        answer_weight = 0
        adjacent_list = [[] for _ in range(N)]
        for u, v in mst:
            adjacent_list[u].append(v)
            adjacent_list[v].append(u)

        def dfs(vertex: int, parent: int, depth: int) -> None:
            nonlocal answer_weight
            for neighbor in adjacent_list[vertex]:
                if neighbor == parent:
                    continue
                edge_weight = edge2weight[
                    (min(vertex, neighbor), max(vertex, neighbor))
                ]
                answer_weight += edge_weight * (depth + 1)
                dfs(neighbor, vertex, depth + 1)

        dfs(root, -1, 0)

        if self._gold_answer > answer_weight:
            # Answer is better than gold, something is wrong
            return 0.0

        # Return reward based on strategy
        if self._rewards["rewarding_strategy"] == "(gold/answer)^beta":
            if answer_weight == 0:
                if self._gold_answer == 0:
                    return self._rewards["rewarding_weight"] * 1.0
                else:
                    return 0.0
            return self._rewards["rewarding_weight"] * (
                (self._gold_answer / answer_weight) ** self._rewards["rewarding_beta"]
            )
        elif self._rewards["rewarding_strategy"] == "gold=answer":
            return self._rewards["rewarding_weight"] * (
                self._gold_answer == answer_weight
            )
        else:
            raise NotImplementedError(
                f"Unknown rewarding strategy: {self._rewards['rewarding_strategy']}"
            )

    def render(self) -> Image.Image:
        """Render undirected graph with edge weights."""
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
        title = "Minimum Weighted Spanning Tree"
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

        # Draw edges (undirected, no arrows) with weights
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
            # All nodes are blue (no special highlighting)
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
            # White text for blue nodes
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

"""Graph Contain Tree Counting environment for gym-v."""

from __future__ import annotations

from importlib import resources
import math
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class GraphContainTreeCountingEnv(Env):
    # Meta: source=RLVE, category=graphs, turn=single
    """RLVE Graph Contain Tree Counting as a single-turn visual environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given an **undirected graph** G and a **tree** T, each with {N} vertices labeled from `0` to `{N_minus_1}`.

- Graph G has the following undirected edge set E1:
{G_edges}

- Tree T has the following undirected edge set E2:
{T_edges}

Please compute the number of **bijections** `p` (i.e., permutations) from the vertices of T to the vertices of G such that: for every edge `(u, v)` in E2, the edge `(p(u), p(v))` exists in E1.

**Output Format:** A single integer representing the number of valid bijections."""

    def __init__(
        self,
        max_n: int = 8,
        edge_density: float = 0.5,
        node_radius: int = 20,
        image_size: int = 800,
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
        self._G_edges: list[tuple[int, int]] | None = None
        self._T_edges: list[tuple[int, int]] | None = None
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """\
            Graph Contain Tree Counting Problem:

            Given an undirected graph G and a tree T with the same number of vertices,
            count the number of bijections (permutations) p from vertices of T to
            vertices of G such that for every edge (u, v) in T, the edge (p(u), p(v))
            exists in G. This counts how many ways the tree structure can be embedded
            in the graph.

            In the image:
            - Left: Graph G (may contain cycles and extra edges)
            - Right: Tree T (exactly N-1 edges, no cycles)
            - Vertices are numbered and shown as circles
            - Edges are shown as lines connecting vertices

            **Output Format:** A single integer representing the number of valid bijections."""
        )

    def _get_state_text(self) -> str:
        """Return the text representation of the graph and tree data."""
        return self._prompt_generate()

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
            text=None,
            metadata={
                "state_text": state_text,
                "text_prompt": self._prompt,
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
        N = int(self.np_random.integers(6, self._max_n + 1))
        self._N = N

        # Generate tree T using random spanning tree algorithm
        T_edges = []
        permutation = list(range(N))
        self.np_random.shuffle(permutation)
        for index, vertex in enumerate(permutation):
            if index == 0:
                continue
            u, v = vertex, permutation[int(self.np_random.integers(0, index))]
            u, v = min(u, v), max(u, v)
            T_edges.append((u, v))
        self.np_random.shuffle(T_edges)
        self._T_edges = T_edges

        # Generate graph G by taking a permutation of T and adding more edges
        G_edges = []
        self.np_random.shuffle(permutation)
        for u, v in T_edges:
            u2, v2 = permutation[u], permutation[v]
            if u2 > v2:
                u2, v2 = v2, u2
            G_edges.append((u2, v2))

        # Add more edges to G based on edge_density
        num_edges = int(self._edge_density * N * (N - 1) / 2)
        if len(G_edges) < num_edges:
            all_edges = set((u, v) for u in range(N) for v in range(u + 1, N))
            remaining_edges = list(all_edges - set(G_edges))
            self.np_random.shuffle(remaining_edges)
            num_to_add = min(len(remaining_edges), num_edges - len(G_edges))
            G_edges.extend(remaining_edges[:num_to_add])
        self.np_random.shuffle(G_edges)
        self._G_edges = G_edges

        # Compute reference answer using dynamic programming with inclusion-exclusion
        G_adj = [[False] * N for _ in range(N)]
        for u, v in G_edges:
            G_adj[u][v] = G_adj[v][u] = True

        # Build adjacency list for tree
        T_adj = [[] for _ in range(N)]
        for u, v in T_edges:
            T_adj[u].append(v)
            T_adj[v].append(u)

        # DP array: f[u][x] = number of ways to map subtree rooted at u when u maps to x
        f = [[0] * N for _ in range(N)]

        ans = 0

        def dfs(u: int, parent: int, whi: list[int]) -> None:
            """DP on the tree for a given subset of vertices."""
            for v in T_adj[u]:
                if v == parent:
                    continue
                dfs(v, u, whi)

            # Compute f[u][x] for each x in the current subset
            for x in whi:
                f[u][x] = 1
                for v in T_adj[u]:
                    if v == parent:
                        continue
                    total = 0
                    for y in whi:
                        if G_adj[x][y]:
                            total += f[v][y]
                    f[u][x] *= total

        def solve(whi: list[int]) -> None:
            """For current subset, run tree-DP and add/subtract from answer."""
            nonlocal ans
            dfs(0, -1, whi)

            # Inclusion-exclusion: subtract if (N - |whi|) is odd, else add
            if (N - len(whi)) & 1:
                for x in whi:
                    ans -= f[0][x]
            else:
                for x in whi:
                    ans += f[0][x]

        def enumerate_subsets(dep: int = 0, vis: list[bool] | None = None) -> None:
            """Recursively enumerate all subsets."""
            if vis is None:
                vis = [False] * N
            if dep == N:
                whi = [i for i in range(N) if vis[i]]
                solve(whi)
                return
            # Exclude dep
            vis[dep] = False
            enumerate_subsets(dep + 1, vis)
            # Include dep
            vis[dep] = True
            enumerate_subsets(dep + 1, vis)

        enumerate_subsets()

        assert ans > 0, "Reference answer should be positive"
        self._oracle_answer = ans

    def _prompt_generate(self) -> str:
        N = self._N
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            G_edges="\n".join(f"({u}, {v})" for u, v in self._G_edges),
            T_edges="\n".join(f"({u}, {v})" for u, v in self._T_edges),
        )

    def _process(self, answer: str | None) -> int | None:
        if answer is None:
            return None
        answer = answer.strip()
        try:
            return int(answer)
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        if processed_result <= 0:
            return 0.0
        a, b = self._oracle_answer, processed_result
        return (min(a, b) / max(a, b)) ** 10

    def render(self) -> Image.Image:
        if self._N is None:
            return Image.new(
                "RGB", (self._image_size, self._image_size // 2), (250, 250, 250)
            )

        N = self._N
        G_edges = self._G_edges
        T_edges = self._T_edges

        graph_width = self._image_size // 2
        width = self._image_size
        height = graph_width

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 16)
            title_font = ImageFont.truetype(str(font_path), 20)
        else:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()

        def draw_graph(edges, offset_x, title, is_tree=False):
            center_x = offset_x + graph_width // 2
            center_y = height // 2
            radius = min(graph_width, height) // 2 - self._padding

            # Draw title
            bbox = draw.textbbox((0, 0), title, font=title_font)
            tw = bbox[2] - bbox[0]
            draw.text(
                (center_x - tw // 2, self._padding // 4),
                title,
                fill=(0, 0, 0),
                font=title_font,
            )

            # Position nodes in a circle
            positions = []
            for i in range(N):
                angle = 2 * math.pi * i / N - math.pi / 2
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                positions.append((x, y))

            # Draw edges
            edge_color = (34, 139, 34) if is_tree else (150, 150, 150)  # Green for tree
            for u, v in edges:
                x1, y1 = positions[u]
                x2, y2 = positions[v]
                draw.line([(x1, y1), (x2, y2)], fill=edge_color, width=2)

            # Draw nodes
            node_color = (
                (144, 238, 144) if is_tree else (100, 150, 255)
            )  # Light green for tree
            for i, (x, y) in enumerate(positions):
                draw.ellipse(
                    [
                        x - self._node_radius,
                        y - self._node_radius,
                        x + self._node_radius,
                        y + self._node_radius,
                    ],
                    fill=node_color,
                    outline=(30, 30, 30),
                    width=2,
                )

                text = str(i)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.text((x - tw // 2, y - th // 2), text, fill=(0, 0, 0), font=font)

        # Draw graph G on the left
        draw_graph(G_edges, 0, "Graph G", is_tree=False)

        # Draw tree T on the right
        draw_graph(T_edges, graph_width, "Tree T", is_tree=True)

        # Draw separator line
        draw.line(
            [(graph_width, self._padding), (graph_width, height - self._padding)],
            fill=(100, 100, 100),
            width=2,
        )

        return img

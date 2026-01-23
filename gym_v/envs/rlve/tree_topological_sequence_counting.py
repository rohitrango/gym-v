"""Tree Topological Sequence Counting environment for gym-v (self-contained)."""

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


class RLVETreeTopologicalSequenceCountingEnv(Env):
    """RLVE Tree Topological Sequence Counting as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""Please count the number of permutations of the integers from 0 to {N_minus_1}, denoted as p[0], p[1], ..., p[{N_minus_1}], such that the following {N_minus_1} constraints are satisfied: {constraints}
Note that each constraint above is of the form `p[i] < p[j]` or `p[i] > p[j]`, and collectively, these constraints correspond to a tree — that is, a connected undirected graph with no cycles — on {N} vertices labeled from 0 to {N_minus_1}.
You should output the number of valid permutations modulo {MOD}."""

    def __init__(
        self,
        max_n: int = 10,
        max_mod: int = 1000000,
        node_radius: int = 22,
        image_size: int = 700,
        padding: int = 60,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._max_mod = max_mod
        self._node_radius = node_radius
        self._image_size = image_size
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._N: int | None = None
        self._MOD: int | None = None
        self._edges: list[tuple[int, str, int]] | None = None
        self._reference_answer: int | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Tree Topological Sequence Counting Problem:

            Count the number of permutations p[0], p[1], ..., p[N-1] of integers
            from 0 to N-1 that satisfy a set of ordering constraints. The constraints
            form a tree structure (connected undirected graph with no cycles).

            Each constraint is of the form "p[i] < p[j]" or "p[i] > p[j]", meaning
            that in the permutation, vertex i must appear before/after vertex j.

            The problem asks for the count of valid permutations modulo MOD.

            In the image:
            - Vertices are numbered and shown as circles in a tree layout
            - Edges connect vertices with ordering constraints
            - Edge labels show the constraint direction ("<" or ">")
            - All nodes are shown in light blue
            - The tree structure is displayed using a hierarchical layout

            Output format: A single integer representing the count modulo MOD.
            Example: "12345"
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
        """Generate tree topological sequence counting problem instance."""
        N = int(self.np_random.integers(3, self._max_n + 1))
        self._N = N

        MOD = int(self.np_random.integers(2, self._max_mod + 1))
        self._MOD = MOD

        # Generate a random permutation
        p = list(range(N))
        self.np_random.shuffle(p)

        # Generate tree edges using random permutation
        edges = []
        permutations = list(range(N))
        self.np_random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u = vertex
            v = int(self.np_random.choice(permutations[:index]))
            u, v = min(u, v), max(u, v)
            edges.append((u, "<" if p[u] < p[v] else ">", v))
        self.np_random.shuffle(edges)

        # Verify it's a valid tree
        for u, w, v in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, w, v in edges)) == N - 1

        tree = nx.Graph()
        tree.add_edges_from((u, v) for u, w, v in edges)
        assert nx.is_tree(tree)

        self._edges = edges

        # Precompute binomial coefficients up to N
        C = [[0] * (N + 1) for _ in range(N + 1)]
        for i in range(N + 1):
            C[i][0] = 1
            for j in range(1, i + 1):
                C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % MOD

        def dfs(u: int, parent: int, h1: list, h2: list) -> tuple[list, int]:
            # f_raw[k]: number of ways (raw) to have exactly k nodes before u
            f_raw = [0, 1]  # only u itself => 1 way with k=1
            sz = 1  # size of subtree rooted at u

            # First, merge all children v where u < v (v must come after u)
            for v in h1[u]:
                if v == parent:
                    continue
                f_v, sz_v = dfs(v, u, h1, h2)
                g = f_raw[:]  # copy old
                new_sz = sz + sz_v
                new_f = [0] * (new_sz + 1)
                for j in range(1, sz + 1):
                    gj = g[j]
                    if gj == 0:
                        continue
                    for i_count in range(j, sz_v + j):
                        # Combine with child-subtree counts that place at least (i_count-j+1) before v
                        diff = f_v[sz_v] - f_v[i_count - j]
                        if diff < 0:
                            diff += MOD
                        term = gj
                        term = term * C[i_count - 1][j - 1] % MOD
                        term = term * C[sz + sz_v - i_count][sz - j] % MOD
                        term = term * diff % MOD
                        new_f[i_count] = (new_f[i_count] + term) % MOD
                f_raw = new_f
                sz = new_sz

            # Then, merge all children v where u > v (v must come before u)
            for v in h2[u]:
                if v == parent:
                    continue
                f_v, sz_v = dfs(v, u, h1, h2)
                g = f_raw[:]
                new_sz = sz + sz_v
                new_f = [0] * (new_sz + 1)
                for j in range(1, sz + 1):
                    gj = g[j]
                    if gj == 0:
                        continue
                    for i_count in range(j + 1, sz_v + j + 1):
                        # Combine with child-subtree counts that place exactly (i_count-j) before v
                        term = gj
                        term = term * C[i_count - 1][j - 1] % MOD
                        term = term * C[sz + sz_v - i_count][sz - j] % MOD
                        term = term * f_v[i_count - j] % MOD
                        new_f[i_count] = (new_f[i_count] + term) % MOD
                f_raw = new_f
                sz = new_sz

            # Turn raw counts into prefix-sums: f_pref[k] = sum_{t=1..k} f_raw[t]
            f_pref = [0] * (sz + 1)
            for i_count in range(1, sz + 1):
                s = f_pref[i_count - 1] + f_raw[i_count]
                if s >= MOD:
                    s -= MOD
                f_pref[i_count] = s

            return f_pref, sz

        # Build directed adjacency lists
        h1 = [[] for _ in range(N + 1)]
        h2 = [[] for _ in range(N + 1)]
        for a, sign, b in edges:
            x, y = a + 1, b + 1
            if sign == "<":
                h1[x].append(y)
                h2[y].append(x)
            else:
                h1[y].append(x)
                h2[x].append(y)

        f_root, _ = dfs(1, 0, h1, h2)
        # The answer is the number of ways to have all N nodes before root (i.e. full ordering)
        self._reference_answer = f_root[N] % MOD

    def _prompt_generate(self) -> str:
        N = self._N
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            constraints="; ".join(
                "p[{}] {} p[{}]".format(u, w, v) for u, w, v in self._edges
            ),
            MOD=self._MOD,
        )

    def _process(self, answer: str | None) -> int | None:
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
        processed_result = self._process(answer)
        if processed_result is not None:
            if not (0 <= processed_result < self._MOD):
                return -0.5
            if processed_result == self._reference_answer:
                return 1.0
            else:
                return -0.1
        else:
            return -1.0

    def render(self) -> Image.Image:
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
            font = ImageFont.truetype(str(font_path), 22)
            constraint_font = ImageFont.truetype(str(font_path), 16)
            title_font = ImageFont.truetype(str(font_path), 28)
            info_font = ImageFont.truetype(str(font_path), 20)
        else:
            font = ImageFont.load_default()
            constraint_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()

        # Add title
        title = "Tree Topological Sequence Counting"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, 20),
            title,
            fill=(50, 50, 70),
            font=title_font,
        )

        # Build NetworkX graph for layout
        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from((u, v) for u, _, v in edges)

        # Try hierarchical layout, fallback to spring layout
        try:
            from networkx.drawing.nx_agraph import graphviz_layout

            pos = graphviz_layout(G, prog="dot")
        except (ImportError, Exception):
            # Fallback to spring layout
            pos = nx.spring_layout(G, seed=42, k=1.5 / math.sqrt(len(G.nodes())))

        # Scale positions to image coordinates with margins
        margin = self._padding + 40
        width = self._image_size - 2 * margin
        height = self._image_size - 2 * margin - 80

        # Find bounds of positions
        if pos:
            xs = [x for x, y in pos.values()]
            ys = [y for x, y in pos.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Handle single node case
            if max_x == min_x:
                max_x = min_x + 1
            if max_y == min_y:
                max_y = min_y + 1

            scaled_pos = {}
            for node, (x, y) in pos.items():
                scaled_pos[node] = (
                    margin + (x - min_x) / (max_x - min_x) * width,
                    margin + 60 + (max_y - y) / (max_y - min_y) * height,  # Flip y-axis
                )
        else:
            # Fallback to circular layout if no positions
            scaled_pos = {}
            center_x = self._image_size // 2
            center_y = self._image_size // 2 + 20
            radius = (self._image_size - 2 * self._padding - 80) // 2
            for i in range(N):
                angle = 2 * math.pi * i / N - math.pi / 2
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                scaled_pos[i] = (x, y)

        # Draw edges with constraint labels
        for u, constraint, v in edges:
            x1, y1 = scaled_pos[u]
            x2, y2 = scaled_pos[v]

            # Draw edge line
            draw.line([(x1, y1), (x2, y2)], fill=(120, 120, 140), width=3)

            # Draw constraint label
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            constraint_text = constraint
            bbox = draw.textbbox((0, 0), constraint_text, font=constraint_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            # Constraint label background
            padding = 6
            draw.ellipse(
                [
                    mid_x - tw // 2 - padding,
                    mid_y - th // 2 - padding,
                    mid_x + tw // 2 + padding,
                    mid_y + th // 2 + padding,
                ],
                fill=(255, 250, 240),
                outline=(180, 160, 140),
                width=2,
            )
            draw.text(
                (mid_x - tw // 2, mid_y - th // 2),
                constraint_text,
                fill=(200, 60, 40),
                font=constraint_font,
            )

        # Draw nodes
        for i in range(N):
            x, y = scaled_pos[i]

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

            # Draw node with light blue
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
            text = str(i)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x - tw // 2, y - th // 2),
                text,
                fill=(255, 255, 255),
                font=font,
            )

        # Add info text
        info_text = f"Vertices: {N}  |  Edges: {len(edges)}  |  MOD: {self._MOD}"
        bbox = draw.textbbox((0, 0), info_text, font=info_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, self._image_size - 40),
            info_text,
            fill=(100, 100, 120),
            font=info_font,
        )

        return img

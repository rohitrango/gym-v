"""Tree Change One Edge Diameter environment for gym-v."""

from __future__ import annotations

from collections import deque
from importlib import resources
import math
from textwrap import dedent
from typing import Any

import networkx as nx
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVETreeChangeOneEdgeDiameterEnv(Env):
    """RLVE Tree Change One Edge Diameter as a single-turn visual environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from `1` to `{N}`. The tree contains the following edges:
{edges}

You may remove one edge from the tree and add a new edge (possibly the same edge) such that the resulting graph is still a tree. Your goal is to {maximize_or_minimize} the diameter of the resulting tree; the **diameter** of a tree is defined as the number of edges on the longest path between any two vertices.

**Output Format:** Output four integers `u1 v1 u2 v2` (do NOT include the backticks or quotes), separated by spaces, where:
- `(u1, v1)` is the edge to be removed
- `(u2, v2)` is the edge to be added"""

    def __init__(
        self,
        N: int = 8,
        node_radius: int = 22,
        image_size: int = 700,
        padding: int = 60,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._N = N
        self._node_radius = node_radius
        self._image_size = image_size
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._edges: list[tuple[int, int]] | None = None
        self._minimize_or_maximize: str | None = None
        self._gold_answer: int | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Tree Change One Edge Diameter Problem:

            Given a tree (connected undirected graph with no cycles) with N vertices,
            you can remove one edge and add a new edge such that the result is still a tree.

            Goal: Minimize or maximize the diameter (longest path) of the resulting tree.

            In the image:
            - Vertices are numbered and shown as circles
            - Edges connect vertices as lines
            - The tree structure is displayed in a hierarchical or circular layout
            - Find which edge to remove and which to add to optimize the diameter

            Output Format: Output four integers `u1 v1 u2 v2` (do NOT include the \
            backticks or quotes), separated by spaces, where:
            - `(u1, v1)` is the edge to be removed
            - `(u2, v2)` is the edge to be added
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the current state."""
        return self._prompt or ""

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
            metadata={"state_text": state_text, "text_prompt": self._prompt},
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
            metadata={"state_text": state_text, "text_prompt": self._prompt},
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
        N = self._N
        assert N >= 4, "N should be greater than or equal to 4"

        edges = []
        permutations = list(range(N))
        self.np_random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, self.np_random.choice(permutations[:index])
            u, v = min(u, v), max(u, v)
            edges.append((u + 1, v + 1))  # Convert to 1-based indexing
        self.np_random.shuffle(edges)

        for u, v in edges:
            assert 1 <= u < v <= N
        assert len(edges) == len(set(edges)) == N - 1

        self._edges = edges
        self._minimize_or_maximize = self.np_random.choice(["minimize", "maximize"])

        A = [[] for _ in range(N + 1)]
        for u, v in edges:
            A[u].append(v)
            A[v].append(u)

        def get_diameter(start, skip_u=None, skip_v=None):
            dist = [-1] * (N + 1)
            dist[start] = 0
            q = deque([start])
            far = start
            while q:
                u = q.popleft()
                for v in A[u]:
                    if skip_u is not None and (
                        (u == skip_u and v == skip_v) or (u == skip_v and v == skip_u)
                    ):
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                        if dist[v] > dist[far]:
                            far = v
            P = [-1] * (N + 1)
            dist2 = [-1] * (N + 1)
            dist2[far] = 0
            q = deque([far])
            far2 = far
            while q:
                u = q.popleft()
                for v in A[u]:
                    if skip_u is not None and (
                        (u == skip_u and v == skip_v) or (u == skip_v and v == skip_u)
                    ):
                        continue
                    if dist2[v] == -1:
                        dist2[v] = dist2[u] + 1
                        P[v] = u
                        q.append(v)
                        if dist2[v] > dist2[far2]:
                            far2 = v
            D = []
            u = far2
            while u != -1:
                D.append(u)
                u = P[u]
            return D

        def get_farthest(start, skip_u=None, skip_v=None):
            dist = [-1] * (N + 1)
            dist[start] = 0
            q = deque([start])
            far = start
            while q:
                u = q.popleft()
                for v in A[u]:
                    if skip_u is not None and (
                        (u == skip_u and v == skip_v) or (u == skip_v and v == skip_u)
                    ):
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                        if dist[v] > dist[far]:
                            far = v
            return far

        D = get_diameter(1)
        InDiameter = [False] * (N + 1)
        for u in D:
            InDiameter[u] = True

        f = [0] * (N + 1)
        g = [0] * (N + 1)

        def tree_dp(u, p):
            for v in A[u]:
                if v == p:
                    continue
                tree_dp(v, u)
                if InDiameter[v]:
                    continue
                old_f = f[u]
                g[u] = max(g[u], g[v], f[v] + 1 + old_f)
                f[u] = max(old_f, f[v] + 1)

        tree_dp(D[0], 0)

        L = len(D)
        pref = [0] * L
        cur = 0
        for i in range(L):
            u = D[i]
            if i == 0:
                pref[i] = max(0, g[u], cur + f[u])
            else:
                pref[i] = max(pref[i - 1], g[u], cur + f[u])
            cur = max(cur + 1, f[u] + 1)

        INF = N + 5
        kmin = INF
        kmax = -INF
        x1min = y1min = x2min = y2min = None
        x1max = y1max = x2max = y2max = None

        R = 0
        cur = 0
        for i in range(L - 1, 0, -1):
            u = D[i]
            R = max(R, g[u], cur + f[u])
            cur = max(cur + 1, f[u] + 1)
            left = pref[i - 1]
            cand_min = max(left, R, (R + 1) // 2 + (left + 1) // 2 + 1)
            if cand_min < kmin:
                kmin = cand_min
                x1min, y1min = u, D[i - 1]
            if R + 1 + left > kmax:
                kmax = R + 1 + left
                x1max, y1max = u, D[i - 1]

        for u in D:
            for v in A[u]:
                if not InDiameter[v]:
                    if L + g[v] > kmax:
                        kmax = L + g[v]
                        x1max, y1max = u, v

        D1 = get_diameter(x1min, x1min, y1min)
        x2min = D1[(len(D1) - 1) // 2]
        D2 = get_diameter(y1min, x1min, y1min)
        y2min = D2[(len(D2) - 1) // 2]

        x2max = get_farthest(x1max, x1max, y1max)
        y2max = get_farthest(y1max, x1max, y1max)

        if self._minimize_or_maximize == "minimize":
            self._gold_answer = kmin
            self._oracle_answer = f"{x1min} {y1min} {x2min} {y2min}"
        elif self._minimize_or_maximize == "maximize":
            self._gold_answer = kmax
            self._oracle_answer = f"{x1max} {y1max} {x2max} {y2max}"
        else:
            raise ValueError(
                "minimize_or_maximize should be either 'minimize' or 'maximize'"
            )

    def _prompt_generate(self) -> str:
        N = self._N
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges="\n".join(f"({u}, {v})" for u, v in self._edges),
            maximize_or_minimize=self._minimize_or_maximize,
        )

    def _process(self, answer: str | None) -> tuple[int, int, int, int] | None:
        if answer is not None:
            answer = answer.strip()
            try:
                u1, v1, u2, v2 = map(int, answer.split())
                return u1, v1, u2, v2
            except:
                return None
        else:
            return None

    def _score_answer(self, output: str) -> float:
        processed_result = self._process(output)
        if processed_result is not None:
            u1, v1, u2, v2 = processed_result
            N = self._N

            edges = [
                (u, v) for u, v in self._edges if (u, v) != (min(u1, v1), max(u1, v1))
            ]
            if len(edges) != N - 2:
                assert (
                    len(edges) == N - 1
                ), "There should be exactly N-1 edges in the tree"
                return 0.0
            if not (
                1 <= u2 <= N
                and 1 <= v2 <= N
                and u2 != v2
                and (min(u2, v2), max(u2, v2)) not in edges
            ):
                return 0.0
            edges.append((u2, v2))

            G = nx.Graph()
            G.add_edges_from(edges)
            if not nx.is_tree(G):
                return 0.0
            assert set([u for u, v in edges] + [v for u, v in edges]) == set(
                range(1, N + 1)
            ), "All vertices should be present in the tree"

            answer, gold = nx.diameter(G), self._gold_answer
            if self._minimize_or_maximize == "minimize":
                assert (
                    0 < gold <= answer
                ), "For minimization, answer should be greater than 0 and at least as large as the gold answer"
                return (gold / answer) ** 5
            elif self._minimize_or_maximize == "maximize":
                assert (
                    0 < answer <= gold
                ), "For maximization, answer should be greater than 0 and at most as large as the gold answer"
                return (answer / gold) ** 5
            else:
                raise ValueError(
                    "minimize_or_maximize should be either 'minimize' or 'maximize'"
                )
        else:
            return 0.0

    def render(self) -> Image.Image:
        if self._edges is None:
            return Image.new(
                "RGB", (self._image_size, self._image_size), (240, 245, 248)
            )

        N = self._N
        edges = self._edges

        img = Image.new("RGB", (self._image_size, self._image_size), (240, 245, 248))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 22)
            title_font = ImageFont.truetype(str(font_path), 28)
            info_font = ImageFont.truetype(str(font_path), 20)
        else:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()

        title = "Tree: Change One Edge to Optimize Diameter"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, 20),
            title,
            fill=(50, 50, 70),
            font=title_font,
        )

        # Try hierarchical layout first, fall back to spring if not available
        G = nx.Graph()
        G.add_nodes_from(range(1, N + 1))
        G.add_edges_from(edges)

        try:
            from networkx.drawing.nx_agraph import graphviz_layout

            pos = graphviz_layout(G, prog="dot")
        except:
            # Fallback to spring layout
            pos = nx.spring_layout(G, seed=42, k=2 / math.sqrt(N), iterations=50)

        # Scale positions to image coordinates
        center_x = self._image_size // 2
        center_y = self._image_size // 2 + 20
        max_radius = (self._image_size - 2 * self._padding - 60) // 2

        # Find bounding box of positions
        xs = [x for x, y in pos.values()]
        ys = [y for x, y in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x if max_x != min_x else 1
        height = max_y - min_y if max_y != min_y else 1

        # Scale to fit in available space
        scale = min(2 * max_radius / width, 2 * max_radius / height) * 0.8

        scaled_pos = {}
        for node, (x, y) in pos.items():
            scaled_x = center_x + (x - (min_x + max_x) / 2) * scale
            scaled_y = center_y + (y - (min_y + max_y) / 2) * scale
            scaled_pos[node] = (scaled_x, scaled_y)

        # Draw edges
        for u, v in edges:
            x1, y1 = scaled_pos[u]
            x2, y2 = scaled_pos[v]
            draw.line([(x1, y1), (x2, y2)], fill=(100, 120, 140), width=3)

        # Draw nodes
        for node in range(1, N + 1):
            x, y = scaled_pos[node]

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

            # Draw node circle
            draw.ellipse(
                [
                    x - self._node_radius,
                    y - self._node_radius,
                    x + self._node_radius,
                    y + self._node_radius,
                ],
                fill=(70, 130, 180),  # Steel blue
                outline=(50, 90, 140),
                width=3,
            )

            # Draw node label
            text = str(node)
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
        objective = (
            "Minimize" if self._minimize_or_maximize == "minimize" else "Maximize"
        )
        info_text = (
            f"Vertices: {N}  |  Edges: {len(edges)}  |  Objective: {objective} Diameter"
        )
        bbox = draw.textbbox((0, 0), info_text, font=info_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, self._image_size - 40),
            info_text,
            fill=(100, 100, 120),
            font=info_font,
        )

        return img

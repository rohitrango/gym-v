"""Tree Add One Edge Diameter environment for gym-v."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVETreeAddOneEdgeDiameterEnv(Env):
    """RLVE Tree Add One Edge Diameter as a single-turn visual environment.

    Given a tree with N vertices and a new edge weight L, find which two vertices
    to connect with an edge of weight L to minimize the maximum distance (diameter)
    in the resulting graph.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from `1` to `{N}`. The tree contains the following {N_minus_1} undirected edges, where each tuple `(u, v, w)` represents an edge between vertices `u` and `v` with weight `w`:
{edges}

Let's add **exactly one undirected edge** with weight {L} to the tree. Our goal is to minimize the **longest distance** between any two vertices in the resulting graph. The distance between two vertices is defined as the sum of edge weights along the shortest path connecting them. Output two integers `x y` (do NOT include quotes), separated by a space, indicating the two vertices to which the new edge of weight {L} is added."""

    def __init__(
        self,
        max_n: int = 10,
        node_radius: int = 20,
        image_width: int = 800,
        image_height: int = 700,
        num_players: int = 1,
        rewarding_strategy: str = "(gold/answer)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._node_radius = node_radius
        self._image_width = image_width
        self._image_height = image_height
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self.rewards = {
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        self._N: int | None = None
        self._L: int | None = None
        self._edges: list[tuple[int, int, int]] | None = None
        self._gold_answer: int | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Tree Add One Edge Diameter Problem:

            Given a tree (connected undirected graph with no cycles) with N vertices,
            you can add exactly one edge with a specified weight L to the tree.

            Goal: Minimize the diameter (longest distance between any two vertices)
            in the resulting graph after adding the new edge.

            In the image:
            - Vertices are numbered (1-indexed) and shown as circles
            - Edges are shown as lines with weights labeled
            - The tree structure is laid out hierarchically
            - Consider which edge addition would most reduce the diameter

            Output format: Two integers x y separated by a space, indicating the
            two vertices to connect with the new edge. Example: "1 5"
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
        """Generate problem instance - ported from RLVE."""
        N = int(self.np_random.integers(4, self._max_n + 1))
        self._N = N

        edges = []
        permutations = list(range(N))
        self.np_random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, self.np_random.choice(permutations[:index])
            u, v = min(u, v), max(u, v)
            edges.append((u + 1, v + 1, int(self.np_random.integers(0, N + 1))))
        self.np_random.shuffle(edges)
        self._edges = edges

        # Verify tree structure
        for u, v, w in edges:
            assert 1 <= u < v <= N
        assert len(edges) == len(set((u, v) for u, v, w in edges)) == N - 1

        L = int(self.np_random.integers(0, N + 1))
        self._L = L

        NEG_INF = 0

        # Build adjacency list
        e = [[] for _ in range(N + 1)]
        for u, v, w in edges:
            e[u].append((v, w))
            e[v].append((u, w))
            NEG_INF -= w + 1

        # 1) Find S: the farthest node from node 1
        dis1 = [0] * (N + 1)
        stack = [(1, 0)]
        while stack:
            u, p = stack.pop()
            for v, w in e[u]:
                if v == p:
                    continue
                dis1[v] = dis1[u] + w
                stack.append((v, u))
        S = max(range(1, N + 1), key=lambda i: dis1[i])

        # 2) DFS from S to compute distances (dis) and subtree max-distance (mx), plus parent pointers
        dis = [0] * (N + 1)
        mx = [0] * (N + 1)
        parent = [0] * (N + 1)
        stack2 = [(S, 0, 0)]  # (node, parent, state) state=0: pre, state=1: post
        while stack2:
            u, p, st = stack2.pop()
            if st == 0:
                parent[u] = p
                stack2.append((u, p, 1))
                for v, w in e[u]:
                    if v == p:
                        continue
                    dis[v] = dis[u] + w
                    stack2.append((v, u, 0))
            else:
                mxd = dis[u]
                for v, _ in e[u]:
                    if v == p:
                        continue
                    if mx[v] > mxd:
                        mxd = mx[v]
                mx[u] = mxd

        # 3) Find T: the farthest node from S, and record the original diameter
        T = max(range(1, N + 1), key=lambda i: dis[i])
        diam = dis[T]

        # 4) Extract the diameter path from S to T
        p_nodes = []
        u = T
        while True:
            p_nodes.append(u)
            if u == S:
                break
            u = parent[u]
        p_nodes.reverse()
        cnt = len(p_nodes)

        # 5) Compute prefix distances along the path (pre) and branch depths (val)
        pre = [0] * (cnt + 2)
        val = [0] * (cnt + 2)
        for i in range(1, cnt + 1):
            pre[i] = dis[p_nodes[i - 1]]
        for i in range(1, cnt + 1):
            node = p_nodes[i - 1]
            prev_node = p_nodes[i - 2] if i > 1 else None
            next_node = p_nodes[i] if i < cnt else None
            best = 0
            for v, _ in e[node]:
                if v == prev_node or v == next_node:
                    continue
                depth = mx[v] - dis[node]
                if depth > best:
                    best = depth
            val[i] = best

        # 6) Prepare sorted index lists for the two-pointer checks
        p1 = [0] + sorted(range(1, cnt + 1), key=lambda i: val[i] + pre[i])
        p2 = [0] + sorted(
            range(1, cnt + 1), key=lambda i: val[i] - pre[i], reverse=True
        )

        # 7) Feasibility check: can we achieve diameter <= x after adding the new edge?
        def check(x):
            A = B = C = D = NEG_INF
            mx1 = mx2 = NEG_INF
            j = 0

            # First pass: accumulate constraints from violating pairs
            for idx in range(1, cnt + 1):
                i_idx = p1[idx]
                while j + 1 <= cnt and (
                    val[i_idx] + pre[i_idx] + val[p2[j + 1]] - pre[p2[j + 1]] > x
                ):
                    j += 1
                    k = p2[j]
                    c1 = val[k] + pre[k]
                    if c1 > mx1:
                        mx1 = c1
                    c2 = val[k] - pre[k]
                    if c2 > mx2:
                        mx2 = c2

                # Update A, B, C, D
                t = val[i_idx] + pre[i_idx] + mx1
                if t > A:
                    A = t
                t = val[i_idx] - pre[i_idx] + mx1
                if t > B:
                    B = t
                t = val[i_idx] + pre[i_idx] + mx2
                if t > C:
                    C = t
                t = val[i_idx] - pre[i_idx] + mx2
                if t > D:
                    D = t

                # If no pairs violated for all i, it's already feasible
                if idx == cnt and j == 0:
                    return True

            # Adjust constraints by (L - x)
            delta = L - x
            A += delta
            B += delta
            C += delta
            D += delta

            # Second pass: sliding-window ranges
            a, b, c, d = cnt + 1, 1, 0, cnt
            for i_idx in range(1, cnt + 1):
                while a > 1 and pre[i_idx] + pre[a - 1] >= A:
                    a -= 1
                while b <= cnt and -pre[i_idx] + pre[b] < B:
                    b += 1
                while c < cnt and pre[i_idx] - pre[c + 1] >= C:
                    c += 1
                while d >= 1 and -pre[i_idx] - pre[d] < D:
                    d -= 1

                left = a if a > b else b
                r1 = c if c < d else d
                right = i_idx - 1 if i_idx - 1 < r1 else r1
                if left <= right:
                    return True

            return False

        # 8) Binary search for the minimal achievable diameter
        left, right, ans = 0, diam, diam
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        self._gold_answer = ans

        # Find at least one edge that achieves the optimal diameter
        # Try all possible edges and find one that achieves optimal diameter
        best_edge = (1, 2)  # Default fallback
        for x in range(1, N + 1):
            for y in range(x + 1, N + 1):
                # Skip if edge already exists
                if any((u == x and v == y) or (u == y and v == x) for u, v, _ in edges):
                    continue
                # Check diameter with this edge
                G = nx.MultiGraph()
                G.add_weighted_edges_from(edges)
                G.add_edge(x, y, weight=L)
                diameter = max(
                    max(
                        nx.single_source_dijkstra_path_length(
                            G, u, weight="weight"
                        ).values()
                    )
                    for u in G.nodes()
                )
                if diameter == ans:
                    best_edge = (x, y)
                    break
            if best_edge != (1, 2) or (x == 1 and y == 2):
                break

        self._oracle_answer = f"{best_edge[0]} {best_edge[1]}"

    def _prompt_generate(self) -> str:
        """Generate text prompt - ported from RLVE."""
        N = self._N
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges="\n".join(f"({u}, {v}, {w})" for u, v, w in self._edges),
            L=self._L,
        )

    def _process(self, answer: str | None) -> tuple[int, int] | None:
        """Process answer string - ported from RLVE."""
        if answer is not None:
            answer = answer.strip()
            try:
                x, y = map(int, answer.split())
                return x, y
            except Exception:
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score answer - ported from RLVE."""
        processed_result = self._process(answer)
        if processed_result is not None:
            x, y = processed_result
            if not (1 <= x <= self._N and 1 <= y <= self._N):
                return 0.0

            G = nx.MultiGraph()
            G.add_weighted_edges_from(self._edges)
            G.add_edge(x, y, weight=self._L)
            answer_diam = max(
                max(
                    nx.single_source_dijkstra_path_length(
                        G, u, weight="weight"
                    ).values()
                )
                for u in G.nodes()
            )
            gold = self._gold_answer
            assert (
                0 <= gold <= answer_diam
            ), "The answer should be at least as large as the gold answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta":
                if answer_diam == 0:
                    assert gold == 0, "gold should be zero if answer is zero"
                    return self.rewards["rewarding_weight"] * 1.0
                return self.rewards["rewarding_weight"] * (
                    (gold / answer_diam) ** self.rewards["rewarding_beta"]
                )
            elif self.rewards["rewarding_strategy"] == "gold=answer":
                return self.rewards["rewarding_weight"] * (gold == answer_diam)
            else:
                raise NotImplementedError(
                    "Unknown rewarding strategy: {}".format(
                        self.rewards["rewarding_strategy"]
                    )
                )
        else:
            return 0.0

    def render(self) -> Image.Image:
        """Render tree with hierarchical layout."""
        if self._N is None:
            return Image.new(
                "RGB", (self._image_width, self._image_height), (240, 245, 248)
            )

        N = self._N
        L = self._L
        edges = self._edges

        # Create image with soft blue-gray background
        img = Image.new("RGB", (self._image_width, self._image_height), (240, 245, 248))
        draw = ImageDraw.Draw(img)

        # Load fonts
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 20)
            weight_font = ImageFont.truetype(str(font_path), 16)
            title_font = ImageFont.truetype(str(font_path), 26)
            info_font = ImageFont.truetype(str(font_path), 18)
        else:
            font = ImageFont.load_default()
            weight_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()

        # Draw title
        title = "Tree Add One Edge - Minimize Diameter"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_width // 2 - tw // 2, 20),
            title,
            fill=(50, 50, 70),
            font=title_font,
        )

        # Create NetworkX graph (1-indexed vertices)
        G = nx.Graph()
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)

        # Try hierarchical layout using spring layout with tree structure
        # Use spring layout with a high k value for better tree-like spacing
        try:
            pos = nx.spring_layout(
                G, k=2.0 / np.sqrt(len(G.nodes())), iterations=50, seed=42
            )
        except Exception:
            # Fallback to circular layout
            pos = nx.circular_layout(G)

        # Scale positions to image coordinates with margins
        margin_x = 120
        margin_y = 120
        width = self._image_width - 2 * margin_x
        height = self._image_height - 2 * margin_y - 80

        scaled_pos = {}
        for node, (x, y) in pos.items():
            scaled_pos[node] = (
                margin_x + (x + 0.5) * width,
                margin_y + (0.5 - y) * height,
            )

        # Draw edges
        for u, v in G.edges():
            x1, y1 = scaled_pos[u]
            x2, y2 = scaled_pos[v]
            w = G[u][v]["weight"]

            # Edge color based on weight
            intensity = 120
            edge_color = (intensity, intensity + 20, intensity + 40)
            draw.line([(x1, y1), (x2, y2)], fill=edge_color, width=3)

            # Draw weight label
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            weight_text = str(w)
            bbox = draw.textbbox((0, 0), weight_text, font=weight_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            # Weight label background
            padding = 5
            draw.ellipse(
                [
                    mid_x - tw // 2 - padding,
                    mid_y - th // 2 - padding,
                    mid_x + tw // 2 + padding,
                    mid_y + th // 2 + padding,
                ],
                fill=(255, 250, 245),
                outline=(220, 200, 180),
                width=2,
            )
            draw.text(
                (mid_x - tw // 2, mid_y - th // 2),
                weight_text,
                fill=(200, 80, 40),
                font=weight_font,
            )

        # Draw nodes
        for node in G.nodes():
            x, y = scaled_pos[node]

            # Draw node shadow
            shadow_offset = 3
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

            # Draw node circle (bright blue)
            draw.ellipse(
                [
                    x - self._node_radius,
                    y - self._node_radius,
                    x + self._node_radius,
                    y + self._node_radius,
                ],
                fill=(100, 150, 255),
                outline=(60, 100, 200),
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

        # Draw info at bottom
        info_text = f"Vertices: {N}  |  Edges: {len(edges)}  |  New edge weight L = {L}"
        bbox = draw.textbbox((0, 0), info_text, font=info_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_width // 2 - tw // 2, self._image_height - 45),
            info_text,
            fill=(100, 100, 120),
            font=info_font,
        )

        return img

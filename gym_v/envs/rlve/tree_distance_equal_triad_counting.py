"""Tree Distance Equal Triad Counting environment for gym-v."""

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


class RLVETreeDistanceEqualTriadCountingEnv(Env):
    """RLVE Tree Distance Equal Triad Counting as a single-turn visual environment.

    Given a tree with N vertices, count the number of triads (three-vertex sets
    A, B, C where A < B < C) such that the pairwise distances between all three
    vertices are equal.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices, labeled from `1` to `{N}`. It contains the following {N_minus_1} undirected edges:
{edges}

Please compute the number of three-vertex sets (a triad of vertices A, B, and C such that 1 ≤ A < B < C ≤ {N}) for which the **pairwise distances** are all equal — that is, the distance between A and B, between A and C, and between B and C are all the same. The distance between two vertices is the number of edges on the shortest path connecting them."""

    def __init__(
        self,
        max_n: int = 10,
        node_radius: int = 22,
        image_size: int = 700,
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
        self._edges: list[tuple[int, int]] | None = None
        self._oracle_answer: int | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Tree Distance Equal Triad Counting Problem:

            Given a tree (connected undirected graph with no cycles) with N vertices,
            count how many three-vertex sets (triads A, B, C with A < B < C) have
            the property that all three pairwise distances are equal.

            The distance between two vertices is the number of edges on the shortest
            path connecting them (unique in a tree).

            In the image:
            - Vertices are numbered from 1 to N
            - Edges connect vertices to form a tree structure
            - Each triad forms an equilateral triangle in distance space

            Output format: A single integer representing the count of such triads.
            Example: "5"
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
        """Generate a random tree and compute the answer."""
        # Generate N between 4 and max_n (need at least 4 for triads)
        N = int(self.np_random.integers(4, self._max_n + 1))
        self._N = N

        # Generate random tree using Prüfer-like approach
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
        self._edges = edges

        # Verify tree properties
        assert len(edges) == len(set(edges)) == N - 1
        for u, v in edges:
            assert 1 <= u < v <= N

        # Build adjacency list
        adjacency = [[] for _ in range(N + 1)]
        for a, b in edges:
            adjacency[a].append(b)
            adjacency[b].append(a)

        ans = 0

        # For each candidate center c, we look at its branches (one per neighbor).
        # In each branch we BFS to record how many nodes lie at each distance d from c.
        # Then for each distance d we have counts [c1, c2, ..., ck] across branches,
        # and the number of ways to pick one node in three distinct branches all at that
        # same distance is the 3rd elementary symmetric sum:
        #    e3 = sum_{i<j<k} ci*cj*ck = (S1^3 - 3 S1 S2 + 2 S3)/6,
        # where S1 = sum ci, S2 = sum ci^2, S3 = sum ci^3.

        for c in range(1, N + 1):
            if len(adjacency[c]) < 3:
                continue  # need at least 3 branches

            visited = [False] * (N + 1)
            visited[c] = True

            branch_counts = []
            max_depth = 0

            # BFS each branch separately, marking visited to avoid overlap
            for nbr in adjacency[c]:
                if visited[nbr]:
                    continue
                visited[nbr] = True
                q = deque([(nbr, 1)])
                local = []  # local[d] = number of nodes at distance d in this branch
                while q:
                    u, d = q.popleft()
                    # ensure local is long enough
                    if d >= len(local):
                        local.extend([0] * (d - len(local) + 1))
                    local[d] += 1
                    if d > max_depth:
                        max_depth = d
                    for w in adjacency[u]:
                        if not visited[w]:
                            visited[w] = True
                            q.append((w, d + 1))
                branch_counts.append(local)

            b = len(branch_counts)
            if b < 3:
                continue

            # for each possible distance t, compute the 3-way product sum
            for t in range(1, max_depth + 1):
                S1 = S2 = S3 = 0
                for f in branch_counts:
                    cnt = f[t] if t < len(f) else 0
                    S1 += cnt
                    S2 += cnt * cnt
                    S3 += cnt * cnt * cnt
                # elementary symmetric sum of order 3
                e3 = (S1 * S1 * S1 - 3 * S1 * S2 + 2 * S3) // 6
                ans += e3

        self._oracle_answer = ans

    def _prompt_generate(self) -> str:
        N = self._N
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges="\n".join(f"{u} {v}" for u, v in self._edges),
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

    def _score_answer(self, output: str) -> float:
        """Score the answer based on correctness.

        Returns:
            -1.0: wrong format (not an integer or negative integer)
             0.0: wrong answer
             1.0: correct answer
        """
        processed_result = self._process(output)
        if processed_result is not None:
            if processed_result < 0:
                return 0.0
            if processed_result == self._oracle_answer:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    def render(self) -> Image.Image:
        """Render the tree with a circular layout."""
        if self._N is None:
            return Image.new(
                "RGB", (self._image_size, self._image_size), (240, 245, 248)
            )

        N = self._N
        edges = self._edges

        # Use soft blue-gray background
        img = Image.new("RGB", (self._image_size, self._image_size), (240, 245, 248))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 22)  # Node label font
            title_font = ImageFont.truetype(str(font_path), 28)  # Title font
            info_font = ImageFont.truetype(str(font_path), 20)  # Info font
        else:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()

        # Add title
        title = "Tree Distance Equal Triad Counting"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, 20),
            title,
            fill=(50, 50, 70),
            font=title_font,
        )

        # Calculate circular layout
        center_x = self._image_size // 2
        center_y = self._image_size // 2 + 20
        radius = (self._image_size - 2 * self._padding - 60) // 2

        positions = {}
        for i in range(1, N + 1):
            angle = 2 * math.pi * (i - 1) / N - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions[i] = (x, y)

        # Draw edges with soft gray
        for u, v in edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            draw.line([(x1, y1), (x2, y2)], fill=(120, 140, 160), width=3)

        # Draw nodes
        for i in range(1, N + 1):
            x, y = positions[i]

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

            # Draw node circle (bright orange)
            draw.ellipse(
                [
                    x - self._node_radius,
                    y - self._node_radius,
                    x + self._node_radius,
                    y + self._node_radius,
                ],
                fill=(255, 165, 60),  # Bright orange
                outline=(200, 120, 40),  # Dark orange border
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
                fill=(255, 255, 255),  # White text
                font=font,
            )

        # Add info text
        info_text = (
            f"Vertices: {N}  |  Edges: {len(edges)}  |  Count equal-distance triads"
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

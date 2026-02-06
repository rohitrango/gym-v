"""Minimum Chromatic Number environment for gym-v."""

from __future__ import annotations

from importlib import resources
import math
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class MinimumChromaticNumberEnv(Env):
    # Meta: source=RLVE, category=graphs, turn=single
    """RLVE Minimum Chromatic Number as a single-turn visual environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.
The graph contains the following undirected edges:
{edges}

Your task is to assign a **non-negative integer color** to each vertex, represented as `c[0], c[1], ..., c[{N_minus_1}]`, such that:
- For every edge `(u, v)` in the graph, `c[u] ≠ c[v]` — adjacent vertices must have different colors.
- The total number of **distinct colors used** (i.e., the number of unique values among `c[0]` to `c[{N_minus_1}]`) is **minimized** - try your best to find a valid coloring using as few colors as possible.

**Output Format:**
Your final answer should be a single line containing the color of each vertex in order: `c[0], c[1], ..., c[{N_minus_1}]`, separated by **spaces**.
Example: `0 1 0 2` (do **NOT** include the backticks or quotes); this means vertex 0 is assigned color 0, vertex 1 color 1, vertex 2 color 0, and vertex 3 color 2 (assuming 4 vertices in total)."""

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
        self._rewarding_strategy = rewarding_strategy
        self._rewarding_weight = rewarding_weight
        self._rewarding_beta = rewarding_beta

        self._N: int | None = None
        self._edges: list[tuple[int, int]] | None = None
        self._gold_answer: int | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """\
            Minimum Chromatic Number (Graph Coloring) Problem:

            Given an undirected graph with N vertices, assign a color to each vertex
            such that no two adjacent vertices share the same color, using the
            minimum number of distinct colors possible.

            In the image:
            - Vertices are numbered and shown as circles arranged in a circle
            - Edges are shown as lines connecting vertices
            - Find the minimum number of colors needed to color the graph

            Output Format:
            Your final answer should be a single line containing the color of each
            vertex in order: c[0], c[1], ..., c[N-1], separated by spaces.
            Example: 0 1 0 2 (do NOT include backticks or quotes); this means
            vertex 0 is assigned color 0, vertex 1 color 1, vertex 2 color 0, and
            vertex 3 color 2 (assuming 4 vertices in total)."""
        )

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
        """Generate a random graph and find its chromatic number."""
        N = int(self.np_random.integers(2, self._max_n + 1))
        self._N = N

        num_edges = int(self._edge_density * N * (N - 1) / 2)
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        edge_indices = list(
            self.np_random.choice(len(all_edges), size=num_edges, replace=False)
        )
        edges = [all_edges[i] for i in edge_indices]
        self.np_random.shuffle(edges)
        self._edges = edges

        # Build adjacency information using bitsets
        adjacent = [0] * N
        for u, v in edges:
            adjacent[u] |= 1 << v
            adjacent[v] |= 1 << u

        # Find optimal coloring using backtracking
        colors = [None] * N
        color2set = [0] * N
        reference_answer = list(range(N))
        gold_answer = N

        def DFS(u: int, max_color: int) -> None:
            nonlocal reference_answer, gold_answer
            if max_color + 1 >= gold_answer:
                return
            if u == N:
                reference_answer = colors.copy()
                gold_answer = max_color + 1
                return
            for color in range((max_color + 1) + 1):
                if (color2set[color] & adjacent[u]) == 0:
                    colors[u] = color
                    color2set[color] += 1 << u
                    DFS(u + 1, max(max_color, color))
                    color2set[color] -= 1 << u

        DFS(0, -1)

        self._gold_answer = gold_answer
        self._oracle_answer = " ".join(map(str, reference_answer))

    def _prompt_generate(self) -> str:
        N = self._N
        edges = self._edges
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges="\n".join(f"({u}, {v})" for u, v in edges),
        )

    def _process(self, answer: str | None) -> list[int] | None:
        if answer is None:
            return None
        answer = answer.strip()
        try:
            answer_array = list(map(int, answer.split()))
            return answer_array
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer using RLVE's scoring logic."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        colors = processed_result
        N = self._N

        # Check if answer has correct length
        if len(colors) != N:
            return 0.0
        for u, v in self._edges:
            if colors[u] == colors[v]:
                return 0.0
        gold = self._gold_answer
        answer_num_colors = len(set(colors))

        if answer_num_colors < gold:
            # Should not happen if gold is truly optimal
            return self._rewarding_weight

        if self._rewarding_strategy == "(gold/answer)^beta":
            return self._rewarding_weight * (
                (gold / answer_num_colors) ** self._rewarding_beta
            )
        elif self._rewarding_strategy == "gold=answer":
            return self._rewarding_weight * (answer_num_colors == gold)
        else:
            raise NotImplementedError(
                f"Unknown rewarding strategy: {self._rewarding_strategy}"
            )

    def render(self) -> Image.Image:
        """Render the graph as a beautiful circular layout."""
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
            title_font = ImageFont.truetype(str(font_path), 24)
        else:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()

        # Add title
        title = "Minimum Chromatic Number (Graph Coloring)"
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

        positions = []
        for i in range(N):
            angle = 2 * math.pi * i / N - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y))

        # Draw edges
        for u, v in edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            draw.line([(x1, y1), (x2, y2)], fill=(120, 140, 160), width=3)

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
                fill=(70, 150, 220),
                outline=(40, 100, 170),
                width=3,
            )

            # Draw node number
            text = str(i)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
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

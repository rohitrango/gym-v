"""Graph Isomorphism environment for gym-v."""

from __future__ import annotations

from importlib import resources
import math
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEGraphIsomorphismEnv(Env):
    """RLVE Graph Isomorphism as a single-turn visual environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given two **undirected graphs**, G1 and G2, each with {N} vertices labeled from `0` to `{N_minus_1}`. Both graphs contain exactly {M} **undirected** edges.

- Graph G1 has the following (undirected) edge set E1:
{G1_edges}

- Graph G2 has the following (undirected) edge set E2:
{G2_edges}

Your task is to find a **bijection** (i.e., a permutation) `p` from the vertices of G1 to the vertices of G2 such that: For every edge `(u, v)` in E1, the edge `(p(u), p(v))` exists in E2, and vice versa.

**Output Format:** Your final answer should be a single line containing the permutation `p(0), p(1), ..., p({N_minus_1})`, separated by spaces. Example: `{reversed_permutation}` (do **NOT** include backticks or quotes); this means `p(0) = {N_minus_1}, ..., p({N_minus_1}) = 0`."""

    def __init__(
        self,
        max_n: int = 8,
        edge_density: float = 0.3,
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
        self._G1_edges: list[tuple[int, int]] | None = None
        self._G2_edges: list[tuple[int, int]] | None = None
        self._mapping: list[int] | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Graph Isomorphism Problem:

            Given two undirected graphs G1 and G2 with the same number of vertices,
            find a bijection (permutation) p from vertices of G1 to vertices of G2
            such that for every edge (u, v) in G1, the edge (p(u), p(v)) exists in G2.

            In the image:
            - Left: Graph G1
            - Right: Graph G2
            - Vertices are numbered and shown as circles
            - Edges are shown as lines connecting vertices

            Output format: A single line with the permutation p(0), p(1), ..., p(N-1)
            separated by spaces.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the two graphs."""
        if self._N is None or self._G1_edges is None or self._G2_edges is None:
            return ""
        lines = [
            f"N = {self._N} vertices",
            "",
            "Graph G1 edges:",
        ]
        for u, v in self._G1_edges:
            lines.append(f"  ({u}, {v})")
        lines.append("")
        lines.append("Graph G2 edges:")
        for u, v in self._G2_edges:
            lines.append(f"  ({u}, {v})")
        return "\n".join(lines)

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
        N = int(self.np_random.integers(3, self._max_n + 1))
        self._N = N

        num_edges = max(1, int(self._edge_density * N * (N - 1) / 2))
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        G1_edges = list(
            self.np_random.choice(len(all_edges), size=num_edges, replace=False)
        )
        G1_edges = [all_edges[i] for i in G1_edges]
        self.np_random.shuffle(G1_edges)
        self._G1_edges = G1_edges

        mapping = list(range(N))
        self.np_random.shuffle(mapping)
        self._mapping = mapping

        G2_edges = []
        for u, v in G1_edges:
            u2, v2 = mapping[u], mapping[v]
            if u2 > v2:
                u2, v2 = v2, u2
            G2_edges.append((u2, v2))
        self.np_random.shuffle(G2_edges)
        self._G2_edges = G2_edges

        self._oracle_answer = " ".join(map(str, mapping))

    def _prompt_generate(self) -> str:
        N = self._N
        G1_edges = self._G1_edges
        G2_edges = self._G2_edges
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            M=len(G1_edges),
            G1_edges="\n".join(f"({u}, {v})" for u, v in G1_edges),
            G2_edges="\n".join(f"({u}, {v})" for u, v in G2_edges),
            reversed_permutation=" ".join(map(str, range(N - 1, -1, -1))),
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
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        permutation = processed_result
        N = self._N
        if len(permutation) != N:
            return 0.0
        if len(set(permutation)) != N:
            return 0.0
        if not all(0 <= i < N for i in permutation):
            return 0.0
        new_G2_edges = set()
        for u, v in self._G1_edges:
            u2, v2 = permutation[u], permutation[v]
            if u2 > v2:
                u2, v2 = v2, u2
            new_G2_edges.add((u2, v2))

        overlap = len(new_G2_edges & set(self._G2_edges))
        return (overlap / len(self._G2_edges)) ** 10

    def render(self) -> Image.Image:
        if self._N is None:
            return Image.new(
                "RGB", (self._image_size, self._image_size // 2), (250, 250, 250)
            )

        N = self._N
        G1_edges = self._G1_edges
        G2_edges = self._G2_edges

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

        def draw_graph(edges, offset_x, title):
            center_x = offset_x + graph_width // 2
            center_y = height // 2
            radius = min(graph_width, height) // 2 - self._padding

            # Draw title with more padding from top
            bbox = draw.textbbox((0, 0), title, font=title_font)
            tw = bbox[2] - bbox[0]
            draw.text(
                (center_x - tw // 2, self._padding // 4),
                title,
                fill=(0, 0, 0),
                font=title_font,
            )

            # Rotate G2 by 180 degrees visually to make it look different from G1
            angle_offset = math.pi if "G2" in title else 0

            positions = []

            for i in range(N):
                angle = 2 * math.pi * i / N - math.pi / 2 + angle_offset
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                positions.append((x, y))

            for u, v in edges:
                x1, y1 = positions[u]
                x2, y2 = positions[v]
                draw.line([(x1, y1), (x2, y2)], fill=(150, 150, 150), width=2)

            for i, (x, y) in enumerate(positions):
                draw.ellipse(
                    [
                        x - self._node_radius,
                        y - self._node_radius,
                        x + self._node_radius,
                        y + self._node_radius,
                    ],
                    fill=(100, 150, 255),
                    outline=(30, 30, 30),
                    width=2,
                )

                text = str(i)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.text(
                    (x - tw // 2, y - th // 2), text, fill=(255, 255, 255), font=font
                )

        draw_graph(G1_edges, 0, "Graph G1")
        draw_graph(G2_edges, graph_width, "Graph G2")

        draw.line(
            [(graph_width, self._padding), (graph_width, height - self._padding)],
            fill=(100, 100, 100),
            width=2,
        )

        return img

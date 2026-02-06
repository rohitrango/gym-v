"""Mixed Graph Eulerian Circuit environment for gym-v (self-contained)."""

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


class MixedGraphEulerianCircuitEnv(Env):
    # Meta: source=RLVE, category=graphs, turn=single
    """RLVE Mixed Graph Eulerian Circuit as a single-turn visual environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a **graph** with {N} vertices labeled from 0 to {N_minus_1}.

The graph contains the following **undirected** edges:
{undirected_edges}

It also contains the following **directed** edges (each `<u, v>` represents a directed edge from vertex `u` to vertex `v`):
{directed_edges}

It is guaranteed that if all directed edges are treated as undirected, the resulting graph is connected and has no repeated edges, and every vertex has an even degree.

Please find an **Eulerian circuit** in this graph — a closed path that starts and ends at the same vertex and **visits each edge exactly once**.
Output a single line containing the sequence of vertex labels visited in order, separated by spaces."""

    def __init__(
        self,
        max_n: int = 10,
        image_size: int = 800,
        padding: int = 80,
        node_radius: int = 20,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._image_size = image_size
        self._padding = padding
        self._node_radius = node_radius
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._N: int | None = None
        self._undirected_edges: list[tuple[int, int]] | None = None
        self._directed_edges: list[tuple[int, int]] | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            You are given a graph with N vertices labeled from 0 to N-1.

            The graph contains undirected edges and directed edges. It is guaranteed
            that if all directed edges are treated as undirected, the resulting graph
            is connected and has no repeated edges, and every vertex has an even
            degree.

            Please find an Eulerian circuit in this graph - a closed path that starts
            and ends at the same vertex and visits each edge exactly once.

            In the image:
            - Vertices are numbered and shown as blue circles
            - Undirected edges are shown as plain lines (no arrows)
            - Directed edges are shown with arrows indicating direction
            - All edges must be traversed exactly once, respecting edge directions

            Output a single line containing the sequence of vertex labels visited in
            order, separated by spaces.
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
        N = int(self.np_random.integers(3, self._max_n + 1))
        self._N = N

        while True:
            degrees = [0] * N
            edges = []
            for v in range(1, N - 1):
                neighbors = list(
                    self.np_random.choice(
                        v, size=int(self.np_random.integers(0, v + 1)), replace=False
                    )
                )
                for u in neighbors:
                    assert u < v, "Undirected edges should be added in increasing order"
                    edges.append((u, v))
                    degrees[u] += 1
                    degrees[v] += 1
            for u in range(N - 1):
                if degrees[u] % 2 == 1:
                    v = N - 1
                    edges.append((u, v))
                    degrees[u] += 1
                    degrees[v] += 1
            assert all(
                degree % 2 == 0 for degree in degrees
            ), "All vertices should have even degree in undirected edges"

            self.np_random.shuffle(edges)
            assert len(edges) == len(
                set(edges)
            ), "There should be no repeated undirected edges"
            for u, v in edges:
                assert (
                    0 <= u < v < N
                ), "Undirected edges should be within the range of vertex labels"

            # Check if the undirected graph is connected
            undirected_graph = nx.Graph()
            undirected_graph.add_nodes_from(range(N))
            undirected_graph.add_edges_from(edges)
            if nx.is_connected(undirected_graph):
                assert nx.is_eulerian(
                    undirected_graph
                ), "The undirected graph should be Eulerian"
                break

        eulerian_circuit = list(nx.eulerian_circuit(undirected_graph))
        assert len(eulerian_circuit) == len(
            edges
        ), "The Eulerian circuit should visit each edge exactly once"
        directed_flags = [False] * len(eulerian_circuit)
        num_directed = int(
            self.np_random.integers(1, len(eulerian_circuit))
        )  # At least 1, at most all-1
        flagged_indices = self.np_random.choice(
            len(eulerian_circuit), size=num_directed, replace=False
        )
        for flagged in flagged_indices:
            directed_flags[flagged] = True

        undirected_edges = []
        directed_edges = []
        reference_answer = []
        for (u, v), directed_flag in zip(
            eulerian_circuit, directed_flags, strict=False
        ):
            reference_answer.append(u)
            if directed_flag:
                directed_edges.append((u, v))
            else:
                undirected_edges.append((min(u, v), max(u, v)))
        reference_answer.append(eulerian_circuit[-1][1])
        assert (
            reference_answer[0] == reference_answer[-1]
        ), "The Eulerian circuit should start and end at the same vertex"
        self._oracle_answer = " ".join(map(str, reference_answer))
        assert (
            len(undirected_edges) > 0 and len(directed_edges) > 0
        ), "There should be at least one undirected edge and one directed edge"
        self.np_random.shuffle(undirected_edges)
        self.np_random.shuffle(directed_edges)
        self._undirected_edges = undirected_edges
        self._directed_edges = directed_edges

    def _prompt_generate(self) -> str:
        N = self._N
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            undirected_edges="\n".join(
                f"({u}, {v})" for u, v in self._undirected_edges
            ),
            directed_edges="\n".join(f"<{u}, {v}>" for u, v in self._directed_edges),
        )

    def _process(self, answer: str | None) -> list[int] | None:
        if answer is not None:
            answer = answer.strip()
            try:
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError:
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        processed_result = self._process(answer)
        if processed_result is not None:
            assert isinstance(
                processed_result, list
            ), "processed_result should be a list"

            if len(processed_result) == 0:
                return 0.0
            if not all(0 <= u < self._N for u in processed_result):
                return 0.0
            undirected_edges = {(u, v): 0 for u, v in self._undirected_edges}
            directed_edges = {(u, v): 0 for u, v in self._directed_edges}
            if processed_result[0] != processed_result[-1]:
                return 0.0
            for u, v in zip(processed_result, processed_result[1:], strict=False):
                directed = (u, v) in directed_edges
                undirected = (min(u, v), max(u, v)) in undirected_edges
                assert int(directed) + int(undirected) <= 1
                if directed:
                    directed_edges[(u, v)] += 1
                elif undirected:
                    undirected_edges[(min(u, v), max(u, v))] += 1
                else:
                    return 0.0
            satisfied = sum(count == 1 for count in directed_edges.values()) + sum(
                count == 1 for count in undirected_edges.values()
            )
            assert satisfied <= len(self._undirected_edges) + len(
                self._directed_edges
            ), "satisfied should be less than or equal to the total number of edges"
            # Default rewarding strategy: (satisfied/all)^beta with beta=10
            return (
                satisfied / (len(self._undirected_edges) + len(self._directed_edges))
            ) ** 10
        else:
            return 0.0

    def render(self) -> Image.Image:
        if self._N is None:
            return Image.new(
                "RGB", (self._image_size, self._image_size), (250, 250, 250)
            )

        N = self._N
        undirected_edges = self._undirected_edges
        directed_edges = self._directed_edges

        img = Image.new("RGB", (self._image_size, self._image_size), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 16)
            title_font = ImageFont.truetype(str(font_path), 24)
            legend_font = ImageFont.truetype(str(font_path), 14)
        else:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            legend_font = ImageFont.load_default()

        # Draw title
        title = "Mixed Graph Eulerian Circuit"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, self._padding // 4),
            title,
            fill=(0, 0, 0),
            font=title_font,
        )

        # Calculate positions using circular layout
        center_x = self._image_size // 2
        center_y = self._image_size // 2
        radius = min(
            self._image_size // 2 - self._padding, self._image_size // 2 - self._padding
        )

        positions = []
        for i in range(N):
            angle = 2 * math.pi * i / N - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y))

        # Draw undirected edges first (no arrows)
        for u, v in undirected_edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            draw.line([(x1, y1), (x2, y2)], fill=(150, 150, 150), width=2)

        # Draw directed edges with arrows
        for u, v in directed_edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]

            # Calculate arrow endpoint (stop at node boundary)
            dx, dy = x2 - x1, y2 - y1
            length = math.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx / length, dy / length
                arrow_end_x = x2 - dx * (self._node_radius + 5)
                arrow_end_y = y2 - dy * (self._node_radius + 5)

                # Draw edge line
                draw.line(
                    [(x1, y1), (arrow_end_x, arrow_end_y)], fill=(50, 100, 200), width=2
                )

                # Draw arrowhead
                arrow_size = 12
                angle = math.atan2(dy, dx)
                arrow_left = (
                    arrow_end_x - arrow_size * math.cos(angle - math.pi / 6),
                    arrow_end_y - arrow_size * math.sin(angle - math.pi / 6),
                )
                arrow_right = (
                    arrow_end_x - arrow_size * math.cos(angle + math.pi / 6),
                    arrow_end_y - arrow_size * math.sin(angle + math.pi / 6),
                )
                draw.polygon(
                    [(arrow_end_x, arrow_end_y), arrow_left, arrow_right],
                    fill=(50, 100, 200),
                )

        # Draw nodes
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
            draw.text((x - tw // 2, y - th // 2), text, fill=(255, 255, 255), font=font)

        # Draw legend
        legend_y = self._image_size - self._padding // 2 - 10
        legend_x = self._padding

        # Undirected edge example
        draw.line(
            [(legend_x, legend_y), (legend_x + 40, legend_y)],
            fill=(150, 150, 150),
            width=2,
        )
        draw.text(
            (legend_x + 50, legend_y - 7),
            "Undirected edge",
            fill=(0, 0, 0),
            font=legend_font,
        )

        # Directed edge example
        legend_x += 200
        draw.line(
            [(legend_x, legend_y), (legend_x + 40, legend_y)],
            fill=(50, 100, 200),
            width=2,
        )
        # Arrow for directed edge
        arrow_size = 10
        draw.polygon(
            [
                (legend_x + 40, legend_y),
                (legend_x + 40 - arrow_size, legend_y - arrow_size // 2),
                (legend_x + 40 - arrow_size, legend_y + arrow_size // 2),
            ],
            fill=(50, 100, 200),
        )
        draw.text(
            (legend_x + 50, legend_y - 7),
            "Directed edge",
            fill=(0, 0, 0),
            font=legend_font,
        )

        return img

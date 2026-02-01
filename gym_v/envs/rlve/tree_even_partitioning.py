"""Tree Even Partitioning environment for gym-v (self-contained)."""

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


class RLVETreeEvenPartitioningEnv(Env):
    """RLVE Tree Even Partitioning as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You have a **tree** (i.e., a connected undirected graph with no cycles) with {NK} vertices labeled from `1` to `{NK}`. The tree contains the following {NK} - 1 undirected edges. Each edge is represented as a tuple `(u, v)`, meaning there is an undirected edge connecting vertex `u` to vertex `v`:
{edges}

Partition all vertices into {N} **disjoint** sets such that: (1) each set contains exactly {K} vertices ({K} = {NK} / {N}), AND (2) each set forms a connected subgraph of the tree. Output {N} lines - each line should contain the {K} vertices of one set, separated by spaces; the vertices within a set and the sets themselves may be in any order."""

    def __init__(
        self,
        max_n: int = 4,
        max_k: int = 3,
        node_radius: int = 20,
        image_size: int = 800,
        padding: int = 80,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._max_k = max_k
        self._node_radius = node_radius
        self._image_size = image_size
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._N: int | None = None
        self._K: int | None = None
        self._edges: list[tuple[int, int]] | None = None
        self._groups: list[list[int]] | None = None
        self._oracle_answer: str | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Tree Even Partitioning Problem:

            Given a tree (connected undirected graph with no cycles) with N*K vertices,
            partition all vertices into N disjoint sets such that:
            1) Each set contains exactly K vertices
            2) Each set forms a connected subgraph of the tree

            In the image:
            - Vertices are numbered from 1 to N*K and shown as circles
            - Edges connect vertices with lines
            - Each partition is shown with a different color
            - The tree structure is laid out clearly
            - The visualization shows the solution with colored partitions

            Output format: N lines, each containing K space-separated vertex numbers.
            Example:
            1 2 3
            4 5 6
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
        """Generate tree even partitioning problem instance."""
        N = int(self.np_random.integers(2, self._max_n + 1))
        K = int(self.np_random.integers(2, self._max_k + 1))
        self._N = N
        self._K = K

        # Create N groups of K vertices each
        groups = list(range(1, N * K + 1))
        self.np_random.shuffle(groups)
        groups = [groups[i * K : (i + 1) * K] for i in range(N)]
        self._groups = groups

        edges = []

        # Generate tree structure: for each group, connect vertices within the group
        for i, group in enumerate(groups):
            assert len(group) == K, f"Group {i} should have exactly {K} vertices"
            for index, vertex in enumerate(group):
                if index == 0:
                    continue
                # Connect to a random previous vertex in the same group
                prev_vertex = group[int(self.np_random.integers(0, index))]
                u, v = min(vertex, prev_vertex), max(vertex, prev_vertex)
                edges.append((u, v))

            # Connect this group to a previous group
            if i == 0:
                continue
            # Pick random vertex from current group
            u = int(self.np_random.choice(group))
            # Pick random vertex from a random previous group
            prev_group_idx = int(self.np_random.integers(0, i))
            v = int(self.np_random.choice(groups[prev_group_idx]))
            u, v = min(u, v), max(u, v)
            edges.append((u, v))

        self.np_random.shuffle(edges)
        self._edges = edges

        # Verify it's a valid tree
        for u, v in edges:
            assert 1 <= u < v <= N * K
        assert len(edges) == len(set(edges)) == N * K - 1

        tree = nx.Graph()
        tree.add_edges_from(edges)
        assert nx.is_tree(tree)

        self._oracle_answer = "\n".join(" ".join(map(str, group)) for group in groups)

    def _prompt_generate(self) -> str:
        N, K = self._N, self._K
        return self.prompt_template.format(
            NK=N * K,
            N=N,
            K=K,
            edges="\n".join(f"({u}, {v})" for u, v in self._edges),
        )

    def _process(self, answer: str | None) -> list[list[int]] | None:
        if answer is not None:
            answer = answer.strip()
            try:
                groups = []
                for line in answer.splitlines():
                    line = line.strip()
                    if line:
                        groups.append(list(map(int, line.split())))
                        if len(groups[-1]) != self._K:
                            return None
                if len(groups) != self._N:
                    return None
                return groups
            except ValueError:
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        if set(vertex for group in processed_result for vertex in group) != set(
            range(1, self._N * self._K + 1)
        ):
            return 0.0
        labels = [None] * (self._N * self._K + 1)
        for label, group in enumerate(processed_result):
            assert 0 <= label < self._N, f"Label {label} is out of range"
            assert (
                len(group) == self._K
            ), f"Group {group} should have exactly {self._K} vertices"
            for vertex in group:
                assert labels[vertex] is None, f"Vertex {vertex} is already labeled"
                labels[vertex] = label

        # Count edges within each group (to check connectivity)
        edge_numbers = [0] * self._N
        for u, v in self._edges:
            if labels[u] == labels[v]:
                edge_numbers[labels[u]] += 1

        # For a group of K vertices to be connected in a tree, it needs exactly K-1 edges
        assert all(
            0 <= edge_number <= self._K - 1 for edge_number in edge_numbers
        ), "Edge numbers are out of range"
        connected = sum(int(edge_number == self._K - 1) for edge_number in edge_numbers)
        assert connected <= self._N, "Connected components exceed N"

        # Reward: (connected/all)^beta with beta=5
        return (connected / self._N) ** 5

    def render(self) -> Image.Image:
        if self._N is None:
            return Image.new(
                "RGB", (self._image_size, self._image_size), (240, 245, 248)
            )

        N = self._N
        K = self._K
        NK = N * K
        edges = self._edges
        groups = self._groups

        # Create image with soft blue-gray background
        img = Image.new("RGB", (self._image_size, self._image_size), (240, 245, 248))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 18)
            title_font = ImageFont.truetype(str(font_path), 26)
            info_font = ImageFont.truetype(str(font_path), 18)
        else:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()

        # Add title
        title = "Tree Even Partitioning Problem"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, 20),
            title,
            fill=(50, 50, 70),
            font=title_font,
        )

        # Create vertex to group mapping
        vertex_to_group = {}
        for group_idx, group in enumerate(groups):
            for vertex in group:
                vertex_to_group[vertex] = group_idx

        # Define colors for different partitions
        partition_colors = [
            (255, 182, 193),  # Light pink
            (173, 216, 230),  # Light blue
            (144, 238, 144),  # Light green
            (255, 218, 185),  # Peach
            (221, 160, 221),  # Plum
            (255, 250, 205),  # Lemon chiffon
            (176, 224, 230),  # Powder blue
            (255, 228, 196),  # Bisque
        ]

        # Build NetworkX graph for layout
        G = nx.Graph()
        G.add_edges_from(edges)

        # Use spring layout for better tree visualization
        try:
            # Try hierarchical layout first (better for trees)
            from networkx.drawing.nx_agraph import graphviz_layout

            pos = graphviz_layout(G, prog="dot")
        except:
            # Fallback to spring layout
            pos = nx.spring_layout(G, seed=42, k=2 / math.sqrt(NK))

        # Scale positions to image coordinates with margins
        margin = self._padding
        width = self._image_size - 2 * margin
        height = self._image_size - 2 * margin - 100

        # Normalize positions
        xs = [x for x, y in pos.values()]
        ys = [y for x, y in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        x_range = max_x - min_x if max_x > min_x else 1
        y_range = max_y - min_y if max_y > min_y else 1

        scaled_pos = {}
        for node, (x, y) in pos.items():
            scaled_x = margin + ((x - min_x) / x_range) * width
            scaled_y = margin + 60 + ((y - min_y) / y_range) * height
            scaled_pos[node] = (scaled_x, scaled_y)

        # Draw edges
        for u, v in edges:
            x1, y1 = scaled_pos[u]
            x2, y2 = scaled_pos[v]

            # Use different edge color if vertices are in same partition
            if vertex_to_group[u] == vertex_to_group[v]:
                # Same partition: thicker, darker edge
                draw.line([(x1, y1), (x2, y2)], fill=(80, 80, 80), width=3)
            else:
                # Different partitions: thinner, lighter edge
                draw.line([(x1, y1), (x2, y2)], fill=(150, 150, 150), width=2)

        # Draw nodes with partition colors
        for vertex in range(1, NK + 1):
            x, y = scaled_pos[vertex]
            group_idx = vertex_to_group[vertex]

            # Get color for this partition
            node_color = partition_colors[group_idx % len(partition_colors)]
            outline_color = tuple(max(0, c - 60) for c in node_color)

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

            # Draw node circle
            draw.ellipse(
                [
                    x - self._node_radius,
                    y - self._node_radius,
                    x + self._node_radius,
                    y + self._node_radius,
                ],
                fill=node_color,
                outline=outline_color,
                width=2,
            )

            # Draw node label
            text = str(vertex)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x - tw // 2, y - th // 2),
                text,
                fill=(0, 0, 0),
                font=font,
            )

        # Add legend showing partitions
        legend_y = self._image_size - 60
        legend_text = f"Partitions: {N} groups of {K} vertices each"
        bbox = draw.textbbox((0, 0), legend_text, font=info_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, legend_y),
            legend_text,
            fill=(100, 100, 120),
            font=info_font,
        )

        # Draw color legend for partitions
        legend_start_y = legend_y + 25
        box_size = 15
        spacing = 10

        # Calculate total width of legend
        total_width = N * (box_size + spacing) - spacing
        start_x = (self._image_size - total_width) // 2

        for i in range(N):
            x = start_x + i * (box_size + spacing)
            color = partition_colors[i % len(partition_colors)]
            outline = tuple(max(0, c - 60) for c in color)

            # Draw color box
            draw.rectangle(
                [x, legend_start_y, x + box_size, legend_start_y + box_size],
                fill=color,
                outline=outline,
                width=1,
            )

            # Draw partition number
            label = str(i + 1)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x + box_size // 2 - tw // 2, legend_start_y + box_size // 2 - th // 2),
                label,
                fill=(0, 0, 0),
                font=font,
            )

        return img

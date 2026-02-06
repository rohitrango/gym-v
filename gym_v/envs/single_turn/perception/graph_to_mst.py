"""Graph to MST (Minimum Spanning Tree) perception environment."""

from __future__ import annotations

import io
import json
import logging
import random
import string
from textwrap import dedent
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image, ImageFilter

from gym_v import Env, Observation, get_logger

logger = get_logger()

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


class GraphToMSTEnv(Env):
    # Meta: source=Perception, category=perception, turn=single
    """Graph to MST perception environment.

    The agent must perceive a weighted undirected graph visualization
    and identify the edges that form the Minimum Spanning Tree.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (640, 480),
        min_nodes: int = 5,
        max_nodes: int = 8,
        min_weight: int = 1,
        max_weight: int = 20,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._current_graph: nx.Graph | None = None
        self._current_mst_edges: list[list[str | int]] | None = None
        self._current_mst_weight: int | None = None
        self._current_image: Image.Image | None = None

        self._node_colors = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFEAA7",
            "#DDA0DD",
            "#98D8C8",
            "#F7DC6F",
            "#BB8FCE",
            "#85C1E9",
            "#F8B500",
            "#00CED1",
        ]

    @property
    def description(self) -> str:
        return dedent("""
            You are given a weighted undirected graph visualization.
            Your task is to identify the edges that form the Minimum Spanning Tree (MST).

            The MST is a subset of edges that connects all nodes with the minimum total weight.

            Output the MST edges as a list of [node1, node2, weight] tuples, and the total weight.

            Example: {"mst_edges": [["A", "B", 3], ["B", "C", 2], ["C", "D", 4]], "total_weight": 9}

            Output ONLY the JSON string.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self._generate_new_problem()

        obs = Observation(
            image=self._current_image,
            text=None,
            metadata={
                "num_nodes": self._current_graph.number_of_nodes(),
                "num_edges": self._current_graph.number_of_edges(),
            },
        )

        info = {
            "oracle_answer": json.dumps(
                {
                    "mst_edges": self._current_mst_edges,
                    "total_weight": self._current_mst_weight,
                }
            ),
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
        reward = self._compute_reward(action_str)
        info = {
            "oracle_answer": json.dumps(
                {
                    "mst_edges": self._current_mst_edges,
                    "total_weight": self._current_mst_weight,
                }
            ),
        }

        obs = Observation(image=self._current_image, text=None)

        return (
            {agent_id: obs for agent_id in self._agent_ids},
            {agent_id: reward for agent_id in self._agent_ids},
            {
                **{agent_id: True for agent_id in self._agent_ids},
                "__all__": True,
            },
            {
                **{agent_id: False for agent_id in self._agent_ids},
                "__all__": False,
            },
            {agent_id: info for agent_id in self._agent_ids},
        )

    def _compute_reward(self, action: str) -> float:
        """Compute reward by checking if MST is correct."""
        try:
            parsed_action = json.loads(action)
            # Check total weight matches
            if parsed_action.get("total_weight") == self._current_mst_weight:
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    def render(self) -> Image.Image:
        return self._current_image

    def _generate_new_problem(self):
        """Generate a random weighted connected graph and compute MST."""
        num_nodes = random.randint(self.min_nodes, self.max_nodes)
        labels = list(string.ascii_uppercase[:num_nodes])

        # Create a connected weighted graph
        G = nx.Graph()
        G.add_nodes_from(labels)

        # First, create a spanning tree to ensure connectivity
        shuffled = labels.copy()
        random.shuffle(shuffled)
        for i in range(len(shuffled) - 1):
            weight = random.randint(self.min_weight, self.max_weight)
            G.add_edge(shuffled[i], shuffled[i + 1], weight=weight)

        # Add additional random edges
        for i, u in enumerate(labels):
            for j, v in enumerate(labels):
                if i >= j:
                    continue
                if G.has_edge(u, v):
                    continue
                if random.random() < 0.4:
                    weight = random.randint(self.min_weight, self.max_weight)
                    G.add_edge(u, v, weight=weight)

        self._current_graph = G

        # Compute MST
        mst = nx.minimum_spanning_tree(G)
        self._current_mst_edges = []
        self._current_mst_weight = 0

        for u, v, data in mst.edges(data=True):
            weight = data["weight"]
            # Sort nodes alphabetically for consistent output
            if u > v:
                u, v = v, u
            self._current_mst_edges.append([u, v, weight])
            self._current_mst_weight += weight

        # Sort edges for consistent output
        self._current_mst_edges.sort(key=lambda x: (x[0], x[1]))

        self._current_image = self._render_graph(G, mst)

    def _render_graph(self, G: nx.Graph, mst: nx.Graph) -> Image.Image:
        """Render the graph as a PIL Image."""
        fig, ax = plt.subplots(
            figsize=(self.img_size[0] / 100, self.img_size[1] / 100), dpi=100
        )

        # Choose layout
        layout_choice = random.choice(["spring", "circular", "kamada_kawai", "shell"])

        try:
            if layout_choice == "spring":
                pos = nx.spring_layout(G, seed=self._seed, k=2)
            elif layout_choice == "circular":
                pos = nx.circular_layout(G)
            elif layout_choice == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G)
            elif layout_choice == "shell":
                pos = nx.shell_layout(G)
            else:
                pos = nx.spring_layout(G, seed=self._seed)
        except Exception:
            pos = nx.spring_layout(G, seed=self._seed)

        # Node colors
        num_nodes = G.number_of_nodes()
        colors = random.sample(
            self._node_colors * (num_nodes // len(self._node_colors) + 1), num_nodes
        )

        # Draw nodes
        node_size = random.randint(1500, 2200)
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=colors,
            node_size=node_size,
            edgecolors="black",
            linewidths=2,
        )

        # Draw node labels
        font_size = random.randint(12, 15)
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=font_size,
            font_weight="bold",
        )

        # Draw all edges
        edge_width = random.uniform(1.5, 2.5)
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color="#888888",
            width=edge_width,
            alpha=0.7,
        )

        # Draw edge weights
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(
            G,
            pos,
            ax=ax,
            edge_labels=edge_labels,
            font_size=10,
            font_color="darkblue",
            font_weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8
            ),
        )

        # Add title
        if random.random() < 0.8:
            ax.set_title("Weighted Undirected Graph", fontsize=14, fontweight="bold")

        ax.axis("off")
        plt.tight_layout()

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close(fig)

        # Optional noise
        if random.random() < 0.15:
            img = self._add_noise(img)

        return img

    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Add subtle noise or blur to the image."""
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        return img

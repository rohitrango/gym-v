"""Graph to Adjacency perception environment."""

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

# Workaround for networkx + numpy 2.0 compatibility
if not hasattr(np, "alltrue"):
    np.alltrue = np.all

from gym_v import Env, Observation, get_logger

logger = get_logger()

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


class PerceptionGraphToAdjacencyEnv(Env):
    """Graph to Adjacency perception environment.

    The agent must perceive a graph visualization and extract the
    underlying structure as an adjacency list in JSON format.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (640, 480),
        min_nodes: int = 4,
        max_nodes: int = 8,
        edge_probability: float = 0.4,
        allow_directed: bool = True,
        allow_weighted: bool = True,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.edge_probability = edge_probability
        self.allow_directed = allow_directed
        self.allow_weighted = allow_weighted
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._current_graph: nx.Graph | nx.DiGraph | None = None
        self._current_adjacency: dict[str, Any] | None = None
        self._current_image: Image.Image | None = None
        self._is_directed: bool = False
        self._is_weighted: bool = False

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

        self._layouts = ["spring", "circular", "shell", "kamada_kawai", "spectral"]

    @property
    def description(self) -> str:
        return dedent("""
            You are given a graph visualization.
            Your task is to extract the graph structure as a JSON object.

            For undirected graphs, output an adjacency list where each node maps to its neighbors.
            For directed graphs, each node maps to its outgoing neighbors.
            For weighted graphs, neighbors are represented as [neighbor, weight] pairs.

            Undirected Example: {"A": ["B", "C"], "B": ["A"], "C": ["A"]}
            Directed Example: {"A": ["B"], "B": ["C"], "C": []}
            Weighted Example: {"A": [["B", 5], ["C", 3]], "B": [], "C": []}

            Also include metadata about the graph type.
            Full format: {"directed": false, "weighted": false, "adjacency": {...}}

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
                "is_directed": self._is_directed,
                "is_weighted": self._is_weighted,
                "num_nodes": self._current_graph.number_of_nodes(),
                "num_edges": self._current_graph.number_of_edges(),
            },
        )

        info = {
            "oracle_answer": json.dumps(self._current_adjacency),
            "is_directed": self._is_directed,
            "is_weighted": self._is_weighted,
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
        info = {"oracle_answer": json.dumps(self._current_adjacency)}

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
        """Compute reward by comparing action with oracle answer."""
        try:
            parsed_action = json.loads(action)
            if parsed_action == self._current_adjacency:
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    def render(self) -> Image.Image:
        return self._current_image

    def _generate_new_problem(self):
        """Generate a random graph and render it."""
        num_nodes = random.randint(self.min_nodes, self.max_nodes)

        # Decide graph type
        self._is_directed = self.allow_directed and random.random() < 0.4
        self._is_weighted = self.allow_weighted and random.random() < 0.4

        # Generate node labels
        labels = list(string.ascii_uppercase[:num_nodes])

        # Create graph
        if self._is_directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        G.add_nodes_from(labels)

        # Add edges with given probability
        for i, u in enumerate(labels):
            for j, v in enumerate(labels):
                if i >= j and not self._is_directed:
                    continue
                if i == j:
                    continue
                if random.random() < self.edge_probability:
                    if self._is_weighted:
                        weight = random.randint(1, 20)
                        G.add_edge(u, v, weight=weight)
                    else:
                        G.add_edge(u, v)

        # Ensure graph is connected for undirected graphs
        if not self._is_directed and not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                u = random.choice(list(components[i]))
                v = random.choice(list(components[i + 1]))
                if self._is_weighted:
                    G.add_edge(u, v, weight=random.randint(1, 20))
                else:
                    G.add_edge(u, v)

        self._current_graph = G
        self._current_adjacency = self._graph_to_adjacency(G)
        self._current_image = self._render_graph(G)

    def _graph_to_adjacency(self, G: nx.Graph | nx.DiGraph) -> dict[str, Any]:
        """Convert networkx graph to adjacency list format."""
        adjacency = {}

        for node in sorted(G.nodes()):
            if self._is_weighted:
                if self._is_directed:
                    neighbors = [
                        [neighbor, G[node][neighbor]["weight"]]
                        for neighbor in sorted(G.successors(node))
                    ]
                else:
                    neighbors = [
                        [neighbor, G[node][neighbor]["weight"]]
                        for neighbor in sorted(G.neighbors(node))
                    ]
            else:
                if self._is_directed:
                    neighbors = sorted(list(G.successors(node)))
                else:
                    neighbors = sorted(list(G.neighbors(node)))
            adjacency[node] = neighbors

        return {
            "directed": self._is_directed,
            "weighted": self._is_weighted,
            "adjacency": adjacency,
        }

    def _render_graph(self, G: nx.Graph | nx.DiGraph) -> Image.Image:
        """Render the graph as a PIL Image."""
        fig, ax = plt.subplots(
            figsize=(self.img_size[0] / 100, self.img_size[1] / 100), dpi=100
        )

        # Choose layout
        layout_name = random.choice(self._layouts)
        try:
            if layout_name == "spring":
                pos = nx.spring_layout(G, seed=self._seed, k=2)
            elif layout_name == "circular":
                pos = nx.circular_layout(G)
            elif layout_name == "shell":
                pos = nx.shell_layout(G)
            elif layout_name == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G)
            elif layout_name == "spectral":
                pos = nx.spectral_layout(G)
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
        node_size = random.randint(1500, 2500)
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
        font_size = random.randint(12, 16)
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=font_size,
            font_weight="bold",
        )

        # Draw edges
        edge_color = random.choice(["#333333", "#555555", "#666666", "#444444"])
        edge_width = random.uniform(1.5, 3.0)

        if self._is_directed:
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edge_color=edge_color,
                width=edge_width,
                arrows=True,
                arrowsize=20,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.1",
            )
        else:
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edge_color=edge_color,
                width=edge_width,
            )

        # Draw edge weights if weighted
        if self._is_weighted:
            edge_labels = nx.get_edge_attributes(G, "weight")
            nx.draw_networkx_edge_labels(
                G,
                pos,
                ax=ax,
                edge_labels=edge_labels,
                font_size=10,
                font_color="red",
                font_weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.8,
                ),
            )

        # Add title
        if random.random() < 0.7:
            graph_type = []
            if self._is_directed:
                graph_type.append("Directed")
            else:
                graph_type.append("Undirected")
            if self._is_weighted:
                graph_type.append("Weighted")
            title = " ".join(graph_type) + " Graph"
            ax.set_title(title, fontsize=14, fontweight="bold")

        ax.axis("off")
        plt.tight_layout()

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close(fig)

        # Optional noise
        if random.random() < 0.2:
            img = self._add_noise(img)

        return img

    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Add subtle noise or blur to the image."""
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        return img

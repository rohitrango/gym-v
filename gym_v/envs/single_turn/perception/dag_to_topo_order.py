"""DAG to Topological Order perception environment."""

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


class DAGToTopoOrderEnv(Env):
    # Meta: source=Perception, category=perception, turn=single
    """DAG to Topological Order perception environment.

    The agent must perceive a DAG (Directed Acyclic Graph) visualization
    and extract a valid topological ordering of the nodes.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (640, 480),
        min_nodes: int = 5,
        max_nodes: int = 9,
        edge_probability: float = 0.35,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.edge_probability = edge_probability
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._current_graph: nx.DiGraph | None = None
        self._current_topo_order: list[str] | None = None
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
            You are given a DAG (Directed Acyclic Graph) visualization.
            Your task is to provide a valid topological ordering of the nodes.

            A topological ordering is a linear ordering of nodes such that for every
            directed edge (u, v), node u comes before node v in the ordering.

            Example: {"topological_order": ["A", "B", "C", "D", "E"]}

            Note: There may be multiple valid topological orderings. Any valid one is accepted.

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
                {"topological_order": self._current_topo_order}
            ),
            "graph_edges": list(self._current_graph.edges()),
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
                {"topological_order": self._current_topo_order}
            ),
            "graph_edges": list(self._current_graph.edges()),
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
        """Compute reward by checking if the given order is a valid topological order."""
        try:
            parsed_action = json.loads(action)
            topo_order = parsed_action.get("topological_order", [])

            # Check if all nodes are present
            if set(topo_order) != set(self._current_graph.nodes()):
                return 0.0

            # Check topological order validity: for each edge (u, v), u must come before v
            pos = {node: i for i, node in enumerate(topo_order)}
            for u, v in self._current_graph.edges():
                if pos[u] >= pos[v]:
                    return 0.0

            return 1.0
        except Exception:
            return 0.0

    def render(self) -> Image.Image:
        return self._current_image

    def _generate_new_problem(self):
        """Generate a random DAG and compute topological order."""
        num_nodes = random.randint(self.min_nodes, self.max_nodes)
        labels = list(string.ascii_uppercase[:num_nodes])

        # Create DAG by only allowing edges from earlier to later nodes
        # This ensures no cycles
        G = nx.DiGraph()
        G.add_nodes_from(labels)

        # Shuffle labels to randomize the "layer" order
        shuffled_labels = labels.copy()
        random.shuffle(shuffled_labels)

        # Add edges only from earlier to later in shuffled order
        for i, u in enumerate(shuffled_labels):
            for j in range(i + 1, len(shuffled_labels)):
                v = shuffled_labels[j]
                if random.random() < self.edge_probability:
                    G.add_edge(u, v)

        # Ensure graph has some edges
        while G.number_of_edges() < num_nodes - 1:
            i = random.randint(0, len(shuffled_labels) - 2)
            j = random.randint(i + 1, len(shuffled_labels) - 1)
            u, v = shuffled_labels[i], shuffled_labels[j]
            if not G.has_edge(u, v):
                G.add_edge(u, v)

        self._current_graph = G
        self._current_topo_order = list(nx.topological_sort(G))
        self._current_image = self._render_dag(G)

    def _render_dag(self, G: nx.DiGraph) -> Image.Image:
        """Render the DAG as a PIL Image."""
        fig, ax = plt.subplots(
            figsize=(self.img_size[0] / 100, self.img_size[1] / 100), dpi=100
        )

        # Use layout that respects topological order somewhat
        layout_choice = random.choice(["spring", "kamada_kawai", "shell", "layered"])

        try:
            if layout_choice == "spring":
                pos = nx.spring_layout(G, seed=self._seed, k=2)
            elif layout_choice == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G)
            elif layout_choice == "shell":
                pos = nx.shell_layout(G)
            elif layout_choice == "layered":
                # Custom layered layout based on topological order
                topo_order = list(nx.topological_sort(G))
                pos = {}
                num_nodes = len(topo_order)
                for i, node in enumerate(topo_order):
                    # Add some randomness to x position
                    x = i / max(num_nodes - 1, 1) + random.uniform(-0.1, 0.1)
                    y = random.uniform(0.2, 0.8)
                    pos[node] = (x, y)
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

        # Draw directed edges with arrows
        edge_color = random.choice(["#333333", "#555555", "#666666"])
        edge_width = random.uniform(1.5, 2.5)

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color=edge_color,
            width=edge_width,
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            min_source_margin=20,
            min_target_margin=20,
        )

        # Add title
        if random.random() < 0.8:
            ax.set_title("Directed Acyclic Graph (DAG)", fontsize=14, fontweight="bold")

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

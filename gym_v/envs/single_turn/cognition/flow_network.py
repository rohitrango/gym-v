"""Flow Network perception environment."""

from __future__ import annotations

import io
import json
import logging
import string
from textwrap import dedent
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageFilter

from gym_v import Env, Observation, get_logger

logger = get_logger()

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


class FlowNetworkEnv(Env):
    # Meta: source=Perception, category=perception, turn=single
    """Flow Network perception environment.

    The agent must perceive a network flow graph visualization with
    capacities and compute the maximum flow from source to sink.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (640, 480),
        min_nodes: int = 5,
        max_nodes: int = 8,
        min_capacity: int = 1,
        max_capacity: int = 15,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._current_graph: nx.DiGraph | None = None
        self._current_source: str | None = None
        self._current_sink: str | None = None
        self._current_max_flow: int | None = None
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
            You are given a flow network visualization with edge capacities.
            Your task is to compute the maximum flow from source (S) to sink (T).

            The source node is marked as 'S' and the sink node is marked as 'T'.
            Edge labels show the capacity of each edge.

            Output the maximum flow value and the edges used.

            Example: {"max_flow": 23, "source": "S", "sink": "T"}

            Output ONLY the JSON string.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed

        self._generate_new_problem()

        obs = Observation(
            image=self._current_image,
            text=None,
            metadata={
                "num_nodes": self._current_graph.number_of_nodes(),
                "num_edges": self._current_graph.number_of_edges(),
                "source": self._current_source,
                "sink": self._current_sink,
            },
        )

        info = {
            "oracle_answer": json.dumps(
                {
                    "max_flow": self._current_max_flow,
                    "source": self._current_source,
                    "sink": self._current_sink,
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
                    "max_flow": self._current_max_flow,
                    "source": self._current_source,
                    "sink": self._current_sink,
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
        """Compute reward by checking if max flow value is correct."""
        try:
            parsed_action = json.loads(action)
            if parsed_action.get("max_flow") == self._current_max_flow:
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    def render(self) -> Image.Image:
        return self._current_image

    def _generate_new_problem(self):
        """Generate a random flow network and compute maximum flow."""
        num_nodes = self.py_random.randint(self.min_nodes, self.max_nodes)

        # Use S for source, T for sink, and letters for intermediate nodes
        intermediate_labels = list(string.ascii_uppercase[: num_nodes - 2])
        # Remove S and T from intermediate if present
        intermediate_labels = [
            lbl for lbl in intermediate_labels if lbl not in ["S", "T"]
        ]
        while len(intermediate_labels) < num_nodes - 2:
            for c in string.ascii_uppercase:
                if c not in intermediate_labels and c not in ["S", "T"]:
                    intermediate_labels.append(c)
                    if len(intermediate_labels) >= num_nodes - 2:
                        break

        self._current_source = "S"
        self._current_sink = "T"
        labels = (
            [self._current_source]
            + intermediate_labels[: num_nodes - 2]
            + [self._current_sink]
        )

        # Create directed graph
        G = nx.DiGraph()
        G.add_nodes_from(labels)

        # Create a layered structure for the flow network
        # Layer 0: source, Layer 1-k: intermediate, Layer k+1: sink
        num_layers = self.py_random.randint(2, min(4, num_nodes - 1))
        nodes_per_layer = [1]  # source
        remaining = num_nodes - 2
        for _ in range(num_layers - 2):
            if remaining > 0:
                count = self.py_random.randint(1, max(1, remaining // 2))
                nodes_per_layer.append(count)
                remaining -= count
        nodes_per_layer.append(max(1, remaining))  # last intermediate layer
        nodes_per_layer.append(1)  # sink

        # Assign nodes to layers
        layers = []
        idx = 0
        for count in nodes_per_layer:
            layer = labels[idx : idx + count]
            layers.append(layer)
            idx += count
            if idx >= len(labels):
                break

        # Ensure source and sink are in correct positions
        layers[0] = [self._current_source]
        if self._current_sink not in layers[-1]:
            layers[-1] = [self._current_sink]

        # Add edges between consecutive layers
        for i in range(len(layers) - 1):
            for u in layers[i]:
                for v in layers[i + 1]:
                    if self.py_random.random() < 0.6:
                        capacity = self.py_random.randint(self.min_capacity, self.max_capacity)
                        G.add_edge(u, v, capacity=capacity)

        # Ensure source has at least one outgoing edge
        if G.out_degree(self._current_source) == 0 and len(layers) > 1:
            v = self.py_random.choice(layers[1])
            G.add_edge(
                self._current_source,
                v,
                capacity=self.py_random.randint(self.min_capacity, self.max_capacity),
            )

        # Ensure sink has at least one incoming edge
        if G.in_degree(self._current_sink) == 0 and len(layers) > 1:
            u = self.py_random.choice(layers[-2])
            G.add_edge(
                u,
                self._current_sink,
                capacity=self.py_random.randint(self.min_capacity, self.max_capacity),
            )

        # Add some cross-layer edges for complexity
        for i in range(len(layers) - 2):
            for j in range(i + 2, len(layers)):
                for u in layers[i]:
                    for v in layers[j]:
                        if self.py_random.random() < 0.15:
                            capacity = self.py_random.randint(
                                self.min_capacity, self.max_capacity
                            )
                            if not G.has_edge(u, v):
                                G.add_edge(u, v, capacity=capacity)

        self._current_graph = G

        # Compute maximum flow
        try:
            flow_value, _ = nx.maximum_flow(
                G, self._current_source, self._current_sink, capacity="capacity"
            )
            self._current_max_flow = int(flow_value)
        except nx.NetworkXError:
            # No path exists, regenerate
            self._generate_new_problem()
            return

        self._current_image = self._render_flow_network(G)

    def _render_flow_network(self, G: nx.DiGraph) -> Image.Image:
        """Render the flow network as a PIL Image."""
        fig, ax = plt.subplots(
            figsize=(self.img_size[0] / 100, self.img_size[1] / 100), dpi=100
        )

        # Use a layered layout
        try:
            # Try to create a left-to-right layout
            pos = {}
            # BFS from source to assign layers
            layers = {self._current_source: 0}
            queue = [self._current_source]
            while queue:
                node = queue.pop(0)
                for neighbor in G.successors(node):
                    if neighbor not in layers:
                        layers[neighbor] = layers[node] + 1
                        queue.append(neighbor)

            # Assign positions
            max_layer = max(layers.values()) if layers else 0
            layer_nodes = {}
            for node, layer in layers.items():
                if layer not in layer_nodes:
                    layer_nodes[layer] = []
                layer_nodes[layer].append(node)

            for layer, nodes in layer_nodes.items():
                x = layer / max(max_layer, 1)
                for i, node in enumerate(nodes):
                    y = (i + 1) / (len(nodes) + 1)
                    pos[node] = (x, y)

            # Add any unassigned nodes
            for node in G.nodes():
                if node not in pos:
                    pos[node] = (self.py_random.random(), self.py_random.random())

        except Exception:
            pos = nx.spring_layout(G, seed=self._seed)

        # Node colors - special colors for source and sink
        node_colors = []
        for node in G.nodes():
            if node == self._current_source:
                node_colors.append("#32CD32")  # Green for source
            elif node == self._current_sink:
                node_colors.append("#FF4500")  # Red for sink
            else:
                node_colors.append("#87CEEB")  # Light blue for others

        # Draw nodes
        node_size = self.py_random.randint(1800, 2400)
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_size,
            edgecolors="black",
            linewidths=2,
        )

        # Draw node labels
        font_size = self.py_random.randint(14, 16)
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=font_size,
            font_weight="bold",
        )

        # Draw directed edges with arrows
        edge_width = self.py_random.uniform(1.5, 2.5)
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color="#555555",
            width=edge_width,
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            min_source_margin=25,
            min_target_margin=25,
        )

        # Draw edge capacities
        edge_labels = nx.get_edge_attributes(G, "capacity")
        nx.draw_networkx_edge_labels(
            G,
            pos,
            ax=ax,
            edge_labels=edge_labels,
            font_size=10,
            font_color="darkred",
            font_weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="lightyellow",
                edgecolor="none",
                alpha=0.9,
            ),
        )

        # Add title and legend
        ax.set_title("Flow Network (S=Source, T=Sink)", fontsize=14, fontweight="bold")

        ax.axis("off")
        plt.tight_layout()

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close(fig)

        # Optional noise
        if self.py_random.random() < 0.1:
            img = self._add_noise(img)

        return img

    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Add subtle noise or blur to the image."""
        if self.py_random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.py_random.uniform(0.3, 0.8)))
        return img

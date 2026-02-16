"""Tree to Traversal perception environment."""

from __future__ import annotations

import io
import json
import logging
import random
from textwrap import dedent
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

from gym_v import Env, Observation, get_logger

logger = get_logger()

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


class TreeNode:
    """Binary tree node."""

    def __init__(self, val: int):
        self.val = val
        self.left: TreeNode | None = None
        self.right: TreeNode | None = None


class TreeToTraversalEnv(Env):
    # Meta: source=Perception, category=perception, turn=single
    """Tree to Traversal perception environment.

    The agent must perceive a binary tree visualization and extract
    the tree traversal sequences (preorder, inorder, postorder).
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (640, 480),
        min_nodes: int = 5,
        max_nodes: int = 12,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._current_root: TreeNode | None = None
        self._current_traversals: dict[str, list[int]] | None = None
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
            You are given a binary tree visualization.
            Your task is to extract the tree traversal sequences as a JSON object.

            Output the preorder, inorder, and postorder traversals.

            Example: {"preorder": [1, 2, 4, 5, 3, 6], "inorder": [4, 2, 5, 1, 6, 3], "postorder": [4, 5, 2, 6, 3, 1]}

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
                "num_nodes": len(self._current_traversals["preorder"]),
            },
        )

        info = {
            "oracle_answer": json.dumps(self._current_traversals),
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
        info = {"oracle_answer": json.dumps(self._current_traversals)}

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
            if parsed_action == self._current_traversals:
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    def render(self) -> Image.Image:
        return self._current_image

    def _generate_new_problem(self):
        """Generate a random binary tree and render it."""
        num_nodes = random.randint(self.min_nodes, self.max_nodes)

        # Generate unique values
        values = random.sample(range(1, 100), num_nodes)

        # Build random binary tree
        self._current_root = self._build_random_tree(values)

        # Compute traversals
        self._current_traversals = {
            "preorder": self._preorder(self._current_root),
            "inorder": self._inorder(self._current_root),
            "postorder": self._postorder(self._current_root),
        }

        # Render
        self._current_image = self._render_tree(self._current_root)

    def _build_random_tree(self, values: list[int]) -> TreeNode:
        """Build a random binary tree from values."""
        if not values:
            return None

        # Use first value as root
        root = TreeNode(values[0])

        # Insert remaining values randomly
        for val in values[1:]:
            self._insert_random(root, val)

        return root

    def _insert_random(self, root: TreeNode, val: int):
        """Insert a value at a random position in the tree."""
        queue = [root]
        while queue:
            node = queue.pop(0)
            # Randomly choose to go left or right
            if random.random() < 0.5:
                if node.left is None:
                    node.left = TreeNode(val)
                    return
                else:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)
            else:
                if node.right is None:
                    node.right = TreeNode(val)
                    return
                else:
                    queue.append(node.right)
                if node.left is not None:
                    queue.append(node.left)

        # Fallback: BFS insert
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node.left is None:
                node.left = TreeNode(val)
                return
            queue.append(node.left)
            if node.right is None:
                node.right = TreeNode(val)
                return
            queue.append(node.right)

    def _preorder(self, root: TreeNode) -> list[int]:
        """Preorder traversal."""
        if root is None:
            return []
        return [root.val] + self._preorder(root.left) + self._preorder(root.right)

    def _inorder(self, root: TreeNode) -> list[int]:
        """Inorder traversal."""
        if root is None:
            return []
        return self._inorder(root.left) + [root.val] + self._inorder(root.right)

    def _postorder(self, root: TreeNode) -> list[int]:
        """Postorder traversal."""
        if root is None:
            return []
        return self._postorder(root.left) + self._postorder(root.right) + [root.val]

    def _render_tree(self, root: TreeNode) -> Image.Image:
        """Render the tree as a PIL Image."""
        fig, ax = plt.subplots(
            figsize=(self.img_size[0] / 100, self.img_size[1] / 100), dpi=100
        )

        # Compute positions using a simple layout
        positions = {}
        self._compute_positions(root, positions, x=0.5, y=0.9, dx=0.25, level=0)

        # Get all nodes
        nodes = []
        self._collect_nodes(root, nodes)

        # Draw edges first
        self._draw_edges(ax, root, positions)

        # Draw nodes
        node_radius = 0.04
        for node in nodes:
            x, y = positions[id(node)]
            color = random.choice(self._node_colors)

            circle = mpatches.Circle(
                (x, y),
                node_radius,
                facecolor=color,
                edgecolor="black",
                linewidth=2,
                zorder=2,
            )
            ax.add_patch(circle)

            # Draw value
            ax.text(
                x,
                y,
                str(node.val),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                zorder=3,
            )

        # Add title
        if random.random() < 0.8:
            ax.set_title("Binary Tree", fontsize=14, fontweight="bold")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
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

    def _compute_positions(
        self, node: TreeNode, positions: dict, x: float, y: float, dx: float, level: int
    ):
        """Compute positions for tree nodes."""
        if node is None:
            return

        positions[id(node)] = (x, y)
        dy = 0.12

        if node.left:
            self._compute_positions(
                node.left,
                positions,
                x=x - dx / (1.5**level),
                y=y - dy,
                dx=dx,
                level=level + 1,
            )

        if node.right:
            self._compute_positions(
                node.right,
                positions,
                x=x + dx / (1.5**level),
                y=y - dy,
                dx=dx,
                level=level + 1,
            )

    def _collect_nodes(self, node: TreeNode, nodes: list):
        """Collect all nodes in the tree."""
        if node is None:
            return
        nodes.append(node)
        self._collect_nodes(node.left, nodes)
        self._collect_nodes(node.right, nodes)

    def _draw_edges(self, ax, node: TreeNode, positions: dict):
        """Draw edges between nodes."""
        if node is None:
            return

        x, y = positions[id(node)]

        if node.left:
            lx, ly = positions[id(node.left)]
            ax.plot([x, lx], [y, ly], color="#555555", linewidth=2, zorder=1)
            self._draw_edges(ax, node.left, positions)

        if node.right:
            rx, ry = positions[id(node.right)]
            ax.plot([x, rx], [y, ry], color="#555555", linewidth=2, zorder=1)
            self._draw_edges(ax, node.right, positions)

    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Add subtle noise or blur to the image."""
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        return img

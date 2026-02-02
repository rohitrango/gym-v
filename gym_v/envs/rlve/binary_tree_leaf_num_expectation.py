"""Binary Tree Leaf Number Expectation environment for gym-v (self-contained)."""

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


class RLVEBinaryTreeLeafNumExpectationEnv(Env):
    """RLVE Binary Tree Leaf Number Expectation as a single-turn environment.

    Problem: Given a uniformly random binary tree with N nodes, what is the
    expected number of leaf nodes? The answer should be in the form A/B where
    A and B are coprime positive integers.

    The answer formula is: N*(N+1) / (2*(2*N-1)) in simplified form.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = (
        r"""We uniformly at random generate a **binary tree** with exactly {N} nodes """
        r"""(all distinct binary trees with {N} nodes are equally likely). Two binary """
        r"""trees are considered identical if and only if:
- both are empty, **OR**
- both are non-empty, and their left subtrees are identical and their right subtrees are identical.

What is the expected number of **leaf** nodes (nodes whose left and right children are """
        r"""both empty) in the generated binary tree? Output the result as `A/B` (do NOT """
        r"""include quotes), where A and B are positive integers separated by a slash `/`."""
    )

    def __init__(
        self,
        max_n: int = 15,
        image_width: int = 800,
        image_height: int = 600,
        node_radius: int = 20,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}
        self._max_n = max_n
        self._image_width = image_width
        self._image_height = image_height
        self._node_radius = node_radius

        self._N: int | None = None
        self._oracle_answer: str | None = None
        self._gold_answer: tuple[int, int] | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        """Return description adapted for visual input."""
        return dedent(
            """\
            Binary Tree Leaf Number Expectation:

            Given a uniformly random binary tree with N nodes (all distinct binary
            trees with N nodes are equally likely), compute the expected number of
            leaf nodes. A leaf node is a node whose left and right children are both
            empty.

            Two binary trees are considered identical if and only if:
            - both are empty, OR
            - both are non-empty, and their left subtrees are identical and their
              right subtrees are identical.

            In the image:
            - A sample binary tree with N nodes is shown
            - Nodes are arranged in a hierarchical layout (levels from top to bottom)
            - Leaf nodes are highlighted in green
            - Internal nodes are shown in light blue
            - Edges connect parent nodes to their children
            - The tree structure illustrates what a binary tree looks like

            Output format: A/B (do NOT include quotes), where A and B are positive
            integers separated by a slash `/`."""
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the problem input."""
        if self._N is None:
            return ""
        return f"N = {self._N}"

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
        info = {"oracle_answer": self._oracle_answer}

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
        """Generate problem instance with random N."""
        N = int(self.np_random.integers(1, self._max_n + 1))
        self._N = N

        # Compute expected number of leaf nodes
        # Formula: N*(N+1) / (2*(2*N-1))
        A = N * (N + 1)
        B = 2 * (2 * N - 1)
        gcd_AB = math.gcd(A, B)
        A //= gcd_AB
        B //= gcd_AB

        self._gold_answer = (A, B)
        self._oracle_answer = f"{A}/{B}"

    def _prompt_generate(self) -> str:
        """Generate text prompt."""
        return self.prompt_template.format(N=self._N)

    def _process(self, answer: str | None) -> tuple[int, int] | None:
        """Process answer string into tuple of (A, B)."""
        if answer is not None:
            answer = answer.strip()
            try:
                parts = answer.split("/")
                if len(parts) != 2:
                    return None
                A, B = map(int, map(str.strip, parts))
                return (A, B)
            except (ValueError, AttributeError):
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score answer - must match gold answer in simplified form."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        A, B = processed_result
        if not (A > 0 and B > 0):
            return 0.0
        gold_A, gold_B = self._gold_answer
        # Simplify the user's answer
        gcd_AB = math.gcd(A, B)
        A //= gcd_AB
        B //= gcd_AB

        if (A, B) == (gold_A, gold_B):
            return 1.0
        else:
            return 0.0

    def render(self) -> Image.Image:
        """Render a sample binary tree with hierarchical layout."""
        if self._N is None:
            return Image.new(
                "RGB", (self._image_width, self._image_height), (255, 255, 255)
            )

        N = self._N

        # Create image with light background
        img = Image.new("RGB", (self._image_width, self._image_height), (245, 248, 250))
        draw = ImageDraw.Draw(img)

        # Load fonts
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 18)
            title_font = ImageFont.truetype(str(font_path), 24)
            label_font = ImageFont.truetype(str(font_path), 16)
        else:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        # Add title
        title = f"Sample Binary Tree with N={N} nodes"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_width // 2 - tw // 2, 20),
            title,
            fill=(40, 60, 80),
            font=title_font,
        )

        # Generate a random binary tree structure for visualization
        # Build a binary tree using random insertion
        tree_structure = self._generate_sample_binary_tree(N)

        # Compute positions using level-order layout
        positions, is_leaf = self._compute_tree_positions(tree_structure)

        # Draw edges first (so they appear behind nodes)
        for node_id, children in tree_structure.items():
            if node_id not in positions:
                continue
            x1, y1 = positions[node_id]
            left_child, right_child = children

            if left_child is not None and left_child in positions:
                x2, y2 = positions[left_child]
                draw.line([(x1, y1), (x2, y2)], fill=(120, 140, 160), width=2)

            if right_child is not None and right_child in positions:
                x2, y2 = positions[right_child]
                draw.line([(x1, y1), (x2, y2)], fill=(120, 140, 160), width=2)

        # Draw nodes
        for node_id in range(N):
            if node_id not in positions:
                continue

            x, y = positions[node_id]

            # Determine node color based on whether it's a leaf
            if is_leaf[node_id]:
                # Leaf nodes in green
                node_fill = (144, 238, 144)  # Light green
                node_outline = (34, 139, 34)  # Forest green
            else:
                # Internal nodes in light blue
                node_fill = (135, 206, 250)
                node_outline = (70, 130, 180)

            # Draw node circle
            draw.ellipse(
                [
                    x - self._node_radius,
                    y - self._node_radius,
                    x + self._node_radius,
                    y + self._node_radius,
                ],
                fill=node_fill,
                outline=node_outline,
                width=2,
            )

            # Draw node label
            text = str(node_id)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x - tw // 2, y - th // 2),
                text,
                fill=(0, 0, 0),
                font=font,
            )

        # Add legend
        legend_y = self._image_height - 80
        legend_x = 60

        # Leaf node example
        draw.ellipse(
            [legend_x - 15, legend_y - 15, legend_x + 15, legend_y + 15],
            fill=(144, 238, 144),
            outline=(34, 139, 34),
            width=2,
        )
        draw.text(
            (legend_x + 25, legend_y - 10),
            "Leaf Node",
            fill=(40, 60, 80),
            font=label_font,
        )

        # Internal node example
        legend_x += 180
        draw.ellipse(
            [legend_x - 15, legend_y - 15, legend_x + 15, legend_y + 15],
            fill=(135, 206, 250),
            outline=(70, 130, 180),
            width=2,
        )
        draw.text(
            (legend_x + 25, legend_y - 10),
            "Internal Node",
            fill=(40, 60, 80),
            font=label_font,
        )

        # Add note at bottom
        note = "Note: This is one random sample. The problem asks for the expected leaf count across all possible trees."
        bbox = draw.textbbox((0, 0), note, font=label_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_width // 2 - tw // 2, self._image_height - 35),
            note,
            fill=(100, 120, 140),
            font=label_font,
        )

        return img

    def _generate_sample_binary_tree(
        self, n: int
    ) -> dict[int, tuple[int | None, int | None]]:
        """Generate a random binary tree structure for visualization.

        Returns a dictionary mapping node_id -> (left_child, right_child).
        Uses random insertion to create a binary tree.
        """
        if n == 0:
            return {}

        # Tree structure: node_id -> (left_child, right_child)
        tree = {0: (None, None)}

        for i in range(1, n):
            # Randomly choose a parent from existing nodes
            nodes = list(tree.keys())
            parent = int(self.np_random.choice(nodes))

            # Try to insert as left child first, then right child
            left_child, right_child = tree[parent]
            if left_child is None:
                tree[parent] = (i, right_child)
            elif right_child is None:
                tree[parent] = (left_child, i)
            else:
                # Parent already has two children, pick another parent
                # Find a node with at least one empty child
                candidates = [
                    node
                    for node in nodes
                    if tree[node][0] is None or tree[node][1] is None
                ]
                if candidates:
                    parent = int(self.np_random.choice(candidates))
                    left_child, right_child = tree[parent]
                    if left_child is None:
                        tree[parent] = (i, right_child)
                    else:
                        tree[parent] = (left_child, i)

            # Initialize new node
            tree[i] = (None, None)

        return tree

    def _compute_tree_positions(
        self, tree: dict[int, tuple[int | None, int | None]]
    ) -> tuple[dict[int, tuple[float, float]], dict[int, bool]]:
        """Compute positions for tree nodes using hierarchical layout.

        Returns:
            positions: dict mapping node_id -> (x, y) coordinates
            is_leaf: dict mapping node_id -> whether it's a leaf
        """
        if not tree:
            return {}, {}

        # Find levels using BFS from root (node 0)
        levels = {0: 0}
        queue = deque([0])
        max_level = 0

        while queue:
            node = queue.popleft()
            level = levels[node]
            max_level = max(max_level, level)

            left_child, right_child = tree[node]
            if left_child is not None:
                levels[left_child] = level + 1
                queue.append(left_child)
            if right_child is not None:
                levels[right_child] = level + 1
                queue.append(right_child)

        # Group nodes by level
        nodes_by_level = {}
        for node, level in levels.items():
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(node)

        # Compute positions
        positions = {}
        margin_top = 80
        margin_bottom = 120
        margin_x = 60
        usable_height = self._image_height - margin_top - margin_bottom
        usable_width = self._image_width - 2 * margin_x

        for level in range(max_level + 1):
            nodes_at_level = nodes_by_level.get(level, [])
            y = margin_top + (level * usable_height / max(1, max_level))

            # Distribute nodes evenly across width
            num_nodes = len(nodes_at_level)
            for i, node in enumerate(nodes_at_level):
                if num_nodes == 1:
                    x = self._image_width / 2
                else:
                    x = margin_x + (i * usable_width / (num_nodes - 1))
                positions[node] = (x, y)

        # Determine which nodes are leaves
        is_leaf = {}
        for node, (left, right) in tree.items():
            is_leaf[node] = left is None and right is None

        return positions, is_leaf

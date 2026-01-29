"""Weighted Binary Tree environment for gym-v (self-contained)."""

from __future__ import annotations

from collections import deque
from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEWeightedBinarytreeEnv(Env):
    """RLVE Weighted Binary Tree as a single-turn environment.

    Problem: Given N nodes with scores d_i (where i = 0 to N-1), construct a binary
    tree where the in-order traversal is fixed as 0, 1, 2, ..., N-1. The score of
    a tree is computed recursively as: score(tree) = score(left) * score(right) + d_root,
    where empty subtrees have score 1. Find the tree structure that maximizes the
    score and output its pre-order traversal.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a binary tree with {N} nodes, labeled from 0 to {N_minus_1}.
The **in-order traversal** of the tree is: `0, 1, ..., {N_minus_1}` — that is, the in-order sequence is fixed in increasing order of node labels.

Each node `i` has an associated score `d_i` (where `0 ≤ i < {N}`), given as:
{scores}

The **score of a binary tree** is defined recursively as follows:
- `score(tree) = score(left_subtree) × score(right_subtree) + d_i`, where `i` is the root of the current subtree.
- If a subtree is **empty**, its score is defined to be `1`.
- If a node is a **leaf**, its score is simply `d_i` (ignore its empty subtrees).

Your task is to construct the binary tree that satisfies the above rules and has the **maximum possible score**, and then give its **pre-order traversal**.

Output Format:
Your final answer should be a single line containing the node labels in **pre-order traversal**, separated by **spaces**.
Example: `{all_node_sequence}` (do **NOT** include the backticks or quotes)."""

    def __init__(
        self,
        max_n: int = 8,
        max_score: int = 10,
        image_width: int = 800,
        image_height: int = 600,
        node_radius: int = 25,
        num_players: int = 1,
        rewarding_strategy: str = "(answer/gold)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}
        self._max_n = max_n
        self._max_score = max_score
        self._image_width = image_width
        self._image_height = image_height
        self._node_radius = node_radius

        self.rewards = {
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        self._N: int | None = None
        self._scores: list[int] | None = None
        self._gold: int | None = None
        self._oracle_answer: str | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        """Return description adapted for visual input."""
        return dedent(
            """
            Weighted Binary Tree:

            You are given N nodes (labeled 0 to N-1) where each node i has a weight/score d_i.
            You must construct a binary tree such that:
            1. The in-order traversal is fixed: 0, 1, 2, ..., N-1
            2. The tree maximizes a specific score function

            The score is computed recursively:
            - score(tree) = score(left_subtree) × score(right_subtree) + d_root
            - Empty subtrees have score = 1
            - Leaf nodes have score = d_i

            In the image:
            - Nodes are displayed with their labels and weights (e.g., "node: weight")
            - The tree structure shows a sample valid binary tree
            - Nodes are arranged in a hierarchical layout from top (root) to bottom (leaves)
            - Edges connect parent nodes to their children
            - Node weights are the key values for computing the optimal tree score

            Your task: Find the binary tree structure that maximizes the score, then
            output its pre-order traversal (visit root, then left subtree, then right subtree).

            Output format: Space-separated node labels in pre-order (e.g., "3 1 0 2 5 4 6")
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
            text=state_text,
            metadata={"text_prompt": self._prompt},
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
            text=state_text,
            metadata={"text_prompt": self._prompt},
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
        """Generate problem instance - ported from RLVE."""
        N = int(self.np_random.integers(3, self._max_n + 1))
        self._N = N

        # Generate random node scores
        self._scores = [
            int(self.np_random.integers(1, self._max_score + 1)) for _ in range(N)
        ]

        # Dynamic programming to find optimal tree structure
        # dpF[i][j] = maximum score for subtree with in-order traversal [i..j]
        # roots[i][j] = optimal root for subtree with in-order traversal [i..j]
        dpF = [[0] * N for _ in range(N)]
        roots = [[None] * N for _ in range(N)]

        # Base case: single node
        for i, score in enumerate(self._scores):
            dpF[i][i] = score
            roots[i][i] = i

        # Fill DP table for increasing subtree lengths
        for length in range(2, N + 1):
            for i in range(N - length + 1):
                j = i + length - 1
                for root in range(i, j + 1):
                    # Score for left and right subtrees
                    left = dpF[i][root - 1] if i <= root - 1 else 1
                    right = dpF[root + 1][j] if root + 1 <= j else 1
                    score = left * right + self._scores[root]

                    if dpF[i][j] <= score:
                        dpF[i][j] = score
                        roots[i][j] = root

        self._gold = dpF[0][N - 1]

        # Build pre-order traversal of optimal tree
        def preorder(i: int, j: int) -> list[int]:
            if i > j:
                return []
            root = roots[i][j]
            return [root] + preorder(i, root - 1) + preorder(root + 1, j)

        self._oracle_answer = " ".join(map(str, preorder(0, N - 1)))

    def _prompt_generate(self) -> str:
        """Generate text prompt."""
        scores_str = "\n".join(f"d_{i}={score}" for i, score in enumerate(self._scores))
        return self.prompt_template.format(
            N=self._N,
            N_minus_1=self._N - 1,
            scores=scores_str,
            all_node_sequence=" ".join(map(str, range(self._N))),
        )

    def _process(self, answer: str | None) -> list[int] | None:
        """Process answer string into list of node IDs."""
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
        """Score answer - ported from RLVE."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0

        # Check if it's a valid permutation
        if len(processed_result) != self._N:
            return 0.0
        if len(set(processed_result)) != self._N:
            return 0.0
        for i in processed_result:
            if not (0 <= i < self._N):
                return 0.0

        # Compute the score of the tree represented by this pre-order traversal
        def get_score(
            inorder_l: int, inorder_r: int, preorder: list[int]
        ) -> int | None:
            """Compute score for tree with given in-order and pre-order traversals."""
            if len(preorder) != inorder_r - inorder_l + 1:
                return None

            root = preorder[0]
            if not (inorder_l <= root <= inorder_r):
                return None

            if inorder_l == inorder_r:
                return self._scores[root]

            # Split preorder into left and right subtrees
            left_size = root - inorder_l
            left_preorder = preorder[1 : 1 + left_size]
            right_preorder = preorder[1 + left_size :]

            left_score = (
                get_score(inorder_l, root - 1, left_preorder)
                if inorder_l <= root - 1
                else 1
            )
            right_score = (
                get_score(root + 1, inorder_r, right_preorder)
                if root + 1 <= inorder_r
                else 1
            )

            if left_score is not None and right_score is not None:
                return left_score * right_score + self._scores[root]
            else:
                return None

        answer_score = get_score(0, self._N - 1, processed_result)
        if answer_score is None:
            return 0.0

        # Ensure answer doesn't exceed gold (shouldn't happen, but safety check)
        assert (
            answer_score <= self._gold
        ), f"Answer score {answer_score} exceeds gold {self._gold}"

        # Compute reward based on strategy
        if self.rewards["rewarding_strategy"] == "(answer/gold)^beta":
            return self.rewards["rewarding_weight"] * (
                (answer_score / self._gold) ** self.rewards["rewarding_beta"]
            )
        elif self.rewards["rewarding_strategy"] == "gold=answer":
            return self.rewards["rewarding_weight"] * (answer_score == self._gold)
        else:
            raise NotImplementedError(
                f"Unknown rewarding strategy: {self.rewards['rewarding_strategy']}"
            )

    def render(self) -> Image.Image:
        """Render the weighted binary tree with hierarchical layout."""
        if self._N is None or self._scores is None:
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
            font = ImageFont.truetype(str(font_path), 16)
            title_font = ImageFont.truetype(str(font_path), 24)
            label_font = ImageFont.truetype(str(font_path), 14)
            weight_font = ImageFont.truetype(str(font_path), 12)
        else:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            weight_font = ImageFont.load_default()

        # Add title
        title = f"Binary Tree with N={N} nodes (with weights)"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_width // 2 - tw // 2, 20),
            title,
            fill=(40, 60, 80),
            font=title_font,
        )

        # Generate a sample binary tree structure for visualization
        tree_structure = self._generate_sample_binary_tree(N)

        # Compute positions using hierarchical layout
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

        # Draw nodes with weights
        for node_id in range(N):
            if node_id not in positions:
                continue

            x, y = positions[node_id]

            # All nodes in light blue (weights are the important feature)
            node_fill = (173, 216, 230)  # Light blue
            node_outline = (70, 130, 180)  # Steel blue

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

            # Draw node label (just the node number)
            text = str(node_id)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x - tw // 2, y - th // 2 - 3),
                text,
                fill=(0, 0, 0),
                font=font,
            )

            # Draw weight below node label
            weight_text = f"w:{self._scores[node_id]}"
            bbox = draw.textbbox((0, 0), weight_text, font=weight_font)
            ww = bbox[2] - bbox[0]
            wh = bbox[3] - bbox[1]
            draw.text(
                (x - ww // 2, y + th // 2 - 3),
                weight_text,
                fill=(0, 0, 128),  # Dark blue for weights
                font=weight_font,
            )

        # Add weight list at bottom
        legend_y = self._image_height - 80
        weights_text = "Node weights: " + ", ".join(
            f"{i}→{self._scores[i]}" for i in range(N)
        )
        bbox = draw.textbbox((0, 0), weights_text, font=label_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_width // 2 - tw // 2, legend_y),
            weights_text,
            fill=(40, 60, 80),
            font=label_font,
        )

        # Add note at bottom
        note = "Note: This is one sample tree structure. Find the structure that maximizes the tree score."
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
        """
        if n == 0:
            return {}

        # Tree structure: node_id -> (left_child, right_child)
        tree = {0: (None, None)}

        for i in range(1, n):
            # Find a node with at least one empty child
            candidates = [
                node
                for node in tree.keys()
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

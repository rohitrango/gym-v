"""Preorder Traversal environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEPreorderTraversalEnv(Env):
    """RLVE Preorder Traversal as a single-turn environment.

    Given a binary tree's in-order and post-order traversal sequences,
    reconstruct the tree and output its pre-order traversal sequence.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a binary tree with nodes labeled from 0 to {N_minus_1}.

Its **in-order traversal** sequence is: {inorder_traversal}
Its **post-order traversal** sequence is: {postorder_traversal}

Your task is to reconstruct the tree and output its **pre-order traversal** sequence.

Output Format:
Your final answer should be a single line containing the pre-order traversal, with node labels separated by **spaces**.
Example: `{all_node_sequence}` (do **NOT** include the backticks or quotes).
"""

    def __init__(
        self,
        max_n: int = 7,
        cell_px: int = 80,
        padding: int = 40,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._tree: dict[str, Any] | None = None
        self._inorder: list[int] | None = None
        self._postorder: list[int] | None = None
        self._preorder: list[int] | None = None
        self._prompt: str | None = None
        self._reference_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n:
            size_hint = f"{self._n} nodes"
        else:
            size_hint = "N nodes"

        return dedent(
            f"""
            Preorder Traversal Problem:

            Given a binary tree with {size_hint} labeled from 0 to N-1, you are provided with:
            - Its in-order traversal sequence (Left → Root → Right)
            - Its post-order traversal sequence (Left → Right → Root)

            Task: Reconstruct the binary tree and output its pre-order traversal sequence (Root → Left → Right).

            In the visualization:
            - The binary tree structure is shown with nodes arranged in levels
            - Each node is labeled with its number
            - Solid lines connect parent nodes to their children
            - The pre-order traversal path is highlighted in red showing the visit order
            - Nodes are numbered to show the sequence of visits in preorder
            - A legend explains the traversal orders:
              * Pre-order: Root → Left → Right (what you need to find)
              * In-order: Left → Root → Right (given)
              * Post-order: Left → Right → Root (given)

            Output format: A single line with node labels separated by spaces (e.g., "0 1 2 3 4").
            """
        ).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._generate()
        self._prompt = self._prompt_generate()
        self._last_image = self.render()

        obs = Observation(
            image=self._last_image,
            text=self._prompt,
            metadata={
                "rlve_prompt": self._prompt,
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
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
        obs = Observation(
            image=self._last_image,
            text=None,
            metadata={
                "rlve_prompt": self._prompt,
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
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
        """Generate a binary tree and compute its traversals.

        Ports generation logic from RLVE using self.np_random.
        """
        N = int(self.np_random.integers(3, self._max_n + 1))
        self._n = N

        nodes = list(range(N))
        self.np_random.shuffle(nodes)

        def build(nodes: list[int]) -> dict[str, Any] | None:
            if not nodes:
                return None
            root_index = int(self.np_random.integers(0, len(nodes)))
            return {
                "root": nodes[root_index],
                "left": build(nodes[:root_index]),
                "right": build(nodes[root_index + 1 :]),
            }

        tree = build(nodes)
        self._tree = tree

        def preorder_traversal(node: dict[str, Any] | None) -> list[int]:
            if node is None:
                return []
            return (
                [node["root"]]
                + preorder_traversal(node["left"])
                + preorder_traversal(node["right"])
            )

        def inorder_traversal(node: dict[str, Any] | None) -> list[int]:
            if node is None:
                return []
            return (
                inorder_traversal(node["left"])
                + [node["root"]]
                + inorder_traversal(node["right"])
            )

        def postorder_traversal(node: dict[str, Any] | None) -> list[int]:
            if node is None:
                return []
            return (
                postorder_traversal(node["left"])
                + postorder_traversal(node["right"])
                + [node["root"]]
            )

        self._inorder = inorder_traversal(tree)
        self._postorder = postorder_traversal(tree)
        self._preorder = preorder_traversal(tree)
        self._reference_answer = " ".join(map(str, self._preorder))

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N_minus_1=self._n - 1,
            inorder_traversal=" ".join(map(str, self._inorder)),
            postorder_traversal=" ".join(map(str, self._postorder)),
            all_node_sequence=" ".join(map(str, range(self._n))),
        )

    def _process(self, answer: str | None) -> list[int] | None:
        """Process the answer string into a list of integers."""
        if answer is not None:
            answer = answer.strip()
            if not answer:
                return None
            try:
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError:
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on correctness.

        Returns:
            -1.0: wrong format (not parseable as integers)
             0.0: wrong length
            reward: (correct_count / N) ** beta for partial credit
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if len(processed_result) != self._n:
                return 0.0

            # Use mean([gold=answer])^beta strategy
            beta = 10.0
            correct_count = sum(
                float(a == b) for a, b in zip(self._preorder, processed_result)
            )
            return (correct_count / self._n) ** beta
        else:
            return -1.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the binary tree with preorder traversal visualization.

        Shows:
        - Binary tree structure with nodes and edges
        - Nodes labeled with their values
        - Preorder traversal path highlighted in red
        - Visit order numbers on nodes
        - Legend explaining the traversal types
        """
        if self._tree is None:
            raise RuntimeError("No tree generated")

        cell_px = self._cell_px
        padding = self._padding

        # Calculate tree dimensions (level by level)
        def get_tree_depth(node: dict[str, Any] | None) -> int:
            if node is None:
                return 0
            return 1 + max(get_tree_depth(node["left"]), get_tree_depth(node["right"]))

        def count_nodes(node: dict[str, Any] | None) -> int:
            if node is None:
                return 0
            return 1 + count_nodes(node["left"]) + count_nodes(node["right"])

        depth = get_tree_depth(self._tree)
        node_count = count_nodes(self._tree)

        # Calculate positions for all nodes
        positions: dict[int, tuple[float, float]] = {}

        def calculate_positions(
            node: dict[str, Any] | None,
            x: float,
            y: float,
            h_spacing: float,
            level: int,
        ) -> None:
            if node is None:
                return
            positions[node["root"]] = (x, y)

            next_y = y + cell_px * 1.5
            next_spacing = h_spacing / 2

            if node["left"]:
                calculate_positions(
                    node["left"], x - h_spacing, next_y, next_spacing, level + 1
                )
            if node["right"]:
                calculate_positions(
                    node["right"], x + h_spacing, next_y, next_spacing, level + 1
                )

        # Initial spacing based on depth
        initial_spacing = cell_px * (2 ** (depth - 1))
        center_x = initial_spacing + padding
        start_y = padding + cell_px

        calculate_positions(self._tree, center_x, start_y, initial_spacing / 2, 0)

        # Calculate canvas size
        if positions:
            min_x = min(x for x, y in positions.values())
            max_x = max(x for x, y in positions.values())
            max_y = max(y for x, y in positions.values())
        else:
            min_x = max_x = max_y = 0

        legend_height = 180
        width = int(max_x - min_x + padding * 3 + cell_px)
        height = int(max_y + padding * 2 + cell_px + legend_height)

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load fonts
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_large = ImageFont.truetype(font_path, int(cell_px * 0.35))
            font_medium = ImageFont.truetype(font_path, 16)
            font_small = ImageFont.truetype(font_path, 14)
            font_tiny = ImageFont.truetype(font_path, 12)
        else:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_tiny = ImageFont.load_default()

        # Adjust positions to fit canvas
        x_offset = padding - min_x + cell_px // 2

        adjusted_positions = {
            node: (x + x_offset, y) for node, (x, y) in positions.items()
        }

        # Draw edges first
        def draw_edges(node: dict[str, Any] | None) -> None:
            if node is None:
                return
            parent_pos = adjusted_positions[node["root"]]
            if node["left"]:
                child_pos = adjusted_positions[node["left"]["root"]]
                draw.line(
                    [parent_pos, child_pos], fill=(80, 80, 80), width=3
                )
                draw_edges(node["left"])
            if node["right"]:
                child_pos = adjusted_positions[node["right"]["root"]]
                draw.line(
                    [parent_pos, child_pos], fill=(80, 80, 80), width=3
                )
                draw_edges(node["right"])

        draw_edges(self._tree)

        # Draw preorder traversal path
        if len(self._preorder) > 1:
            path_points = [adjusted_positions[node] for node in self._preorder]
            for i in range(len(path_points) - 1):
                x1, y1 = path_points[i]
                x2, y2 = path_points[i + 1]
                # Draw arrow
                draw.line([(x1, y1), (x2, y2)], fill=(220, 20, 20), width=4)

                # Draw arrowhead
                import math

                angle = math.atan2(y2 - y1, x2 - x1)
                arrow_len = 10
                arrow_angle = math.pi / 6

                end_x = x2 - arrow_len * math.cos(angle)
                end_y = y2 - arrow_len * math.sin(angle)

                left_x = end_x - arrow_len * 0.5 * math.cos(angle - arrow_angle)
                left_y = end_y - arrow_len * 0.5 * math.sin(angle - arrow_angle)
                right_x = end_x - arrow_len * 0.5 * math.cos(angle + arrow_angle)
                right_y = end_y - arrow_len * 0.5 * math.sin(angle + arrow_angle)

                draw.polygon(
                    [(x2, y2), (left_x, left_y), (right_x, right_y)],
                    fill=(220, 20, 20),
                )

        # Draw nodes
        for node_val, (x, y) in adjusted_positions.items():
            # Determine if this node is in the preorder path
            node_color = (100, 150, 255)  # Blue
            border_color = (40, 40, 40)

            # Draw node circle
            radius = cell_px // 2 - 5
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=node_color,
                outline=border_color,
                width=3,
            )

            # Draw node label
            label = str(node_val)
            bbox = draw.textbbox((0, 0), label, font=font_large)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (x - tw // 2, y - th // 2), label, fill=(255, 255, 255), font=font_large
            )

            # Draw visit order number (small badge)
            if node_val in self._preorder:
                visit_order = self._preorder.index(node_val) + 1
                order_str = str(visit_order)
                badge_x = x + radius - 10
                badge_y = y - radius + 10

                # Draw small circle badge
                badge_radius = 12
                draw.ellipse(
                    [
                        badge_x - badge_radius,
                        badge_y - badge_radius,
                        badge_x + badge_radius,
                        badge_y + badge_radius,
                    ],
                    fill=(220, 20, 20),
                    outline=(150, 0, 0),
                    width=2,
                )

                # Draw order number
                bbox = draw.textbbox((0, 0), order_str, font=font_tiny)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    (badge_x - tw // 2, badge_y - th // 2),
                    order_str,
                    fill=(255, 255, 255),
                    font=font_tiny,
                )

        # Draw legend
        legend_y = int(max_y + padding * 2 + cell_px)

        # Title
        title = "Binary Tree Traversal Orders:"
        draw.text((padding, legend_y), title, fill=(30, 30, 30), font=font_medium)
        legend_y += 30

        # Traversal types
        traversal_info = [
            (
                "Pre-order (Root → Left → Right):",
                " ".join(map(str, self._preorder)),
                (220, 20, 20),
            ),
            (
                "In-order (Left → Root → Right):",
                " ".join(map(str, self._inorder)),
                (60, 60, 60),
            ),
            (
                "Post-order (Left → Right → Root):",
                " ".join(map(str, self._postorder)),
                (60, 60, 60),
            ),
        ]

        for label, sequence, color in traversal_info:
            draw.text((padding + 10, legend_y), label, fill=color, font=font_small)
            legend_y += 20
            draw.text(
                (padding + 30, legend_y), sequence, fill=(100, 100, 100), font=font_small
            )
            legend_y += 28

        # Add note about the problem
        note = "Red path shows the pre-order traversal sequence (what you need to find)."
        draw.text((padding, legend_y), note, fill=(150, 150, 150), font=font_tiny)

        return img

"""FBI Binary Tree environment for gym-v (self-contained)."""

from __future__ import annotations

from collections import deque
from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEFbiBinaryTreeEnv(Env):
    """RLVE FBI Binary Tree as a single-turn environment.

    Problem: Given a binary string of length 2^N, construct an FBI tree where:
    - B-string: only contains '0's
    - I-string: only contains '1's
    - F-string: contains both '0' and '1'

    The tree is built recursively by splitting the string in half. The task is
    to output the postorder traversal of the FBI tree (left, right, root).
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""We classify binary strings made up of only `0` and `1` into three types:
- A string consisting of only `0`s is called a **B-string**.
- A string consisting of only `1`s is called an **I-string**.
- A string that contains both `0` and `1` is called an **F-string**.

An **FBI tree** is a binary tree where each node is labeled as either F, B, or I, based on the type of the substring it represents.
Given a binary string `S`, construct an FBI tree `T` using the following recursive rules:
1. The **root node** corresponds to the entire string `S`, and its type is determined using the rules above.
2. If the length of `S` is greater than 1, divide `S` exactly in half into two equal substrings: S₁ (left) and S₂ (right). Recursively build the **left subtree** from S₁, and the **right subtree** from S₂.

Your task is to construct the FBI tree from the following binary string of length 2^{N}:
{string}

Then, output the **postorder traversal** of the tree — a string consisting of the node types in postorder (left, right, root).

Output Format:
Your output should be a single line containing the postorder traversal of the tree. Each node type (F, B, or I) should appear **without any separators**.
Example: `{all_B_answer}` (do **NOT** include the backticks or quotes)."""

    def __init__(
        self,
        max_n: int = 4,
        probability_same_as_before: float = 0.7,
        base_image_width: int = 1000,
        base_image_height: int = 800,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}
        self._max_n = max_n
        self._probability_same_as_before = probability_same_as_before
        self._base_image_width = base_image_width
        self._base_image_height = base_image_height

        self._N: int | None = None
        self._string: str | None = None
        self._oracle_answer: str | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        """Return description adapted for visual input."""
        return dedent(
            """
            FBI Binary Tree rules:
            1) Binary strings are classified as:
               - B-string: only '0's
               - I-string: only '1's
               - F-string: both '0' and '1'

            2) An FBI tree is built recursively:
               - Each node represents a substring
               - If substring length > 1, split it in half into left and right
               - Label each node as B, I, or F based on its substring

            3) Output the postorder traversal (left, right, root) of node labels

            In the image:
            - The binary string is shown at the top
            - The FBI tree is visualized with hierarchical layout
            - Each node shows its substring range and FBI label (B/I/F)
            - Green nodes = B-strings (all 0s)
            - Blue nodes = I-strings (all 1s)
            - Orange nodes = F-strings (mixed 0s and 1s)
            - The tree is built by recursively splitting the string in half

            Output format: A string of F/B/I characters representing the postorder
            traversal (e.g., "BBBFIBF"). Do NOT include quotes or separators.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the binary string."""
        if self._string is None:
            return ""
        return f"Binary string: {self._string}"

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
            metadata={
                "text_prompt": f"{state_text}\n\n{self.description}",
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
            text=state_text,
            metadata={
                "text_prompt": f"{state_text}\n\n{self.description}",
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
        """Generate problem instance - ported from RLVE."""
        N = int(self.np_random.integers(1, self._max_n + 1))
        self._N = N

        # Generate binary string of length 2^N
        string = [str(self.np_random.integers(0, 2))]
        for i in range(1, 2**N):
            if self.np_random.random() < self._probability_same_as_before:
                string.append(string[i - 1])
            else:
                string.append(str(self.np_random.integers(0, 2)))

        self._string = "".join(string)
        assert len(self._string) == (2**N), f"string length should be {2**N}"

        # Compute postorder traversal of FBI tree
        def get_postorder(l: int, r: int) -> str:
            if l == r:
                if self._string[l] == "0":
                    return "B"
                else:
                    return "I"
            left = get_postorder(l, (l + r) // 2)
            right = get_postorder((l + r) // 2 + 1, r)

            # Determine root type based on children
            if left[-1] == "B" and right[-1] == "B":
                root = "B"
            elif left[-1] == "I" and right[-1] == "I":
                root = "I"
            else:
                root = "F"
            return left + right + root

        self._oracle_answer = get_postorder(0, 2**N - 1)
        assert len(self._oracle_answer) == (2 ** (N + 1) - 1), (
            f"reference_answer length should be {2 ** (N + 1) - 1}"
        )

    def _prompt_generate(self) -> str:
        """Generate text prompt."""
        all_B_answer = "B" * len(self._oracle_answer)
        return self.prompt_template.format(
            N=self._N,
            string=self._string,
            all_B_answer=all_B_answer,
        )

    def _process(self, answer: str | None) -> str | None:
        """Process answer string."""
        if answer is not None:
            answer = answer.strip()
            return answer
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score answer - ported from RLVE with beta=5."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        if len(processed_result) != len(self._oracle_answer):
            return 0.0
        for char in processed_result:
            if char not in ("F", "B", "I"):
                return 0.0
        correct_count = sum(
            float(a == b)
            for a, b in zip(self._oracle_answer, processed_result, strict=False)
        )
        mean_correct = correct_count / len(self._oracle_answer)
        return mean_correct**5.0

    def render(self) -> Image.Image:
        """Render the FBI binary tree structure."""
        if self._string is None:
            return Image.new(
                "RGB",
                (self._base_image_width, self._base_image_height),
                (255, 255, 255),
            )

        # Dynamically adjust image size based on tree depth
        num_leaves = 2**self._N
        image_width = max(self._base_image_width, num_leaves * 70)
        image_height = max(self._base_image_height, self._N * 150 + 300)

        # Create image with light background
        img = Image.new("RGB", (image_width, image_height), (245, 248, 250))
        draw = ImageDraw.Draw(img)

        # Load fonts
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 16)
            title_font = ImageFont.truetype(str(font_path), 24)
            label_font = ImageFont.truetype(str(font_path), 14)
            string_font = ImageFont.truetype(str(font_path), 18)
        else:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            string_font = ImageFont.load_default()

        # Add title
        title = f"FBI Binary Tree (N={self._N}, String Length={2**self._N})"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (image_width // 2 - tw // 2, 20),
            title,
            fill=(40, 60, 80),
            font=title_font,
        )

        # Draw the binary string
        string_y = 60
        string_text = f"Binary String: {self._string}"
        bbox = draw.textbbox((0, 0), string_text, font=string_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (image_width // 2 - tw // 2, string_y),
            string_text,
            fill=(40, 60, 80),
            font=string_font,
        )

        # Build tree structure and compute positions
        tree_structure = self._build_fbi_tree_structure()
        positions = self._compute_tree_positions(
            tree_structure, image_width, image_height
        )

        # Draw edges first
        for node_info in tree_structure:
            if node_info["left"] is not None and node_info["right"] is not None:
                node_idx = node_info["idx"]
                left_idx = node_info["left"]["idx"]
                right_idx = node_info["right"]["idx"]

                x1, y1 = positions[node_idx]
                x2, y2 = positions[left_idx]
                x3, y3 = positions[right_idx]

                # Draw edge to left child
                draw.line([(x1, y1), (x2, y2)], fill=(120, 140, 160), width=2)
                # Draw edge to right child
                draw.line([(x1, y1), (x3, y3)], fill=(120, 140, 160), width=2)

        # Draw nodes
        node_radius = 28
        for node_info in tree_structure:
            node_idx = node_info["idx"]
            x, y = positions[node_idx]
            fbi_type = node_info["fbi_type"]
            l, r = node_info["range"]

            # Determine node color based on FBI type
            if fbi_type == "B":
                node_fill = (144, 238, 144)  # Light green
                node_outline = (34, 139, 34)  # Forest green
            elif fbi_type == "I":
                node_fill = (173, 216, 230)  # Light blue
                node_outline = (70, 130, 180)  # Steel blue
            else:  # F
                node_fill = (255, 200, 124)  # Light orange
                node_outline = (255, 140, 0)  # Dark orange

            # Draw node circle
            draw.ellipse(
                [
                    x - node_radius,
                    y - node_radius,
                    x + node_radius,
                    y + node_radius,
                ],
                fill=node_fill,
                outline=node_outline,
                width=2,
            )

            # Draw FBI label in the node
            text = fbi_type
            bbox = draw.textbbox((0, 0), text, font=title_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x - tw // 2, y - th // 2 - 2),
                text,
                fill=(0, 0, 0),
                font=title_font,
            )

            # Draw range label below node
            range_text = f"[{l}:{r + 1}]"
            bbox = draw.textbbox((0, 0), range_text, font=label_font)
            tw = bbox[2] - bbox[0]
            draw.text(
                (x - tw // 2, y + node_radius + 4),
                range_text,
                fill=(80, 100, 120),
                font=label_font,
            )

        # Add legend
        legend_y = image_height - 100
        legend_x = 100

        # B-string example
        draw.ellipse(
            [legend_x - 18, legend_y - 18, legend_x + 18, legend_y + 18],
            fill=(144, 238, 144),
            outline=(34, 139, 34),
            width=2,
        )
        draw.text(
            (legend_x - 8, legend_y - 12),
            "B",
            fill=(0, 0, 0),
            font=string_font,
        )
        draw.text(
            (legend_x + 28, legend_y - 10),
            "B-string (all 0s)",
            fill=(40, 60, 80),
            font=label_font,
        )

        # I-string example
        legend_x += 200
        draw.ellipse(
            [legend_x - 18, legend_y - 18, legend_x + 18, legend_y + 18],
            fill=(173, 216, 230),
            outline=(70, 130, 180),
            width=2,
        )
        draw.text(
            (legend_x - 7, legend_y - 12),
            "I",
            fill=(0, 0, 0),
            font=string_font,
        )
        draw.text(
            (legend_x + 28, legend_y - 10),
            "I-string (all 1s)",
            fill=(40, 60, 80),
            font=label_font,
        )

        # F-string example
        legend_x += 200
        draw.ellipse(
            [legend_x - 18, legend_y - 18, legend_x + 18, legend_y + 18],
            fill=(255, 200, 124),
            outline=(255, 140, 0),
            width=2,
        )
        draw.text(
            (legend_x - 8, legend_y - 12),
            "F",
            fill=(0, 0, 0),
            font=string_font,
        )
        draw.text(
            (legend_x + 28, legend_y - 10),
            "F-string (mixed 0s and 1s)",
            fill=(40, 60, 80),
            font=label_font,
        )

        # Add note
        note = "Output: Postorder traversal (left, right, root) of FBI labels"
        bbox = draw.textbbox((0, 0), note, font=label_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (image_width // 2 - tw // 2, image_height - 40),
            note,
            fill=(100, 120, 140),
            font=label_font,
        )

        return img

    def _build_fbi_tree_structure(self) -> list[dict[str, Any]]:
        """Build FBI tree structure for visualization.

        Returns a list of node info dicts with:
        - idx: node index in level-order
        - range: (left, right) indices in string
        - fbi_type: 'B', 'I', or 'F'
        - left: left child node info or None
        - right: right child node info or None
        """
        node_counter = [0]  # Use list to allow modification in nested function

        def build_tree(l: int, r: int) -> dict[str, Any]:
            node_idx = node_counter[0]
            node_counter[0] += 1

            # Determine FBI type for this substring
            substring = self._string[l : r + 1]
            has_zero = "0" in substring
            has_one = "1" in substring

            if has_zero and has_one:
                fbi_type = "F"
            elif has_zero:
                fbi_type = "B"
            else:
                fbi_type = "I"

            if l == r:
                # Leaf node
                return {
                    "idx": node_idx,
                    "range": (l, r),
                    "fbi_type": fbi_type,
                    "left": None,
                    "right": None,
                }
            else:
                # Internal node - split in half
                mid = (l + r) // 2
                left_child = build_tree(l, mid)
                right_child = build_tree(mid + 1, r)

                return {
                    "idx": node_idx,
                    "range": (l, r),
                    "fbi_type": fbi_type,
                    "left": left_child,
                    "right": right_child,
                }

        root = build_tree(0, len(self._string) - 1)

        # Flatten tree into list using level-order traversal
        result = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            result.append(node)
            if node["left"] is not None:
                queue.append(node["left"])
            if node["right"] is not None:
                queue.append(node["right"])

        return result

    def _compute_tree_positions(
        self, tree_structure: list[dict[str, Any]], image_width: int, image_height: int
    ) -> dict[int, tuple[float, float]]:
        """Compute positions for tree nodes using hierarchical layout.

        Args:
            tree_structure: List of node info dicts
            image_width: Width of the image
            image_height: Height of the image

        Returns dict mapping node_idx -> (x, y) coordinates.
        """
        if not tree_structure:
            return {}

        # Build adjacency from tree structure
        children_map = {}
        for node_info in tree_structure:
            idx = node_info["idx"]
            children = []
            if node_info["left"] is not None:
                children.append(node_info["left"]["idx"])
            if node_info["right"] is not None:
                children.append(node_info["right"]["idx"])
            children_map[idx] = children

        # Find depth of each node using BFS from root (idx=0)
        depths = {0: 0}
        queue = deque([0])
        max_depth = 0

        while queue:
            node = queue.popleft()
            depth = depths[node]
            max_depth = max(max_depth, depth)

            for child in children_map.get(node, []):
                depths[child] = depth + 1
                queue.append(child)

        # Group nodes by depth
        nodes_by_depth = {}
        for node, depth in depths.items():
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node)

        # Compute positions
        positions = {}
        margin_top = 120
        margin_bottom = 140
        margin_x = 60
        usable_height = image_height - margin_top - margin_bottom
        usable_width = image_width - 2 * margin_x

        for depth in range(max_depth + 1):
            nodes_at_depth = nodes_by_depth.get(depth, [])
            y = margin_top + (depth * usable_height / max(1, max_depth))

            # Distribute nodes evenly across width
            num_nodes = len(nodes_at_depth)
            for i, node in enumerate(sorted(nodes_at_depth)):
                if num_nodes == 1:
                    x = image_width / 2
                else:
                    x = margin_x + (i * usable_width / (num_nodes - 1))
                positions[node] = (x, y)

        return positions

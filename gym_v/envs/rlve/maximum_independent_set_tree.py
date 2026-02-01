"""Maximum Independent Set Tree environment for gym-v."""

from __future__ import annotations

from importlib import resources
import math
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEMaximumIndependentSetTreeEnv(Env):
    """RLVE Maximum Independent Set Tree as a single-turn visual environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices, labeled from `0` to `{N_minus_1}`.

The tree contains the following {N} - 1 = {N_minus_1} undirected edges. Each edge is represented as a tuple `(u, v)`, meaning there is an undirected edge **connecting vertex `u` to vertex `v`**:
{edges}

Each vertex has a weight, given as a list `R` of length {N}, where `R[i]` is the weight of vertex `i`. The weights are as follows:
{R}

Your task is to select a set of distinct vertices `x_1, x_2, ..., x_k` (you determine `k`), such that **no two selected vertices are adjacent**.
Your goal is to **maximize the total weight**: R[x_1] + R[x_2] + ... + R[x_k].

**Output Format:**
Your final answer should be a single line containing the selected vertices in **any order**, separated by **spaces**.
Example: `0 1 {N_minus_1}` (do **NOT** include the backticks or quotes); this means k = 3, with selected vertices x_1 = 0, x_2 = 1, and x_3 = {N_minus_1}."""

    def __init__(
        self,
        max_n: int = 10,
        node_radius: int = 22,
        image_size: int = 700,
        padding: int = 60,
        num_players: int = 1,
        rewarding_strategy: str = "(answer/gold)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 3.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._node_radius = node_radius
        self._image_size = image_size
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self.rewards = {
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        self._N: int | None = None
        self._edges: list[tuple[int, int]] | None = None
        self._R: list[int] | None = None
        self._reference_weight: int | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Maximum Independent Set Tree Problem:

            Given a tree (connected undirected graph with no cycles) with N vertices,
            each vertex has a weight. Select a set of distinct vertices such that no
            two selected vertices are adjacent (i.e., they form an independent set).

            Goal: Maximize the total weight of selected vertices.

            In the image:
            - Vertices are numbered and shown as circles
            - Each vertex shows its weight inside (e.g., "0:5" means vertex 0 has weight 5)
            - Edges connect adjacent vertices
            - Find the maximum weighted independent set

            Output format: A single line with selected vertex indices separated by spaces.
            Example: "0 2 4" (selects vertices 0, 2, and 4)
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
        """Generate a random tree with weights and compute maximum independent set."""
        N = int(self.np_random.integers(3, self._max_n + 1))
        self._N = N

        # Generate tree structure
        edges = []
        childrens = [[] for _ in range(N)]

        permutations = list(range(N))
        self.np_random.shuffle(permutations)
        root = permutations[0]

        for index, child in enumerate(permutations):
            if index == 0:
                continue
            parent = self.np_random.choice(permutations[:index])
            childrens[parent].append(child)
            u, v = min(parent, child), max(parent, child)
            edges.append((u, v))

        self._edges = edges

        # Generate weights for each vertex
        self._R = [int(self.np_random.integers(1, N + 1)) for _ in range(N)]

        # Compute maximum independent set using dynamic programming
        # dpF[u][0] = max weight when u is not selected
        # dpF[u][1] = max weight when u is selected
        dpF = [None] * N

        def dp(u: int) -> None:
            dpF[u] = [0, self._R[u]]
            for child in childrens[u]:
                dp(child)
                dpF[u][0] += max(dpF[child])
                dpF[u][1] += dpF[child][0]

        dp(root)
        self._reference_weight = max(dpF[root])

        # Backtrack to find the actual set
        picked = []

        def Pick(u: int, pick: bool) -> None:
            if pick:
                picked.append(u)
            for child in childrens[u]:
                if pick:
                    Pick(child, False)
                else:
                    Pick(child, bool(dpF[child][0] < dpF[child][1]))

        Pick(root, dpF[root][0] < dpF[root][1])

        self._oracle_answer = " ".join(map(str, picked))

    def _prompt_generate(self) -> str:
        N = self._N
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges="\n".join(f"({u}, {v})" for u, v in self._edges),
            R="\n".join(f"R[{i}] = {self._R[i]}" for i in range(N)),
        )

    def _process(self, answer: str | None) -> list[int] | None:
        if answer is None:
            return None
        answer = answer.strip()
        if not answer:  # Empty string
            return None
        try:
            answer_array = list(map(int, answer.split()))
            return answer_array
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score answer based on the weight of the independent set found."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0

        picked = processed_result
        N = self._N

        # Check for validity
        if len(set(picked)) != len(picked):
            return 0.0
        if not all(0 <= vertex < N for vertex in picked):
            return 0.0

        # Check that no two picked vertices are adjacent
        picked_set = set(picked)
        for u, v in self._edges:
            if u in picked_set and v in picked_set:
                return 0.0

        # Calculate weight
        answer_weight = sum(self._R[u] for u in picked)
        gold = self._reference_weight

        # Should never exceed gold
        assert (
            answer_weight <= gold
        ), f"Answer weight {answer_weight} exceeds gold {gold}"

        if self.rewards["rewarding_strategy"] == "(answer/gold)^beta":
            return self.rewards["rewarding_weight"] * (
                (answer_weight / gold) ** self.rewards["rewarding_beta"]
            )
        elif self.rewards["rewarding_strategy"] == "gold=answer":
            return self.rewards["rewarding_weight"] * (gold == answer_weight)
        else:
            raise ValueError(
                f"Invalid rewarding strategy: {self.rewards['rewarding_strategy']}"
            )

    def render(self) -> Image.Image:
        if self._N is None:
            return Image.new(
                "RGB", (self._image_size, self._image_size), (240, 245, 248)
            )

        N = self._N
        edges = self._edges
        R = self._R

        # Create image with soft blue-gray background
        img = Image.new("RGB", (self._image_size, self._image_size), (240, 245, 248))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 20)
            weight_font = ImageFont.truetype(str(font_path), 16)
            title_font = ImageFont.truetype(str(font_path), 28)
            info_font = ImageFont.truetype(str(font_path), 20)
        else:
            font = ImageFont.load_default()
            weight_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()

        # Add title
        title = "Maximum Independent Set Tree"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, 20),
            title,
            fill=(50, 50, 70),
            font=title_font,
        )

        # Compute circular layout
        center_x = self._image_size // 2
        center_y = self._image_size // 2 + 20
        radius = (self._image_size - 2 * self._padding - 60) // 2

        positions = []
        for i in range(N):
            angle = 2 * math.pi * i / N - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y))

        # Draw edges
        for u, v in edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]

            # Draw edge line in gray
            draw.line([(x1, y1), (x2, y2)], fill=(150, 150, 170), width=3)

        # Draw nodes with weights
        for i, (x, y) in enumerate(positions):
            # Draw node shadow
            shadow_offset = 4
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

            # Draw node (orange/yellow for vertices with weight)
            draw.ellipse(
                [
                    x - self._node_radius,
                    y - self._node_radius,
                    x + self._node_radius,
                    y + self._node_radius,
                ],
                fill=(255, 180, 80),
                outline=(200, 130, 50),
                width=3,
            )

            # Draw node label with weight
            text = f"{i}:{R[i]}"
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x - tw // 2, y - th // 2),
                text,
                fill=(255, 255, 255),
                font=font,
            )

        # Add graph information
        info_text = f"Vertices: {N}  |  Edges: {len(edges)}  |  Maximize weighted independent set"
        bbox = draw.textbbox((0, 0), info_text, font=info_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, self._image_size - 40),
            info_text,
            fill=(100, 100, 120),
            font=info_font,
        )

        return img

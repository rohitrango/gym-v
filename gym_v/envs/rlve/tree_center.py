"""Tree Center environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
import math
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.envs.rlve.parameter_controllers import get_controller_for_env
from gym_v.logger import get_logger

logger = get_logger()


class RLVETreeCenterEnv(Env):
    """RLVE Tree Center as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices, labeled from `0` to `{N_minus_1}`.

Each vertex has a cost, given as a list `C` of length {N}, where `C[i]` is the cost of vertex i:
{C}

The tree contains the following {N} - 1 = {N_minus_1} undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning there is an undirected edge **connecting vertex `u` to vertex `v` with weight `w`:
{edges}

Your task is to select a single vertex `r` (where `r` is in the range 0 to {N_minus_1}).
Try your best to **minimize** dist(0, r) * C[0] + dist(1, r) * C[1] + ... + dist({N_minus_1}, r) * C[{N_minus_1}], where `dist(i, j)` is the distance between vertices i and j in the tree. The distance between two vertices is defined as the sum of the weights of the edges on the unique path connecting them (since the graph is a tree, there is exactly one unique path between any two vertices).

**Output Format:** Your final answer should be a single integer `r` (the index of the selected vertex). Example: `0` (do **NOT** include the backticks or quotes)."""

    def __init__(
        self,
        max_n: int | None = None,
        node_radius: int = 22,
        image_size: int = 700,
        padding: int = 60,
        num_players: int = 1,
        difficulty: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(difficulty=difficulty, **kwargs)
        self._node_radius = node_radius
        self._image_size = image_size
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Check if explicit parameters or difficulty is provided
        self._use_explicit_params = max_n is not None
        self._use_difficulty = difficulty is not None

        self._parameter_controller = get_controller_for_env(
            self.__class__.__name__,
            self._difficulty if self._difficulty is not None else 0,
        )

        if self._use_explicit_params:
            self._max_n = max_n
        elif self._use_difficulty:
            self._apply_difficulty_parameters()
        else:
            # Use original defaults (backward compatibility)
            self._max_n = 10

        self._N: int | None = None
        self._C: list[int] | None = None
        self._edges: list[tuple[int, int, int]] | None = None
        self._oracle_answer: int | None = None
        self._gold_answer: int | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    def _apply_difficulty_parameters(self) -> None:
        """Apply parameters from the controller."""
        if self._use_difficulty and self._parameter_controller is not None:
            params = self._parameter_controller.get_parameters()
            self._max_n = params.get("max_n", 10)

    @property
    def description(self) -> str:
        return dedent(
            """
            Tree Center Problem:

            Given a tree (connected undirected graph with no cycles) with N vertices,
            each vertex has a cost C[i]. Select a single vertex r to minimize the
            weighted sum: dist(0, r) * C[0] + dist(1, r) * C[1] + ... + dist(N-1, r) * C[N-1],
            where dist(i, j) is the sum of edge weights on the unique path between
            vertices i and j.

            In the image:
            - Vertices are numbered and shown as circles
            - The center node(s) (optimal solution) are highlighted in gold/orange
            - Regular nodes are shown in light blue
            - Edges are shown as lines with weights labeled
            - Each node's cost is displayed

            Output Format: Your final answer should be a single integer r (the index
            of the selected vertex). Example: 0 (do NOT include backticks or quotes).
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
            "oracle_answer": str(self._oracle_answer),
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
            "oracle_answer": str(self._oracle_answer),
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
        """Generate tree center problem instance."""
        N = int(self.np_random.integers(2, self._max_n + 1))
        self._N = N

        # Generate costs for each vertex
        C = [int(self.np_random.integers(1, N + 1)) for _ in range(N)]
        self._C = C

        # Generate tree edges using random permutation
        edges = []
        permutations = list(range(N))
        self.np_random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u = vertex
            v = int(self.np_random.choice(permutations[:index]))
            if u > v:
                u, v = v, u
            w = int(self.np_random.integers(1, N + 1))
            edges.append((u, v, w))
        self.np_random.shuffle(edges)
        self._edges = edges

        # Build adjacency list
        adjacent = [[] for _ in range(N)]
        for u, v, w in edges:
            adjacent[u].append((v, w))
            adjacent[v].append((u, w))

        # Compute optimal solution using tree DP
        self._oracle_answer = 0
        self._gold_answer = 0
        subtree_sumC = [0] * N

        def DFS(u: int, parent: int, depth: int) -> None:
            subtree_sumC[u] = C[u]
            self._gold_answer += depth * C[u]
            for v, w in adjacent[u]:
                if v == parent:
                    continue
                DFS(v, u, depth + w)
                subtree_sumC[u] += subtree_sumC[v]

        DFS(0, -1, 0)

        def FindSolution(u: int, parent: int, now_answer: int) -> None:
            if now_answer < self._gold_answer:
                self._oracle_answer = u
                self._gold_answer = now_answer
            for v, w in adjacent[u]:
                if v == parent:
                    continue
                FindSolution(
                    v,
                    u,
                    now_answer
                    + (subtree_sumC[0] - subtree_sumC[v]) * w
                    - subtree_sumC[v] * w,
                )

        FindSolution(0, -1, self._gold_answer)

    def _prompt_generate(self) -> str:
        N = self._N
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            C="\n".join(f"C[{i}]={Ci}" for i, Ci in enumerate(self._C)),
            edges="\n".join(f"({u}, {v}, {w})" for u, v, w in self._edges),
        )

    def _process(self, answer: str | None) -> int | None:
        if answer is not None:
            answer = answer.strip()
            try:
                int_answer = int(answer)
                return int_answer
            except ValueError:
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        root = processed_result
        if not (0 <= root < self._N):
            return 0.0
        adjacent = [[] for _ in range(self._N)]
        for u, v, w in self._edges:
            adjacent[u].append((v, w))
            adjacent[v].append((u, w))

        # Compute the answer for the selected root
        answer_value = 0

        def DFS(u: int, parent: int, depth: int) -> None:
            nonlocal answer_value
            answer_value += depth * self._C[u]
            for v, w in adjacent[u]:
                if v == parent:
                    continue
                DFS(v, u, depth + w)

        DFS(root, -1, 0)

        gold = self._gold_answer
        assert gold <= answer_value, "gold <= answer"

        # Use (gold/answer)^beta rewarding strategy with beta=8
        return (gold / answer_value) ** 8

    def render(self) -> Image.Image:
        if self._N is None:
            return Image.new(
                "RGB", (self._image_size, self._image_size), (240, 245, 248)
            )

        N = self._N
        edges = self._edges

        # Create image with soft blue-gray background
        img = Image.new("RGB", (self._image_size, self._image_size), (240, 245, 248))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 22)
            weight_font = ImageFont.truetype(str(font_path), 16)
            title_font = ImageFont.truetype(str(font_path), 28)
            info_font = ImageFont.truetype(str(font_path), 20)
            cost_font = ImageFont.truetype(str(font_path), 14)
        else:
            font = ImageFont.load_default()
            weight_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()
            cost_font = ImageFont.load_default()

        # Add title
        title = "Tree Center Problem"
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

        # Draw edges with weights
        max_weight = max(w for _, _, w in edges) if edges else 1
        for u, v, w in edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]

            # Draw edge line
            intensity = int(100 + 80 * (w / max_weight))
            edge_color = (intensity, intensity + 20, intensity + 40)
            draw.line([(x1, y1), (x2, y2)], fill=edge_color, width=3)

            # Draw weight label
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            weight_text = str(w)
            bbox = draw.textbbox((0, 0), weight_text, font=weight_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            # Weight label background
            padding = 5
            draw.ellipse(
                [
                    mid_x - tw // 2 - padding,
                    mid_y - th // 2 - padding,
                    mid_x + tw // 2 + padding,
                    mid_y + th // 2 + padding,
                ],
                fill=(255, 250, 245),
                outline=(220, 200, 180),
                width=2,
            )
            draw.text(
                (mid_x - tw // 2, mid_y - th // 2),
                weight_text,
                fill=(200, 80, 40),
                font=weight_font,
            )

        # Draw nodes
        for i, (x, y) in enumerate(positions):
            # Check if this is the center node
            is_center = i == self._oracle_answer

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

            # Draw node with color based on whether it's the center
            if is_center:
                # Gold/orange for center node
                node_fill = (255, 215, 0)
                node_outline = (218, 165, 32)
            else:
                # Light blue for regular nodes
                node_fill = (135, 206, 250)
                node_outline = (70, 130, 180)

            draw.ellipse(
                [
                    x - self._node_radius,
                    y - self._node_radius,
                    x + self._node_radius,
                    y + self._node_radius,
                ],
                fill=node_fill,
                outline=node_outline,
                width=3,
            )

            # Draw node label
            text = str(i)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x - tw // 2, y - th // 2),
                text,
                fill=(0, 0, 0) if is_center else (255, 255, 255),
                font=font,
            )

            # Draw cost label below node
            cost_text = f"C={self._C[i]}"
            bbox = draw.textbbox((0, 0), cost_text, font=cost_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x - tw // 2, y + self._node_radius + 5),
                cost_text,
                fill=(80, 80, 100),
                font=cost_font,
            )

        # Add info text
        info_text = (
            f"Vertices: {N}  |  Edges: {len(edges)}  |  Center: {self._oracle_answer}"
        )
        bbox = draw.textbbox((0, 0), info_text, font=info_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, self._image_size - 40),
            info_text,
            fill=(100, 100, 120),
            font=info_font,
        )

        return img

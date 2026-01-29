"""Maximum Clique environment for gym-v."""

from __future__ import annotations

from importlib import resources
import math
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEMaximumCliqueEnv(Env):
    """RLVE Maximum Clique as a single-turn visual environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The graph contains the following undirected edges:
{edges}

Your task is to select a subset of vertices `v1, v2, ..., vk` such that:
- 0 ≤ v1, v2, ..., vk < {N} and all selected vertices are **distinct**.
- The selected vertices form a **clique** — that is, **every pair** of distinct selected vertices is connected by **at least one edge**.
- Your goal is to **maximize** the number of selected vertices k.

**Output Format:**
Your final answer should be a single line containing the selected vertex indices `v1, v2, ..., vk`, separated by **spaces**.
Example: `0 2 3` (do **NOT** include the backticks or quotes); this means the selected clique has size k = 3, with vertices 0, 2, and 3."""

    def __init__(
        self,
        max_n: int = 12,
        edge_density: float = 0.5,
        node_radius: int = 18,
        image_size: int = 700,
        padding: int = 60,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._edge_density = edge_density
        self._node_radius = node_radius
        self._image_size = image_size
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._N: int | None = None
        self._edges: list[tuple[int, int]] | None = None
        self._gold_answer: int | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Maximum Clique Problem:

            Given an undirected graph with N vertices, find the largest subset of
            vertices such that every pair of vertices in the subset is connected
            by an edge (forming a clique).

            In the image:
            - Vertices are numbered and shown as circles
            - Edges are shown as lines connecting vertices
            - Find the maximum clique (fully connected subgraph)

            Output format: A single line containing the selected vertex indices
            separated by spaces. Example: "0 2 3" means vertices 0, 2, and 3
            form a clique.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the current state."""
        return self._prompt or ""

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
        N = int(self.np_random.integers(2, self._max_n + 1))
        self._N = N

        num_edges = int(self._edge_density * N * (N - 1) / 2)
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        edge_indices = list(
            self.np_random.choice(len(all_edges), size=num_edges, replace=False)
        )
        edges = [all_edges[i] for i in edge_indices]
        self.np_random.shuffle(edges)
        self._edges = edges

        adjacent = [0] * N
        for u, v in edges:
            adjacent[u] |= 1 << v
            adjacent[v] |= 1 << u

        reference_answer = []
        clique = []

        def DFS(u: int, allowed_set: int) -> None:
            nonlocal reference_answer
            if len(clique) + (N - u) <= len(reference_answer):
                return
            if u == N:
                if len(clique) > len(reference_answer):
                    reference_answer = clique.copy()
                return
            if allowed_set & (1 << u):
                clique.append(u)
                DFS(u + 1, allowed_set & adjacent[u])
                clique.pop()
            DFS(u + 1, allowed_set)

        DFS(0, (1 << N) - 1)

        self._gold_answer = len(reference_answer)
        self._oracle_answer = " ".join(map(str, reference_answer))

    def _prompt_generate(self) -> str:
        N = self._N
        edges = self._edges
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges="\n".join(f"({u}, {v})" for u, v in edges),
        )

    def _process(self, answer: str | None) -> list[int] | None:
        if answer is None:
            return None
        answer = answer.strip()
        try:
            answer_array = list(map(int, answer.split()))
            return answer_array
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        clique = processed_result
        N = self._N
        if len(clique) != len(set(clique)):
            return 0.0
        for vertex in clique:
            if not (0 <= vertex < N):
                return 0.0
        edges_set = set(self._edges)
        for u in clique:
            for v in clique:
                if u < v:
                    if (u, v) not in edges_set:
                        return 0.0
        gold = self._gold_answer
        answer_size = len(clique)
        return (answer_size / gold) ** 5

    def render(self) -> Image.Image:
        if self._N is None:
            return Image.new(
                "RGB", (self._image_size, self._image_size), (245, 248, 250)
            )

        N = self._N
        edges = self._edges

        # 使用更柔和的背景色
        img = Image.new("RGB", (self._image_size, self._image_size), (245, 248, 250))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 18)  # 增大字体
            title_font = ImageFont.truetype(str(font_path), 24)
        else:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()

        # 添加标题
        title = "Maximum Clique Problem"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, 20),
            title,
            fill=(50, 50, 70),
            font=title_font,
        )

        center_x = self._image_size // 2
        center_y = self._image_size // 2 + 20  # 向下偏移以留出标题空间
        radius = (self._image_size - 2 * self._padding - 60) // 2

        positions = []
        for i in range(N):
            angle = 2 * math.pi * i / N - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y))

        # 绘制边（使用更粗的线条和更柔和的颜色）
        for u, v in edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            draw.line([(x1, y1), (x2, y2)], fill=(120, 140, 160), width=3)

        # 绘制节点（使用渐变效果的颜色和阴影）
        for i, (x, y) in enumerate(positions):
            # 绘制阴影效果
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

            # 绘制主节点（使用更鲜艳的渐变色）
            draw.ellipse(
                [
                    x - self._node_radius,
                    y - self._node_radius,
                    x + self._node_radius,
                    y + self._node_radius,
                ],
                fill=(70, 180, 120),  # 更鲜艳的绿色
                outline=(40, 120, 80),  # 深绿色边框
                width=3,
            )

            # 绘制节点编号
            text = str(i)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x - tw // 2, y - th // 2), text, fill=(255, 255, 255), font=font)

        # 添加图信息说明
        info_text = f"Vertices: {N}  |  Edges: {len(edges)}"
        bbox = draw.textbbox((0, 0), info_text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, self._image_size - 35),
            info_text,
            fill=(100, 100, 120),
            font=font,
        )

        return img

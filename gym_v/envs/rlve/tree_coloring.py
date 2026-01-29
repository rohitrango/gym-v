"""Tree Coloring environment for gym-v."""

from __future__ import annotations

from importlib import resources
import math
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVETreeColoringEnv(Env):
    """RLVE Tree Coloring as a single-turn visual environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices, labeled from `0` to `{N_minus_1}`.

The tree contains the following {N} - 1 = {N_minus_1} undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning there is an undirected edge **connecting vertex `u` to vertex `v` with weight `w`:
{edges}

Your task is to **select exactly {K} distinct vertices**. These selected vertices are called **colored**, and the remaining {N} - {K} = {N_minus_K} vertices are called **uncolored**. Try your best to **maximize the total distance**, defined as:
- The sum of all pairwise distances **between colored vertices**,
- Plus the sum of all pairwise distances **between uncolored vertices**.

(Note: Since the graph is a tree, there is exactly one unique path between any two vertices.)

**Output Format:**
Your final answer should be a single line containing the {K} selected (colored) vertices in any order, separated by **spaces**.
Example: `{first_K_vertices}` (do **NOT** include the backticks or quotes)."""

    def __init__(
        self,
        max_n: int = 10,
        node_radius: int = 22,
        image_size: int = 700,
        padding: int = 60,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._node_radius = node_radius
        self._image_size = image_size
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._N: int | None = None
        self._K: int | None = None
        self._edges: list[tuple[int, int, int]] | None = None
        self._oracle_answer_distance: int | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Tree Coloring Problem:

            Given a tree (connected undirected graph with no cycles) with N vertices,
            select exactly K distinct vertices to be "colored". The remaining vertices
            are "uncolored".

            Goal: Maximize the total distance, defined as:
            - Sum of pairwise distances between colored vertices
            - Plus sum of pairwise distances between uncolored vertices

            In the image:
            - Vertices are numbered and shown as circles
            - Edges are shown as lines with weights
            - Find K vertices that maximize the total distance

            Output format: A single line with K selected vertex indices separated
            by spaces. Example: "0 3 5"
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
        N = int(self.np_random.integers(3, self._max_n + 1))
        self._N = N

        K = int(self.np_random.integers(1, N))
        self._K = K

        edges = []
        permutations = list(range(N))
        self.np_random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u = vertex
            v = self.np_random.choice(permutations[:index])
            if u > v:
                u, v = v, u
            w = int(self.np_random.integers(1, N + 1))
            edges.append((u, v, w))
        self.np_random.shuffle(edges)
        self._edges = edges

        adjacency_list = [[] for _ in range(N)]
        for u, v, w in edges:
            adjacency_list[u].append((v, w))
            adjacency_list[v].append((u, w))

        best_colored = list(range(K))
        best_distance = self._compute_distance(best_colored, N, K, adjacency_list)

        for _ in range(min(100, 2 ** min(N, 10))):
            colored = list(self.np_random.choice(N, size=K, replace=False))
            distance = self._compute_distance(colored, N, K, adjacency_list)
            if distance > best_distance:
                best_distance = distance
                best_colored = colored

        self._oracle_answer_distance = best_distance
        self._oracle_answer = " ".join(map(str, best_colored))

    def _compute_distance(
        self,
        colored_vertices: list[int],
        N: int,
        K: int,
        adjacency_list: list[list[tuple[int, int]]],
    ) -> int:
        colored = [0] * N
        for v in colored_vertices:
            colored[v] = 1

        Size = [0] * N
        total = 0

        def DFS(u: int, parent: int) -> None:
            nonlocal total
            Size[u] = 1
            for v, w in adjacency_list[u]:
                if v == parent:
                    continue
                DFS(v, u)
                total += w * (
                    colored[v] * (K - colored[v])
                    + (Size[v] - colored[v]) * ((N - K) - (Size[v] - colored[v]))
                )
                Size[u] += Size[v]
                colored[u] += colored[v]

        DFS(0, -1)
        return total

    def _prompt_generate(self) -> str:
        N = self._N
        K = self._K
        edges = self._edges
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            K=K,
            N_minus_K=N - K,
            first_K_vertices=" ".join(map(str, range(K))),
            edges="\n".join(f"({u}, {v}, {w})" for u, v, w in edges),
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
        colored_vertices = processed_result
        N = self._N
        K = self._K
        if len(colored_vertices) != K:
            return 0.0
        if len(set(colored_vertices)) != K:
            return 0.0
        if not all(0 <= vertex < N for vertex in colored_vertices):
            return 0.0
        adjacency_list = [[] for _ in range(N)]
        for u, v, w in self._edges:
            adjacency_list[u].append((v, w))
            adjacency_list[v].append((u, w))

        answer_distance = self._compute_distance(colored_vertices, N, K, adjacency_list)
        gold = self._oracle_answer_distance

        return (answer_distance / gold) ** 2

    def render(self) -> Image.Image:
        if self._N is None:
            return Image.new(
                "RGB", (self._image_size, self._image_size), (240, 245, 248)
            )

        N = self._N
        K = self._K
        edges = self._edges

        # 使用柔和的蓝灰色背景
        img = Image.new("RGB", (self._image_size, self._image_size), (240, 245, 248))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 22)  # 节点编号字体
            weight_font = ImageFont.truetype(str(font_path), 18)  # 权重字体
            title_font = ImageFont.truetype(str(font_path), 28)  # 标题字体
            info_font = ImageFont.truetype(str(font_path), 20)  # 信息栏字体
        else:
            font = ImageFont.load_default()
            weight_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()

        # 添加标题
        title = "Tree Coloring Problem"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, 20),
            title,
            fill=(50, 50, 70),
            font=title_font,
        )

        center_x = self._image_size // 2
        center_y = self._image_size // 2 + 20
        radius = (self._image_size - 2 * self._padding - 60) // 2

        positions = []
        for i in range(N):
            angle = 2 * math.pi * i / N - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y))

        # 绘制边，使用渐变颜色表示权重
        max_weight = max(w for _, _, w in edges) if edges else 1
        for u, v, w in edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]

            # 根据权重调整边的颜色（权重越大，颜色越深）
            intensity = int(100 + 80 * (w / max_weight))
            edge_color = (intensity, intensity + 20, intensity + 40)
            draw.line([(x1, y1), (x2, y2)], fill=edge_color, width=3)

            # 绘制权重标签
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            weight_text = str(w)
            bbox = draw.textbbox((0, 0), weight_text, font=weight_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            # 权重标签背景（圆角矩形效果）
            padding = 6
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

        # 绘制节点
        for i, (x, y) in enumerate(positions):
            # 绘制节点阴影
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

            # 绘制节点主体（橙黄色渐变）
            draw.ellipse(
                [
                    x - self._node_radius,
                    y - self._node_radius,
                    x + self._node_radius,
                    y + self._node_radius,
                ],
                fill=(255, 165, 60),  # 亮橙色
                outline=(200, 120, 40),  # 深橙色边框
                width=3,
            )

            # 绘制节点编号
            text = str(i)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x - tw // 2, y - th // 2),
                text,
                fill=(255, 255, 255),  # 白色文字
                font=font,
            )

        # 添加图信息说明
        info_text = f"Vertices: {N}  |  Edges: {len(edges)}  |  Select K = {K} vertices"
        bbox = draw.textbbox((0, 0), info_text, font=info_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (self._image_size // 2 - tw // 2, self._image_size - 40),
            info_text,
            fill=(100, 100, 120),
            font=info_font,
        )

        return img

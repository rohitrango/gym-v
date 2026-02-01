"""Warehouse Construction environment for gym-v (self-contained)."""

from __future__ import annotations

from collections import deque
from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEWarehouseConstructionEnv(Env):
    """RLVE Warehouse Construction as a single-turn environment.

    Given N factories arranged from top to bottom along a mountain, each with
    distance from the top, product count, and warehouse construction cost,
    find the optimal subset of factories to build warehouses that minimizes
    total cost (construction + transportation).

    Source: https://www.luogu.com.cn/problem/P2120
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given {N} factories arranged from top to bottom along a mountain, indexed from 0 to {N_minus_1}. Factory 0 is at the top and factory {N_minus_1} is at the bottom.

Each factory has
- Distance from factory 0: {D}
- Number of products: {P}
- Cost to build a warehouse at that factory: {C}

You can choose to build warehouses at any subset of factories.
- A warehouse can store any number of products.
- If a factory does not build a warehouse, all its products must be sent **downhill** to a factory with a warehouse (i.e., to a factory with a higher index). Transporting one product over one unit of distance costs 1.
- The total cost is the sum of warehouse construction costs and product transportation costs. Try your best to minimize the total cost.

**Output Format:** Output a single line containing the indices of the factories where warehouses should be built, separated by spaces (in any order)."""

    def __init__(
        self,
        n: int = 6,
        cell_height: int = 80,
        cell_width: int = 120,
        padding: int = 40,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._n = n
        self._cell_height = cell_height
        self._cell_width = cell_width
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._distances: list[int] | None = None
        self._products: list[int] | None = None
        self._costs: list[int] | None = None
        self._prompt: str | None = None
        self._gold_answer: int | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n:
            size_hint = f"{self._n} factories"
        else:
            size_hint = "N factories"

        return dedent(
            f"""
            Warehouse Construction Problem:

            Given {size_hint} arranged from top to bottom along a mountain,
            find the optimal subset of factories to build warehouses that minimizes
            total cost (construction + transportation).

            Rules:
            1) Each factory has: distance from top, product count, warehouse cost
            2) Warehouses can store unlimited products
            3) Products must be transported downhill to nearest warehouse
            4) Transportation cost = 1 per product per unit distance
            5) Total cost = sum of construction costs + transportation costs

            Goal: Minimize total cost by choosing optimal warehouse locations.

            In the visualization:
            - Factories are arranged vertically from top (0) to bottom (N-1)
            - Each factory shows: index, distance, products (P), warehouse cost (C)
            - Mountain terrain is shown on the left side
            - Downhill flow direction is from top to bottom
            - Color intensity indicates relative warehouse construction cost

            Output format: Space-separated indices of factories with warehouses (e.g., "0 3 5").
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
            text=None,
            metadata={
                "state_text": state_text,
                "text_prompt": self._prompt,
                "rlve_gold_answer": self._gold_answer,
            },
        )
        info = {
            "gold_answer": self._gold_answer,
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
                "rlve_gold_answer": self._gold_answer,
            },
        )
        info = {
            "gold_answer": self._gold_answer,
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
        """Generate a warehouse construction problem instance.

        Ports generation logic from RLVE using self.np_random.
        Uses convex hull trick dynamic programming to compute optimal cost.
        """
        N = self._n
        if N < 2:
            raise ValueError("N should be >= 2")

        # Generate strictly increasing distances
        D = list(self.np_random.choice(range(1, 2 * N + 1), N - 1, replace=False))
        D.sort()
        D = [0] + D

        # Generate products and costs
        P = [int(self.np_random.integers(0, N + 1)) for _ in range(N)]
        C = [int(self.np_random.integers(1, N * 2 + 1)) for _ in range(N)]

        self._distances = D
        self._products = P
        self._costs = C

        # Compute optimal cost using convex hull trick DP (from RLVE source)
        Q = [0] * (N + 1)
        R = [0] * (N + 1)
        for i in range(1, N + 1):
            Q[i] = Q[i - 1] + P[i - 1]
            R[i] = R[i - 1] + D[i - 1] * P[i - 1]

        # f[i] will hold the DP value corresponding to "having built a warehouse at factory i-1"
        f = [0] * (N + 1)

        # Helper functions mirroring C++ code
        def decx(idx: int) -> int:
            return Q[idx]

        def decy(idx: int) -> int:
            return f[idx] + R[idx]

        def maked(i: int, u: int) -> int:
            return f[u] + D[i - 1] * (Q[i] - Q[u]) - (R[i] - R[u]) + C[i - 1]

        # Convex hull trick with deque
        dq: deque[int] = deque([0])

        for i in range(1, N + 1):
            # Pop from left while next-oldest is better at x = D[i-1]
            while len(dq) >= 2:
                u1, u2 = dq[0], dq[1]
                if decy(u2) - decy(u1) <= D[i - 1] * (decx(u2) - decx(u1)):
                    dq.popleft()
                else:
                    break

            # Use best u = dq[0] to compute f[i]
            u = dq[0]
            f[i] = maked(i, u)

            # Pop from right while new line i makes it obsolete
            while len(dq) >= 2:
                u1, u2 = dq[-1], dq[-2]
                if (decy(u1) - decy(u2)) * (decx(i) - decx(u1)) >= (
                    decy(i) - decy(u1)
                ) * (decx(u1) - decx(u2)):
                    dq.pop()
                else:
                    break

            # Add new candidate i
            dq.append(i)

        # Find minimum cost among last non-empty factories
        ans = f[N]
        best_x = N
        x = N
        while x > 0 and P[x - 1] == 0:
            x -= 1
            if f[x] < ans:
                ans = f[x]
                best_x = x

        self._gold_answer = ans

        # Backtrack to find the optimal solution (which factories to build warehouses)
        # We need to reconstruct the DP choices
        # f[i] was computed from f[parent[i]] where parent[i] is stored during DP
        # Let's recompute with tracking
        parent = [0] * (N + 1)
        f2 = [0] * (N + 1)
        dq2: deque[int] = deque([0])

        for i in range(1, N + 1):
            while len(dq2) >= 2:
                u1, u2 = dq2[0], dq2[1]
                if decy(u2) - decy(u1) <= D[i - 1] * (decx(u2) - decx(u1)):
                    dq2.popleft()
                else:
                    break
            u = dq2[0]
            f2[i] = maked(i, u)
            parent[i] = u
            while len(dq2) >= 2:
                u1, u2 = dq2[-1], dq2[-2]
                if (decy(u1) - decy(u2)) * (decx(i) - decx(u1)) >= (
                    decy(i) - decy(u1)
                ) * (decx(u1) - decx(u2)):
                    dq2.pop()
                else:
                    break
            dq2.append(i)

        # Backtrack from best_x to find warehouses
        warehouses = []
        curr = best_x
        while curr > 0:
            warehouses.append(curr - 1)  # Convert to 0-indexed factory number
            curr = parent[curr]
        warehouses.reverse()

        self._oracle_answer = " ".join(map(str, warehouses))

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._distances is None:
            raise RuntimeError("No problem generated")
        N = self._n
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            D=" ".join(f"D[{i}]={d}" for i, d in enumerate(self._distances)),
            P=" ".join(f"P[{i}]={p}" for i, p in enumerate(self._products)),
            C=" ".join(f"C[{i}]={c}" for i, c in enumerate(self._costs)),
        )

    def _process(self, answer: str | None) -> list[int] | None:
        """Process the answer string into a list of factory indices."""
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
        """Score the answer based on cost optimality.

        Returns:
            -1.0: wrong format
            -0.5: invalid solution (out of bounds or no warehouse for products)
            (gold/answer)^5: cost-based reward (higher is better, 1.0 for optimal)
        """
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        N = self._n
        D = self._distances
        P = self._products
        C = self._costs

        # Build warehouse flags
        built = [False] * N
        answer_cost = 0

        for idx in processed_result:
            if 0 <= idx < N:
                built[idx] = True
                answer_cost += C[idx]
            else:
                return 0.0
        nearest_warehouse = None
        for i in range(N - 1, -1, -1):
            if built[i]:
                nearest_warehouse = i
            if P[i]:
                if nearest_warehouse is None:
                    return 0.0
                answer_cost += P[i] * (D[nearest_warehouse] - D[i])

        gold = self._gold_answer
        if answer_cost == 0:
            if gold == 0:
                return 1.0
            else:
                return 0.0
        beta = 5.0
        return (gold / answer_cost) ** beta

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the warehouse construction problem as a mountain layout.

        Shows:
        - Vertical factory arrangement (top to bottom)
        - Mountain terrain on the left
        - Each factory with index, distance, products, and warehouse cost
        - Color coding based on warehouse construction cost
        - Downhill flow indicators
        """
        if self._distances is None:
            raise RuntimeError("No problem generated")

        N = self._n
        cell_h = self._cell_height
        cell_w = self._cell_width
        padding = self._padding
        mountain_width = 150
        title_height = 100
        legend_height = 120

        width = padding * 2 + mountain_width + cell_w + 100
        height = padding * 2 + title_height + N * cell_h + legend_height
        img = Image.new("RGB", (width, height), (245, 245, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_title = ImageFont.truetype(font_path, 28)
            font_large = ImageFont.truetype(font_path, 18)
            font_medium = ImageFont.truetype(font_path, 14)
            font_small = ImageFont.truetype(font_path, 12)
        else:
            font_title = ImageFont.load_default()
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Draw title
        title = "Warehouse Construction - Mountain Factory Layout"
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, 20), title, fill=(30, 30, 30), font=font_title)

        # Starting Y for factories
        start_y = padding + title_height

        # Draw mountain terrain on the left
        mountain_x = padding
        mountain_base_y = start_y + N * cell_h
        mountain_color = (120, 120, 100)
        mountain_highlight = (140, 140, 120)

        # Draw mountain outline
        mountain_points = [
            (mountain_x, mountain_base_y),
            (mountain_x + 30, start_y + N * cell_h * 0.8),
            (mountain_x + 60, start_y + N * cell_h * 0.6),
            (mountain_x + 90, start_y + N * cell_h * 0.3),
            (mountain_x + mountain_width - 20, start_y),
            (mountain_x + mountain_width, start_y),
            (mountain_x + mountain_width, mountain_base_y),
        ]
        draw.polygon(mountain_points, fill=mountain_color, outline=mountain_highlight)

        # Add some mountain texture lines
        for i in range(N):
            y = start_y + i * cell_h + cell_h // 2
            x_left = mountain_x + 20 + i * 15
            x_right = mountain_x + mountain_width - 10
            draw.line((x_left, y, x_right, y), fill=mountain_highlight, width=1)

        # Find cost range for color scaling
        max_cost = max(self._costs) if self._costs else 1
        min_cost = min(self._costs) if self._costs else 0
        cost_range = max_cost - min_cost if max_cost > min_cost else 1

        # Draw factories
        factory_x = mountain_x + mountain_width + 20

        for i in range(N):
            y = start_y + i * cell_h
            cx = factory_x
            cy = y + cell_h // 2

            # Color based on warehouse cost (blue = cheap, red = expensive)
            cost_ratio = (self._costs[i] - min_cost) / cost_range
            r = int(100 + cost_ratio * 155)
            g = int(150 - cost_ratio * 100)
            b = int(200 - cost_ratio * 150)
            box_color = (r, g, b)

            # Draw factory box
            box_x0 = cx
            box_y0 = y + 10
            box_x1 = cx + cell_w
            box_y1 = y + cell_h - 10
            draw.rectangle(
                [box_x0, box_y0, box_x1, box_y1],
                fill=box_color,
                outline=(50, 50, 50),
                width=2,
            )

            # Draw connecting line from mountain to factory
            connect_x = mountain_x + mountain_width - 10
            draw.line((connect_x, cy, box_x0, cy), fill=(100, 100, 100), width=2)
            # Downhill arrow
            if i < N - 1:
                arrow_y = cy + cell_h // 2
                draw.line(
                    (connect_x + 5, cy, connect_x + 5, arrow_y),
                    fill=(150, 150, 150),
                    width=1,
                )
                draw.polygon(
                    [
                        (connect_x + 5, arrow_y),
                        (connect_x, arrow_y - 5),
                        (connect_x + 10, arrow_y - 5),
                    ],
                    fill=(150, 150, 150),
                )

            # Factory label
            text_color = (255, 255, 255) if cost_ratio > 0.6 else (20, 20, 20)

            # Factory index
            label = f"Factory {i}"
            label_bbox = draw.textbbox((0, 0), label, font=font_large)
            label_w = label_bbox[2] - label_bbox[0]
            draw.text(
                (cx + (cell_w - label_w) // 2, y + 15),
                label,
                fill=text_color,
                font=font_large,
            )

            # Distance
            dist_text = f"D={self._distances[i]}"
            draw.text((cx + 10, y + 38), dist_text, fill=text_color, font=font_medium)

            # Products
            prod_text = f"P={self._products[i]}"
            draw.text((cx + 10, y + 54), prod_text, fill=text_color, font=font_medium)

            # Warehouse cost
            cost_text = f"C={self._costs[i]}"
            draw.text(
                (cx + cell_w - 50, y + 54), cost_text, fill=text_color, font=font_medium
            )

        # Draw legend
        legend_y = start_y + N * cell_h + 20
        draw.text((padding, legend_y), "Legend:", fill=(30, 30, 30), font=font_large)
        legend_y += 25

        # Distance explanation
        draw.text(
            (padding + 10, legend_y),
            "D = Distance from factory 0 (top)",
            fill=(60, 60, 60),
            font=font_medium,
        )
        legend_y += 20

        # Products explanation
        draw.text(
            (padding + 10, legend_y),
            "P = Number of products at factory",
            fill=(60, 60, 60),
            font=font_medium,
        )
        legend_y += 20

        # Cost explanation
        draw.text(
            (padding + 10, legend_y),
            "C = Cost to build warehouse at factory",
            fill=(60, 60, 60),
            font=font_medium,
        )
        legend_y += 20

        # Flow explanation
        draw.text(
            (padding + 10, legend_y),
            "Products flow downhill to nearest warehouse below",
            fill=(100, 100, 100),
            font=font_small,
        )

        return img

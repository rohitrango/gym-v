"""Minimum dominating set grid environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEMinimumDominatingSetGridEnv(Env):
    """RLVE Minimum Dominating Set Grid as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""We have a grid with {N} rows and {M} columns (1-based indices). The cost of cell (i, j) is F[i][j]:
{F}

Select a set of **distinct** cells S such that every cell is either in S or has at least one **orthogonally adjacent** selected neighbor (up, down, left, or right). Minimize the total cost of selected cells (i.e., the sum of F[i][j] for all (i,j) ∈ S). Output K (the number of selected cells) lines: each line contains two integers `i j` (1-based), the row and column of a selected cell (in any order)."""

    def __init__(
        self,
        max_n_m: int = 4,
        cell_px: int = 60,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n_m = max_n_m
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._N: int | None = None
        self._M: int | None = None
        self._F: list[list[int]] | None = None
        self._gold_answer: int | None = None
        self._oracle_answer: str | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._N and self._M:
            size_hint = f"{self._N} x {self._M}"
        else:
            size_hint = "N x M"
        return dedent(
            f"""
            Minimum Dominating Set Grid rules:
            1) Select a subset of cells from the grid
            2) Every cell must be either selected OR adjacent (orthogonally: up/down/left/right) to a selected cell
            3) Minimize the total cost (sum of F[i][j] for selected cells)
            4) Each cell has a cost shown as F[i][j] in the grid

            In the image:
            - Each cell shows its cost F[i][j]
            - The grid is {size_hint} with 1-based indexing
            - Selected cells (if showing answer) are highlighted in green
            - Coverage ranges are shown with light blue for dominated cells

            Output format: K lines, each with two integers "i j" (1-based row and column)
            representing the selected cells in any order.
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
            metadata={
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
            text=state_text,
            metadata={
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
        """Generate a minimum dominating set grid instance with costs."""
        max_n_m = self._max_n_m
        if max_n_m < 2:
            raise ValueError("max_n_m must be >= 2")

        N = int(self.np_random.integers(2, max_n_m + 1))
        M = int(self.np_random.integers(2, max_n_m + 1))

        self._N = N
        self._M = M

        # Generate random costs for each cell
        F = [
            [int(self.np_random.integers(1, N * M + 1)) for _ in range(M)]
            for _ in range(N)
        ]
        self._F = F

        # Compute the optimal solution using dynamic programming
        S = 1 << M
        ALL = S - 1

        # Precompute helpers
        # popcount for every mask
        ones = [0] * S
        for m in range(S):
            ones[m] = m.bit_count()

        # shift coverage within a row (same row left/right neighbors)
        shift_cov = [0] * S
        for m in range(S):
            shift_cov[m] = (m | ((m << 1) & ALL) | (m >> 1)) & ALL

        # map bit -> column index
        bit_to_idx = {}
        for c in range(M):
            bit_to_idx[1 << c] = c

        # row_sums[i][mask]: cost of choosing 'mask' on row i (1-based rows for DP)
        # add a dummy row N+1 with all zero costs (to flush coverage of the last real row)
        row_sums = [[0] * S for _ in range(N + 2)]  # index 1..N, N+1 is zeros

        for i in range(1, N + 1):
            costs = F[i - 1]
            rs = row_sums[i]
            for mask in range(S):
                total = 0
                x = mask
                while x:
                    t = x & -x
                    total += costs[bit_to_idx[t]]
                    x -= t
                rs[mask] = total
        # row_sums[N+1] already zero

        # supersets list: for each 'need' mask, all p where p is a superset of 'need'
        supersets = [[] for _ in range(S)]
        for need in range(S):
            rem = ALL ^ need  # bits we are free to choose
            x = rem
            while True:
                supersets[need].append(need | x)
                if x == 0:
                    break
                x = (x - 1) & rem

        INF = float("inf")

        # DP arrays: f[p][j] and g[p][j]
        # f: minimal cost; g: number of depots (tie-breaker)
        f = [[INF] * S for _ in range(S)]
        g = [[INF] * S for _ in range(S)]

        # Initialize for first row: previous row (k) is 0
        rs1 = row_sums[1]
        for j in range(S):
            f[j][0] = rs1[j]
            g[j][0] = ones[j]

        # Transition rows 2..N+1 (N+1 is dummy zero-cost row)
        for i in range(2, N + 2):
            nf = [[INF] * S for _ in range(S)]
            ng = [[INF] * S for _ in range(S)]
            rsi = row_sums[i]

            for j in range(S):  # mask for row i-1
                sj = shift_cov[j]
                fj = f[j]
                gj = g[j]
                for k in range(S):  # mask for row i-2
                    base_cost = fj[k]
                    if base_cost == INF:
                        continue
                    base_cnt = gj[k]
                    need = ALL ^ (sj | k)  # columns still needing coverage on row i-1
                    for p in supersets[need]:  # mask for row i
                        v = base_cost + rsi[p]
                        c = base_cnt + ones[p]
                        if v < nf[p][j]:
                            nf[p][j] = v
                            ng[p][j] = c
                        elif v == nf[p][j] and c < ng[p][j]:
                            ng[p][j] = c

            f, g = nf, ng

        # Finalize: last (dummy) row must be p=0; scan any j
        best_cost = INF
        best_cnt = INF
        f0 = f[0]
        g0 = g[0]
        for j in range(S):
            v = f0[j]
            if v < best_cost:
                best_cost = v
                best_cnt = g0[j]
            elif v == best_cost and g0[j] < best_cnt:
                best_cnt = g0[j]

        if best_cost <= 0 or best_cost == INF:
            raise ValueError("gold_answer must be greater than 0")

        self._gold_answer = int(best_cost)

        # Generate a reference solution using a greedy approach
        # Note: This may not be optimal, but it provides a valid solution for testing
        self._oracle_answer = self._generate_greedy_solution()

    def _generate_greedy_solution(self) -> str:
        """Generate a greedy dominating set solution.

        This uses a greedy approach: repeatedly select the uncovered cell
        with minimum cost that covers the most uncovered cells.

        Returns:
            String with selected cells in format "i j\\n" (1-based)
        """
        N, M, F = self._N, self._M, self._F
        covered = [[False] * M for _ in range(N)]
        selected = []

        def get_coverage(i: int, j: int) -> int:
            """Count how many uncovered cells would be covered by selecting (i,j)."""
            count = 0
            for di, dj in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < M and not covered[ni][nj]:
                    count += 1
            return count

        def mark_covered(i: int, j: int) -> None:
            """Mark cell (i,j) and its neighbors as covered."""
            for di, dj in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < M:
                    covered[ni][nj] = True

        # Greedy selection: pick cells with best coverage-to-cost ratio
        while not all(covered[i][j] for i in range(N) for j in range(M)):
            best_cell = None
            best_ratio = -1
            best_coverage = 0

            for i in range(N):
                for j in range(M):
                    if (i, j) not in selected:
                        coverage = get_coverage(i, j)
                        if coverage > 0:
                            cost = F[i][j]
                            ratio = coverage / cost
                            # Prefer higher coverage, then lower cost
                            if ratio > best_ratio or (
                                ratio == best_ratio and coverage > best_coverage
                            ):
                                best_ratio = ratio
                                best_coverage = coverage
                                best_cell = (i, j)

            if best_cell is None:
                # Should not happen if puzzle is valid
                raise ValueError("Cannot find a valid dominating set")

            selected.append(best_cell)
            mark_covered(best_cell[0], best_cell[1])

        # Format as 1-based output
        return "\n".join(f"{i + 1} {j + 1}" for i, j in selected)

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._F is None or self._N is None or self._M is None:
            raise RuntimeError("No grid generated")
        return self.prompt_template.format(
            N=self._N,
            M=self._M,
            F="\n".join(
                " ".join(f"F[{i}][{j}]={Fij}" for j, Fij in enumerate(Fi, start=1))
                for i, Fi in enumerate(self._F, start=1)
            ),
        )

    def _process(self, answer: str | None) -> list[tuple[int, int]] | None:
        """Process the answer string into a list of cell coordinates."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            cells = []
            for line in answer.splitlines():
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) != 2:
                        return None
                    i, j = int(parts[0]), int(parts[1])
                    cells.append((i, j))
            return cells
        except (ValueError, IndexError):
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the submitted answer.

        Returns:
            -1.0: wrong format
            -0.5: invalid solution (out of bounds or duplicate)
            -0.2: unsuccessful solution (not all cells covered)
            (gold/answer)^5: quality score based on cost ratio
        """
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        selected = [[False] * self._M for _ in range(self._N)]
        for i, j in processed_result:
            if not (1 <= i <= self._N and 1 <= j <= self._M):
                return 0.0
            if selected[i - 1][j - 1]:
                return 0.0
            selected[i - 1][j - 1] = True

        # Check coverage: every cell must be selected or adjacent to a selected cell
        dxs = [0, 0, 0, -1, +1]
        dys = [0, -1, +1, 0, 0]
        for i in range(self._N):
            for j in range(self._M):
                if not any(
                    0 <= i + dx < self._N
                    and 0 <= j + dy < self._M
                    and selected[i + dx][j + dy]
                    for dx, dy in zip(dxs, dys, strict=False)
                ):
                    return 0.0

        # Calculate cost
        answer_cost = sum(self._F[i - 1][j - 1] for i, j in processed_result)
        gold = self._gold_answer

        if not (0 < gold <= answer_cost):
            # This should not happen with valid solutions
            return 0.0
        return (gold / answer_cost) ** 5

    def render(
        self,
        show_answer: bool = False,
        answer_cells: list[tuple[int, int]] | None = None,
    ) -> Image.Image | list[Image.Image] | None:
        """Render the grid with costs and optionally show a solution.

        Args:
            show_answer: If True, highlight the reference answer cells
            answer_cells: Optional list of (row, col) in 1-based indexing to highlight

        Returns:
            PIL Image of the grid
        """
        if self._F is None or self._N is None or self._M is None:
            raise RuntimeError("No grid generated")

        rows, cols = self._N, self._M
        cell_px = self._cell_px
        padding = self._padding

        width = padding * 2 + cols * cell_px
        height = padding * 2 + rows * cell_px
        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font = ImageFont.truetype(font_path, int(cell_px * 0.35))
            font_small = ImageFont.truetype(font_path, int(cell_px * 0.22))
        else:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Parse answer cells if showing
        selected_cells = set()
        if show_answer or answer_cells is not None:
            if answer_cells is None and self._oracle_answer:
                # Parse reference answer
                for line in self._oracle_answer.strip().split("\n"):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 2:
                            i, j = int(parts[0]), int(parts[1])
                            selected_cells.add((i - 1, j - 1))  # Convert to 0-based
            elif answer_cells:
                for i, j in answer_cells:
                    selected_cells.add((i - 1, j - 1))  # Convert to 0-based

        # Determine which cells are covered by selected cells
        covered_cells = set()
        if selected_cells:
            for r, c in selected_cells:
                # Mark this cell and its orthogonal neighbors as covered
                for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        covered_cells.add((nr, nc))

        # Draw cell backgrounds first (for highlighting)
        for r in range(rows):
            for c in range(cols):
                x1 = padding + c * cell_px
                y1 = padding + r * cell_px
                x2 = x1 + cell_px
                y2 = y1 + cell_px

                if (r, c) in selected_cells:
                    # Selected cells: green background
                    draw.rectangle(
                        [x1 + 1, y1 + 1, x2 - 1, y2 - 1], fill=(180, 240, 180)
                    )
                elif (r, c) in covered_cells:
                    # Covered (dominated) cells: light blue background
                    draw.rectangle(
                        [x1 + 1, y1 + 1, x2 - 1, y2 - 1], fill=(220, 240, 255)
                    )

        # Draw grid lines
        for r in range(rows + 1):
            y = padding + r * cell_px
            draw.line(
                (padding, y, padding + cols * cell_px, y),
                fill=(80, 80, 80),
                width=2,
            )
        for c in range(cols + 1):
            x = padding + c * cell_px
            draw.line(
                (x, padding, x, padding + rows * cell_px),
                fill=(80, 80, 80),
                width=2,
            )

        # Draw cell costs and labels
        for r in range(rows):
            for c in range(cols):
                cost = self._F[r][c]
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2

                # Draw cost in the center
                cost_text = str(cost)
                bbox = draw.textbbox((0, 0), cost_text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                text_color = (20, 100, 20) if (r, c) in selected_cells else (20, 20, 20)
                draw.text(
                    (cx - tw // 2, cy - th // 2),
                    cost_text,
                    fill=text_color,
                    font=font,
                )

                # Draw cell label (i,j) in top-left corner of cell
                label = f"({r + 1},{c + 1})"
                bbox_label = draw.textbbox((0, 0), label, font=font_small)
                label_x = padding + c * cell_px + 3
                label_y = padding + r * cell_px + 3
                draw.text(
                    (label_x, label_y),
                    label,
                    fill=(100, 100, 100),
                    font=font_small,
                )

        return img

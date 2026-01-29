"""Money Charging Game environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEMoneyChargingGameEnv(Env):
    """RLVE Money Charging Game as a single-turn environment.

    Given N nodes, each with associated values A[i][1], A[i][2], A[i][3],
    compute the probability that a random tree structure satisfies all
    ordering constraints during a weighted random selection process.
    Based on: https://www.luogu.com.cn/problem/P5405
    """

    assets_dir = resources.files("gym_v.envs") / "assets"
    MOD = 998244353

    prompt_template = r"""There are {N} nodes, each associated with values A[i][1], A[i][2], and A[i][3]. For each node `i`, define: P[i][j] = A[i][j] / (A[i][1] + A[i][2] + A[i][3]) for j = 1, 2, 3. The values A are given as follows:
{A}

We define the following random process:
1. For each node `i`, randomly assign W[i] = j with probability P[i][j] for j = 1, 2, 3.
2. Starting from an empty set, repeatedly select a node `i` with probability proportional to W[i], and add it to the set (duplicates are allowed). Continue until all nodes are in the set.
3. Let T[i] denote the first time node `i` is added to the set.

You are also given a set of constraints (each of the form T[u] < T[v]) that correspond to the edges of an undirected tree:
{T_inequalities}
Please compute the total probability that all the above T[u] < T[v] conditions hold during the random process. Output the result modulo {MOD}."""

    def __init__(
        self,
        n: int = 5,
        cell_px: int = 60,
        padding: int = 32,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._n = n
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._parameter: dict[str, Any] = {}
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        n = self._parameter.get("N", self._n) if self._parameter else self._n

        return dedent(
            f"""
            Money Charging Game (Tree Probability Problem):

            Problem: Given {n} nodes with weighted probabilities and a tree structure,
            compute the probability that a random selection process satisfies all tree
            ordering constraints.

            In the image:
            - Top section: Node probability bars showing A[i][1], A[i][2], A[i][3] values
              * Each node has 3 bars representing the three probability weights
              * Bar lengths are proportional to the weight values
              * Colors: Green (A[1]), Blue (A[2]), Red (A[3])
            - Middle section: Tree structure visualization
              * Nodes are arranged showing parent-child relationships
              * Arrows indicate the ordering constraints T[u] < T[v]
              * Node labels show the node index
            - Bottom section: Constraint list
              * Shows all T[u] < T[v] constraints that must be satisfied
              * These constraints form an undirected tree structure

            The random process:
            1. Each node i gets weight W[i] ∈ {{1,2,3}} with probability P[i][j] = A[i][j] / sum(A[i])
            2. Nodes are selected with probability proportional to their weights
            3. T[i] is the first time node i is selected
            4. We compute the probability that all T[u] < T[v] constraints hold

            Output format: A single integer (the probability result modulo {self.MOD}).
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
            text=state_text,
            metadata={"text_prompt": self._prompt},
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
        """Generate a money charging game problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        N = self._n
        assert N >= 2, "N should be greater than or equal to 2"

        A = [int(self.np_random.integers(1, N + 1)) for _ in range(N)]
        B = [int(self.np_random.integers(1, N + 1)) for _ in range(N)]
        C = [int(self.np_random.integers(1, N + 1)) for _ in range(N)]

        self._parameter["N"] = N
        self._parameter["A"] = A
        self._parameter["B"] = B
        self._parameter["C"] = C

        T_inequalities = []
        permutation = list(range(N))
        swap_probability = float(self.np_random.random())
        self.np_random.shuffle(permutation)
        for i in range(1, N):
            u = permutation[int(self.np_random.integers(0, i))]
            v = permutation[i]
            if self.np_random.random() < swap_probability:
                u, v = v, u
            T_inequalities.append((u, v))
        self.np_random.shuffle(T_inequalities)

        self._parameter["T_inequalities"] = T_inequalities

        # Compute reference answer using dynamic programming on tree
        S = []
        for a1, a2, a3 in zip(A, B, C, strict=False):
            total = a1 + a2 + a3
            S.append(pow(total, self.MOD - 2, self.MOD))

        # Precompute inverses of 1..3N
        invs = [0] * (3 * N + 1)
        for k in range(1, 3 * N + 1):
            invs[k] = pow(k, self.MOD - 2, self.MOD)

        # Build the tree (0-indexed) with flags
        G = [[] for _ in range(N)]
        for u, v in T_inequalities:
            G[v].append((u, 1))
            G[u].append((v, 0))

        # DP arrays
        f = [None] * N
        size = [0] * N

        def dfs(x: int, parent: int) -> None:
            size[x] = 1
            # fx[k] will hold the unnormalized convolution numerator
            fx = [0] * (3 * size[x] + 1)
            fx[1] = A[x] * S[x] % self.MOD
            fx[2] = B[x] * S[x] % self.MOD * 2 % self.MOD
            fx[3] = C[x] * S[x] % self.MOD * 3 % self.MOD

            # Merge in each child
            for v, t in G[x]:
                if v == parent:
                    continue
                dfs(v, x)
                fy = f[v]

                new_size = size[x] + size[v]
                tmp = [0] * (3 * new_size + 1)

                # Convolution with the "subtract-and-redistribute" if t==1
                for i in range(1, size[x] * 3 + 1):
                    if fx[i] == 0:
                        continue
                    for j in range(1, size[v] * 3 + 1):
                        res = fx[i] * fy[j] % self.MOD
                        if t:
                            tmp[i + j] = (tmp[i + j] - res) % self.MOD
                            tmp[i] = (tmp[i] + res) % self.MOD
                        else:
                            tmp[i + j] = (tmp[i + j] + res) % self.MOD

                size[x] = new_size
                fx = tmp

            # One division pass, after all children are merged
            for k in range(1, size[x] * 3 + 1):
                fx[k] = fx[k] * invs[k] % self.MOD

            f[x] = fx

        # Run and collect answer
        dfs(0, -1)
        self._oracle_answer = sum(f[0][1 : 3 * size[0] + 1]) % self.MOD

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if not self._parameter:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._parameter["N"],
            A="\n".join(
                f"A[{i}][1, 2, 3] = [{a}, {b}, {c}]"
                for i, (a, b, c) in enumerate(
                    zip(
                        self._parameter["A"],
                        self._parameter["B"],
                        self._parameter["C"],
                        strict=False,
                    )
                )
            ),
            T_inequalities="\n".join(
                f"T[{u}] < T[{v}]" for u, v in self._parameter["T_inequalities"]
            ),
            MOD=self.MOD,
        )

    def _process(self, answer: str | None) -> int | None:
        """Process the answer string into an integer."""
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
        """Score the answer based on correctness.

        Returns:
            -1.0: wrong format (not an integer)
            -0.5: wrong range (not in [0, MOD))
            -0.1: wrong answer (valid format and range, but incorrect)
            +1.0: correct answer
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if not (0 <= processed_result < self.MOD):
                return 0.0
            if processed_result == self._oracle_answer:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the money charging game visualization.

        Shows:
        - Node probability bars for each A[i][1], A[i][2], A[i][3]
        - Tree structure with ordering constraints
        - Legend explaining the problem structure
        """
        if not self._parameter:
            raise RuntimeError("No problem generated")

        N = self._parameter["N"]
        A = self._parameter["A"]
        B = self._parameter["B"]
        C = self._parameter["C"]
        T_inequalities = self._parameter["T_inequalities"]

        padding = self._padding
        cell_px = self._cell_px

        # Calculate dimensions
        node_section_height = 120 + N * 35
        tree_section_height = 200
        constraint_section_height = 120
        title_height = 60

        width = padding * 2 + max(600, N * cell_px + 100)
        height = (
            padding * 4
            + title_height
            + node_section_height
            + tree_section_height
            + constraint_section_height
        )

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_large = ImageFont.truetype(font_path, 28)
            font_medium = ImageFont.truetype(font_path, 18)
            font_small = ImageFont.truetype(font_path, 14)
            font_tiny = ImageFont.truetype(font_path, 12)
        else:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_tiny = ImageFont.load_default()

        # Title
        title = "Money Charging Game - Tree Probability Problem"
        title_bbox = draw.textbbox((0, 0), title, font=font_large)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, padding), title, fill=(30, 30, 30), font=font_large)

        y_cursor = padding + title_height

        # Section 1: Node Probabilities
        draw.text(
            (padding, y_cursor),
            "Node Probability Weights:",
            fill=(30, 30, 30),
            font=font_medium,
        )
        y_cursor += 35

        max_val = max(max(A), max(B), max(C))
        bar_width = min(200, (width - padding * 3 - 150) // 3)

        for i in range(N):
            node_y = y_cursor + i * 35
            node_label = f"Node {i}:"
            draw.text(
                (padding + 10, node_y), node_label, fill=(60, 60, 60), font=font_small
            )

            bar_x_start = padding + 100

            # Draw three bars for A[i], B[i], C[i]
            values = [A[i], B[i], C[i]]
            colors = [(100, 200, 100), (100, 150, 255), (255, 100, 100)]
            labels = ["A1", "A2", "A3"]

            for j, (val, color, label) in enumerate(
                zip(values, colors, labels, strict=False)
            ):
                bar_x = bar_x_start + j * (bar_width + 10)
                bar_height = 20
                bar_length = int((val / max_val) * bar_width)

                # Background bar
                draw.rectangle(
                    [bar_x, node_y, bar_x + bar_width, node_y + bar_height],
                    fill=(220, 220, 220),
                    outline=(150, 150, 150),
                )

                # Filled bar
                draw.rectangle(
                    [bar_x, node_y, bar_x + bar_length, node_y + bar_height],
                    fill=color,
                    outline=None,
                )

                # Value label
                val_label = f"{val}"
                draw.text(
                    (bar_x + bar_width + 5, node_y + 2),
                    val_label,
                    fill=(60, 60, 60),
                    font=font_tiny,
                )

        y_cursor += N * 35 + padding

        # Section 2: Tree Structure
        draw.text(
            (padding, y_cursor),
            "Tree Structure (Ordering Constraints):",
            fill=(30, 30, 30),
            font=font_medium,
        )
        y_cursor += 35

        # Build adjacency list for visualization
        adj = [[] for _ in range(N)]
        for u, v in T_inequalities:
            adj[u].append(v)
            adj[v].append(u)

        # Find root (node with most connections, or node 0)
        root = 0
        max_degree = len(adj[0])
        for i in range(N):
            if len(adj[i]) > max_degree:
                max_degree = len(adj[i])
                root = i

        # Layout nodes in a simple tree structure
        node_positions = {}
        visited = [False] * N

        def layout_tree(
            node: int, x: float, y: float, width: float, parent: int = -1
        ) -> None:
            visited[node] = True
            node_positions[node] = (x, y)

            children = [n for n in adj[node] if n != parent and not visited[n]]
            if children:
                child_width = width / len(children)
                for i, child in enumerate(children):
                    child_x = x - width / 2 + child_width * (i + 0.5)
                    layout_tree(child, child_x, y + 60, child_width, node)

        tree_x = width // 2
        tree_y = y_cursor + 20
        layout_tree(root, tree_x, tree_y, width - padding * 4)

        # Draw edges
        for u, v in T_inequalities:
            if u in node_positions and v in node_positions:
                x1, y1 = node_positions[u]
                x2, y2 = node_positions[v]
                draw.line(
                    [(x1, y1), (x2, y2)],
                    fill=(150, 150, 150),
                    width=2,
                )

                # Draw arrow head
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                draw.ellipse(
                    [mid_x - 3, mid_y - 3, mid_x + 3, mid_y + 3],
                    fill=(255, 100, 100),
                )

        # Draw nodes
        for node, (x, y) in node_positions.items():
            node_radius = 18
            draw.ellipse(
                [x - node_radius, y - node_radius, x + node_radius, y + node_radius],
                fill=(200, 220, 255),
                outline=(50, 50, 150),
                width=2,
            )
            node_text = str(node)
            bbox = draw.textbbox((0, 0), node_text, font=font_medium)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (x - tw // 2, y - th // 2),
                node_text,
                fill=(30, 30, 30),
                font=font_medium,
            )

        y_cursor += tree_section_height

        # Section 3: Constraints List
        draw.text(
            (padding, y_cursor),
            "Constraints (must all hold):",
            fill=(30, 30, 30),
            font=font_medium,
        )
        y_cursor += 30

        # Show constraints in columns
        max_per_column = 6
        num_columns = (len(T_inequalities) + max_per_column - 1) // max_per_column
        col_width = (width - padding * 2) // num_columns

        for idx, (u, v) in enumerate(T_inequalities):
            col = idx // max_per_column
            row = idx % max_per_column
            constraint_x = padding + col * col_width + 20
            constraint_y = y_cursor + row * 20
            constraint_text = f"T[{u}] < T[{v}]"
            draw.text(
                (constraint_x, constraint_y),
                constraint_text,
                fill=(80, 80, 80),
                font=font_small,
            )

        return img

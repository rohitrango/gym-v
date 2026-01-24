"""Spy Network environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import networkx as nx
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVESpyNetworkEnv(Env):
    """RLVE Spy Network as a single-turn environment.

    Given a directed graph where vertices have costs, select a minimum-cost
    subset of vertices such that every vertex in the graph is reachable from
    at least one selected vertex. This is a directed vertex cover problem
    solved using strongly connected components.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a **directed graph** with {N} vertices, labeled from 0 to {N_minus_1}.

The graph contains the following directed edges. Each edge is represented as a tuple (s, t), meaning there is a directed edge **from vertex s to vertex t**:
{edges}

Each vertex i has an associated cost c[i], given as follows:
{costs}

Your task is to select a subset of vertices s_1, s_2, ..., s_k such that:
- Every vertex in the graph is reachable (i.e., there exists a path ending at that vertex) starting from at least one of the selected vertices.
- Your goal is to **minimize** the total cost of the selected vertices: c[s_1] + c[s_2] + ... + c[s_k].

**Output Format:**
Your final answer should be a single line containing the selected vertices: s_1, s_2, ..., s_k, separated by **spaces**.
Example: `0 1 {N_minus_1}` (do **NOT** include the backticks or quotes); this means the selected vertices are 0, 1, and {N_minus_1}, and the total cost is c[0] + c[1] + c[{N_minus_1}] = {c_0} + {c_1} + {c_N_minus_1} = {example_cost}."""

    def __init__(
        self,
        max_n: int = 10,
        edge_density: float = 0.3,
        dominated_probability: float = 0.5,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        unsuccessful_solution: float = -0.3,
        rewarding_strategy: str = "(gold/answer)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 3.0,
        padding: int = 40,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._edge_density = edge_density
        self._dominated_probability = dominated_probability
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "unsuccessful_solution": unsuccessful_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        self._n: int | None = None
        self._edges: list[tuple[int, int]] | None = None
        self._costs: list[int] | None = None
        self._prompt: str | None = None
        self._reference_answer: str | None = None
        self._gold_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n:
            size_hint = f"{self._n} vertices"
        else:
            size_hint = "N vertices"

        return dedent(
            f"""
            Spy Network Problem (Directed Vertex Cover):

            Given a directed graph with {size_hint}, where each vertex has an associated cost,
            select a minimum-cost subset of vertices such that every vertex in the graph is
            reachable from at least one selected vertex.

            In the visualization:
            - Nodes are displayed as circles with vertex IDs
            - Node colors represent costs (darker = higher cost)
            - Directed edges show connections between vertices
            - The optimal solution involves finding strongly connected components (SCCs)
            - Select the minimum-cost vertex from each SCC that has no incoming edges from other SCCs

            Graph structure:
            - Directed edges indicate paths from source to target vertices
            - Edge arrows show direction of reachability
            - Costs are labeled on each vertex

            Output format: Space-separated list of vertex IDs representing the selected vertices.
            Example: "0 2 5" means vertices 0, 2, and 5 are selected.
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
        """Generate a spy network problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        N = int(self.np_random.integers(3, self._max_n + 1))

        dominated = [
            self.np_random.random() < self._dominated_probability for _ in range(N)
        ]
        all_edges = [
            (s, t)
            for s in range(N)
            for t in range(N)
            if s != t and (dominated[s] is False or dominated[t] is True)
        ]

        num_edges = min(
            len(all_edges), int(self._edge_density * N * (N - 1))
        )
        edge_indices = self.np_random.choice(
            len(all_edges), size=num_edges, replace=False
        )
        edges = [all_edges[i] for i in edge_indices]
        self.np_random.shuffle(edges)

        costs = [int(self.np_random.integers(1, N + 1)) for _ in range(N)]

        # Build adjacency list
        adj = [[] for _ in range(N)]
        for s, t in edges:
            adj[s].append(t)

        # Tarjan's algorithm for strongly connected components
        scc_id = [0] * N
        pre = [0] * N
        low = [0] * N
        stack = []
        in_stack = [False] * N

        scc_count = 0
        dfs_clock = 0

        def tarjan(u: int) -> None:
            nonlocal dfs_clock, scc_count
            dfs_clock += 1
            pre[u] = dfs_clock
            low[u] = dfs_clock
            stack.append(u)
            in_stack[u] = True

            for v in adj[u]:
                if pre[v] == 0:
                    tarjan(v)
                    low[u] = min(low[u], low[v])
                elif in_stack[v]:
                    low[u] = min(low[u], pre[v])

            if low[u] == pre[u]:
                while True:
                    x = stack.pop()
                    in_stack[x] = False
                    scc_id[x] = scc_count
                    if x == u:
                        break
                scc_count += 1

        for i in range(N):
            if pre[i] == 0:
                tarjan(i)

        # Find SCCs with no incoming edges from other SCCs
        scc_in_degree = [False] * scc_count
        for u in range(N):
            for v in adj[u]:
                if scc_id[u] != scc_id[v]:
                    scc_in_degree[scc_id[v]] = True

        # Find minimum cost vertex in each SCC with no incoming edges
        min_costs = [None] * scc_count
        min_vertices = [None] * scc_count
        for i, _cost in enumerate(costs):
            s_id = scc_id[i]
            if min_costs[s_id] is None or _cost < min_costs[s_id]:
                min_costs[s_id] = _cost
                min_vertices[s_id] = i

        reference_vertices = [
            min_vertices[s] for s in range(scc_count) if not scc_in_degree[s]
        ]
        gold_answer = sum(costs[vertex] for vertex in reference_vertices)

        self._n = N
        self._edges = edges
        self._costs = costs
        self._reference_answer = " ".join(map(str, reference_vertices))
        self._gold_answer = gold_answer

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            N_minus_1=self._n - 1,
            edges="\n".join(f"({s}, {t})" for s, t in self._edges),
            costs="\n".join(f"c[{i}]={self._costs[i]}" for i in range(self._n)),
            c_0=self._costs[0],
            c_1=self._costs[1],
            c_N_minus_1=self._costs[self._n - 1],
            example_cost=self._costs[0] + self._costs[1] + self._costs[self._n - 1],
        )

    def _process(self, answer: str | None) -> list[int] | None:
        """Process the answer string into a list of integers."""
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
        """Score the answer based on correctness and optimality.

        Returns:
            Reward based on the rewarding strategy:
            - wrong_format: if answer cannot be parsed
            - invalid_solution: if vertices are invalid or duplicated
            - unsuccessful_solution: if not all vertices are reachable
            - rewarding_weight * ((gold/answer)^beta): if valid solution
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            selected_vertices = processed_result

            adj = [[] for _ in range(self._n)]
            for s, t in self._edges:
                adj[s].append(t)

            visited = [False] * self._n

            def dfs(vertex: int) -> None:
                if visited[vertex]:
                    return
                visited[vertex] = True
                for neighbor in adj[vertex]:
                    dfs(neighbor)

            if len(selected_vertices) != len(set(selected_vertices)):
                return self._rewards["invalid_solution"]

            total_cost = 0
            for vertex in selected_vertices:
                if not (0 <= vertex < self._n):
                    return self._rewards["invalid_solution"]
                dfs(vertex)
                total_cost += self._costs[vertex]

            if not all(visited):
                return self._rewards["unsuccessful_solution"]

            gold = self._gold_answer
            assert gold <= total_cost

            if self._rewards["rewarding_strategy"] == "(gold/answer)^beta":
                return self._rewards["rewarding_weight"] * (
                    (gold / total_cost) ** self._rewards["rewarding_beta"]
                )
            elif self._rewards["rewarding_strategy"] == "gold=answer":
                return self._rewards["rewarding_weight"] * (gold == total_cost)
            else:
                raise NotImplementedError(
                    f"Unknown rewarding strategy: {self._rewards['rewarding_strategy']}"
                )
        else:
            return self._rewards["wrong_format"]

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the spy network as a beautiful directed graph.

        Shows:
        - Nodes with vertex IDs and costs
        - Directed edges with arrows
        - Color-coded nodes by cost (darker = higher cost)
        - Clear legend explaining the problem
        """
        if self._n is None or self._edges is None:
            raise RuntimeError("No problem generated")

        # Create NetworkX graph for layout
        G = nx.DiGraph()
        G.add_nodes_from(range(self._n))
        G.add_edges_from(self._edges)

        # Use spring layout for nice visualization
        pos = nx.spring_layout(G, seed=42, k=2.0, iterations=50)

        # Normalize positions to fit in our image
        padding = self._padding
        node_radius = 30
        min_x = min(x for x, y in pos.values())
        max_x = max(x for x, y in pos.values())
        min_y = min(y for x, y in pos.values())
        max_y = max(y for x, y in pos.values())

        # Calculate image dimensions
        graph_width = 600
        graph_height = 400
        legend_height = 200
        width = padding * 2 + graph_width
        height = padding * 3 + graph_height + legend_height

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_large = ImageFont.truetype(font_path, 20)
            font_medium = ImageFont.truetype(font_path, 16)
            font_small = ImageFont.truetype(font_path, 14)
            font_title = ImageFont.truetype(font_path, 24)
        else:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_title = ImageFont.load_default()

        # Scale positions to fit in graph area
        scaled_pos = {}
        for node, (x, y) in pos.items():
            scaled_x = padding + node_radius + (x - min_x) / (max_x - min_x) * (
                graph_width - 2 * node_radius
            )
            scaled_y = padding + node_radius + (y - min_y) / (max_y - min_y) * (
                graph_height - 2 * node_radius
            )
            scaled_pos[node] = (int(scaled_x), int(scaled_y))

        # Find max cost for color scaling
        max_cost = max(self._costs)
        min_cost = min(self._costs)
        cost_range = max_cost - min_cost if max_cost > min_cost else 1

        # Draw edges with arrows
        arrow_size = 10
        for s, t in self._edges:
            x1, y1 = scaled_pos[s]
            x2, y2 = scaled_pos[t]

            # Calculate direction
            dx = x2 - x1
            dy = y2 - y1
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist == 0:
                continue

            # Normalize direction
            dx /= dist
            dy /= dist

            # Start and end points (offset from node centers)
            start_x = x1 + dx * node_radius
            start_y = y1 + dy * node_radius
            end_x = x2 - dx * node_radius
            end_y = y2 - dy * node_radius

            # Draw edge line
            draw.line(
                [(start_x, start_y), (end_x, end_y)],
                fill=(80, 80, 80),
                width=2,
            )

            # Draw arrow head
            arrow_angle = 0.5
            left_x = end_x - dx * arrow_size - dy * arrow_size * arrow_angle
            left_y = end_y - dy * arrow_size + dx * arrow_size * arrow_angle
            right_x = end_x - dx * arrow_size + dy * arrow_size * arrow_angle
            right_y = end_y - dy * arrow_size - dx * arrow_size * arrow_angle

            draw.polygon(
                [(end_x, end_y), (left_x, left_y), (right_x, right_y)],
                fill=(80, 80, 80),
            )

        # Draw nodes
        for node in range(self._n):
            x, y = scaled_pos[node]
            cost = self._costs[node]

            # Color based on cost (light blue to dark blue)
            intensity = (cost - min_cost) / cost_range
            red = int(220 - intensity * 160)
            green = int(230 - intensity * 150)
            blue = int(255 - intensity * 100)
            node_color = (red, green, blue)

            # Draw node circle
            draw.ellipse(
                [x - node_radius, y - node_radius, x + node_radius, y + node_radius],
                fill=node_color,
                outline=(30, 30, 30),
                width=3,
            )

            # Draw vertex ID
            text = str(node)
            bbox = draw.textbbox((0, 0), text, font=font_large)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            text_color = (255, 255, 255) if intensity > 0.5 else (30, 30, 30)
            draw.text((x - tw // 2, y - th // 2), text, fill=text_color, font=font_large)

            # Draw cost below node
            cost_text = f"c={cost}"
            bbox = draw.textbbox((0, 0), cost_text, font=font_small)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle(
                [x - tw // 2 - 3, y + node_radius + 2, x + tw // 2 + 3, y + node_radius + th + 6],
                fill=(255, 255, 255),
                outline=(30, 30, 30),
                width=1,
            )
            draw.text(
                (x - tw // 2, y + node_radius + 4),
                cost_text,
                fill=(30, 30, 30),
                font=font_small,
            )

        # Draw legend
        legend_y = padding * 2 + graph_height

        # Title
        title = "Spy Network - Directed Vertex Cover Problem"
        bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = bbox[2] - bbox[0]
        draw.text(
            ((width - title_width) // 2, legend_y),
            title,
            fill=(30, 30, 30),
            font=font_title,
        )
        legend_y += 40

        # Problem description
        desc_lines = [
            "Goal: Select minimum-cost vertices to reach all vertices via directed paths",
            f"Graph: {self._n} vertices, {len(self._edges)} directed edges",
            "Node colors: Darker blue = higher cost",
            "Arrows show direction: can reach from source to target",
        ]

        for line in desc_lines:
            draw.text((padding, legend_y), line, fill=(60, 60, 60), font=font_small)
            legend_y += 22

        # Cost legend
        legend_y += 10
        draw.text(
            (padding, legend_y),
            "Cost Range:",
            fill=(30, 30, 30),
            font=font_medium,
        )
        legend_y += 25

        # Draw color gradient
        gradient_width = 200
        gradient_height = 20
        gradient_x = padding + 20
        for i in range(gradient_width):
            intensity = i / gradient_width
            red = int(220 - intensity * 160)
            green = int(230 - intensity * 150)
            blue = int(255 - intensity * 100)
            color = (red, green, blue)
            draw.line(
                [(gradient_x + i, legend_y), (gradient_x + i, legend_y + gradient_height)],
                fill=color,
                width=1,
            )

        draw.rectangle(
            [gradient_x, legend_y, gradient_x + gradient_width, legend_y + gradient_height],
            outline=(30, 30, 30),
            width=2,
        )

        # Labels for gradient
        draw.text(
            (gradient_x - 5, legend_y + gradient_height + 5),
            f"Low ({min_cost})",
            fill=(60, 60, 60),
            font=font_small,
        )
        draw.text(
            (gradient_x + gradient_width - 40, legend_y + gradient_height + 5),
            f"High ({max_cost})",
            fill=(60, 60, 60),
            font=font_small,
        )

        return img

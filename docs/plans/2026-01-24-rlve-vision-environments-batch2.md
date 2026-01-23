# RLVE Vision-Friendly Environments - Batch 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Implement the second batch of high-priority vision-friendly RLVE environments (9 graph structures + 10 tree structures + 2 board games + 4 coloring = 25 environments)

**Architecture:**
- Each environment is self-contained in `gym_v/envs/rlve/<env_name>.py`
- Reuse RLVE original `_generate()` and `_score_answer()` logic
- Implement new `render()` function for beautiful graph/tree visualization
- Adjust `description` property to match visual rendering
- Follow existing patterns from `graph_isomorphism.py`, `tree_coloring.py`, `maximum_clique.py`

**Tech Stack:**
- Python 3.11+, PIL/Pillow for rendering
- NetworkX for graph algorithms and layouts
- matplotlib for graph/tree visualization
- NumPy for random generation
- Type hints with modern Python syntax
- Google Python Style Guide

**Strategy:** Sequential implementation by category
- Prioritize by category (graph → tree → board → coloring)
- Group by category for code reuse
- One environment at a time with full testing before moving to next

---

## Batch 2 Overview (25 Environments - ⭐ High Priority)

### Phase 1: Graph Structure Visualization (9 environments)
1. hamiltonian_path
2. hamiltonian_path_existence
3. graph_contain_tree_counting
4. longest_path
5. minimum_spanning_tree_counting
6. minimum_directed_spanning_tree
7. minimum_weighted_spanning_tree
8. mixed_graph_eulerian_circuit
9. maximum_weight_matching

### Phase 2: Tree Structure Visualization (10 environments)
10. tree_center
11. tree_add_one_edge_diameter
12. tree_change_one_edge_diameter
13. tree_distance_equal_triad_counting
14. tree_even_partitioning
15. tree_topological_sequence_counting
16. maximum_independent_set_tree
17. binary_tree_leaf_num_expectation
18. fbi_binary_tree
19. weighted_binarytree

### Phase 3: Board & Placement Problems (2 environments)
20. knights_and_knaves
21. whack_a_mole

### Phase 4: Coloring & Pattern (4 environments)
22. coloring_counting
23. card_coloring_counting
24. maximum_achromatic_number
25. minimum_chromatic_number

---

## Implementation Pattern (Template for Each Environment)

For each environment, follow this exact sequence:

### Task Structure

**Files:**
- Read RLVE source: `/Users/moonshot/Desktop/project/RLVE/Gym/environments/<env_name>/environment.py`
- Create: `gym_v/envs/rlve/<env_name>.py`
- Modify: `gym_v/envs/rlve/__init__.py` (add import)
- Test: Run `pytest tests/test_single_turn_envs.py -k RLVE/<EnvName> -v`

**Step 1: Read and understand RLVE original implementation**
- Study the problem definition, generation algorithm, and scoring logic
- Identify key data structures (graph, edges, weights, tree structure, etc.)
- Note the expected input/output format

**Step 2: Create environment file with class structure**
```python
"""<Environment name> environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import networkx as nx
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVE<EnvName>Env(Env):
    """RLVE <Environment name> as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        max_nodes: int = 10,  # Adjust based on env
        image_width: int = 800,
        image_height: int = 600,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        # Initialize parameters
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}
        self._max_nodes = max_nodes
        self._image_width = image_width
        self._image_height = image_height

        # Environment-specific state
        self._graph: nx.Graph | nx.DiGraph | None = None
        self._prompt: str | None = None
        self._reference_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        """Return description adapted for visual input."""
        return dedent(
            f"""
            [Environment Name] rules:
            1) [Rule 1]
            2) [Rule 2]
            ...

            In the image:
            - [Visual element 1 description]
            - [Visual element 2 description]
            ...

            Output format: [Exact format specification]
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
        info = {"reference_answer": self._reference_answer}

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
        """Generate problem instance - PORT FROM RLVE."""
        # TODO: Port from RLVE/Gym/environments/<env_name>/environment.py
        pass

    def _prompt_generate(self) -> str:
        """Generate text prompt - PORT FROM RLVE."""
        # TODO: Port from RLVE, but may need adjustment
        pass

    def _process(self, answer: str | None) -> Any:
        """Process answer string - PORT FROM RLVE."""
        # TODO: Port from RLVE
        pass

    def _score_answer(self, answer: str) -> float:
        """Score answer - PORT FROM RLVE."""
        # TODO: Port from RLVE
        pass

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render graph/tree as beautiful, clear image - WRITE NEW."""
        # TODO: Implement visual rendering with NetworkX
        pass
```

**Step 3: Port _generate() from RLVE**
- Copy generation logic from RLVE source
- Adapt to use self.np_random instead of random/np.random
- Keep algorithm logic identical
- Store generated graph/tree in instance variables (use NetworkX)

**Step 4: Port _score_answer() and _process() from RLVE**
- Copy scoring logic exactly
- May need minor adaptations for structure
- Ensure scoring matches RLVE behavior

**Step 5: Implement render() function**
- Design visual representation that clearly shows graph/tree structure
- Use NetworkX for graph layout (spring_layout, circular_layout, kamada_kawai_layout, etc.)
- Use PIL/ImageDraw for rendering
- Follow visual style of existing environments:
  - Clear node circles with labels
  - Edges as lines (directed edges with arrows)
  - Edge weights labeled if applicable
  - Node colors for special properties
  - Legend if using colors
  - Use helper functions for common patterns

**Step 6: Write/adjust description property**
- Explain rules clearly
- Describe what's shown in the image (nodes, edges, colors, etc.)
- Specify exact output format expected
- Reference existing descriptions as templates

**Step 7: Update __init__.py**
```python
from gym_v.envs.rlve.<env_name> import RLVE<EnvName>Env
```

**Step 8: Test with test_single_turn_envs.py**
```bash
cd /Users/moonshot/Desktop/project/gym-v
pytest tests/test_single_turn_envs.py -k "RLVE/<EnvName>" -v
```
Expected: PASS with correct reward for oracle answer

**Step 9: Visual verification**
- Check saved images in test_output_rlve_<env_name>/
- Verify multiple seeds produce different valid graphs/trees
- Ensure rendering is clear, beautiful, and matches description

**Step 10: Commit**
```bash
git add gym_v/envs/rlve/<env_name>.py gym_v/envs/rlve/__init__.py
git commit -m "feat(rlve): add <environment name> environment

- Port generation and scoring logic from RLVE
- Implement visual rendering with NetworkX graph layout
- Add comprehensive description for vision input
- Test passes with multiple seeds
"
```

---

## Environment Registration

After implementing environments, they must be registered in `gym_v/envs/__init__.py`:

```python
# RLVE environments
register(
    id="RLVE/<EnvName>-v0",
    entry_point="gym_v.envs.rlve:<EnvName>Env",
    kwargs={},
)
```

And added to test registry in `tests/test_single_turn_envs.py`:

```python
RLVE_ENVS = {
    # ... existing entries ...
    "RLVE/<EnvName>-v0": "rlve_<env_name>",
}
```

---

## Testing Strategy

### Per-Environment Testing
```bash
# Test specific environment
pytest tests/test_single_turn_envs.py -k "RLVE/<EnvName>" -v

# Check visual outputs
ls tests/test_output_rlve_<env_name>/
open tests/test_output_rlve_<env_name>/0_reset.png
```

### Batch Testing
```bash
# Test all RLVE environments
pytest tests/test_single_turn_envs.py -k "RLVE" -v

# Test with multiple seeds for robustness
pytest tests/test_single_turn_envs.py -k "RLVE" -v --count=5
```

### Visual Quality Checklist
For each environment's rendered image:
- [ ] Graph/tree structure is clearly visible
- [ ] Nodes are labeled and distinguishable
- [ ] Edges are clear (with arrows for directed graphs)
- [ ] Edge weights are visible if applicable
- [ ] Special nodes/edges are highlighted appropriately
- [ ] Layout is balanced and readable
- [ ] Matches description of visual elements
- [ ] Works correctly across different graph sizes/seeds

---

## Coding Standards Checklist

For each implementation:
- [ ] Google Python Style Guide compliance
- [ ] Type hints on all function signatures
- [ ] Docstrings for class and complex methods
- [ ] No forward compatibility code (YAGNI)
- [ ] Uses `self.np_random` for randomization
- [ ] Proper error handling for edge cases
- [ ] Clean imports (alphabetized, grouped)
- [ ] No unused variables or dead code
- [ ] Consistent naming with existing environments

---

## Common Patterns Reference

### Graph Rendering Pattern (Undirected)
```python
def render(self) -> Image.Image:
    """Render undirected graph with spring layout."""
    G = self._graph

    # Create image
    img = Image.new("RGB", (self._image_width, self._image_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Load font
    font_path = self.assets_dir / "DejaVuSans.ttf"
    if font_path.exists():
        font = ImageFont.truetype(str(font_path), 20)
        small_font = ImageFont.truetype(str(font_path), 14)
    else:
        font = ImageFont.load_default()
        small_font = font

    # Compute layout with fixed seed for consistency
    pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(len(G.nodes())))

    # Scale positions to image coordinates with margins
    margin = 80
    width = self._image_width - 2 * margin
    height = self._image_height - 2 * margin

    scaled_pos = {}
    for node, (x, y) in pos.items():
        scaled_pos[node] = (
            margin + (x + 0.5) * width,
            margin + (0.5 - y) * height  # Flip y-axis
        )

    # Draw edges
    for u, v in G.edges():
        x1, y1 = scaled_pos[u]
        x2, y2 = scaled_pos[v]
        draw.line([(x1, y1), (x2, y2)], fill=(100, 100, 100), width=2)

        # Draw edge weight if exists
        if G[u][v].get('weight'):
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            weight_text = str(G[u][v]['weight'])
            bbox = draw.textbbox((0, 0), weight_text, font=small_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            # Draw white background for weight text
            draw.rectangle(
                [mid_x - tw/2 - 2, mid_y - th/2 - 2,
                 mid_x + tw/2 + 2, mid_y + th/2 + 2],
                fill=(255, 255, 255), outline=(150, 150, 150)
            )
            draw.text((mid_x - tw/2, mid_y - th/2), weight_text,
                     fill=(0, 0, 0), font=small_font)

    # Draw nodes
    node_radius = 20
    for node in G.nodes():
        x, y = scaled_pos[node]
        # Draw node circle
        draw.ellipse(
            [x - node_radius, y - node_radius,
             x + node_radius, y + node_radius],
            fill=(135, 206, 250),  # Light blue
            outline=(0, 0, 0),
            width=2
        )
        # Draw node label
        label = str(node)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x - tw/2, y - th/2), label, fill=(0, 0, 0), font=font)

    return img
```

### Directed Graph Rendering Pattern
```python
def render(self) -> Image.Image:
    """Render directed graph with arrows."""
    # Similar to above but with arrow drawing

    # Draw directed edges with arrows
    for u, v in G.edges():
        x1, y1 = scaled_pos[u]
        x2, y2 = scaled_pos[v]

        # Calculate arrow endpoint (stop at node boundary)
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx, dy = dx / length, dy / length
            arrow_end_x = x2 - dx * (node_radius + 5)
            arrow_end_y = y2 - dy * (node_radius + 5)

            # Draw edge line
            draw.line([(x1, y1), (arrow_end_x, arrow_end_y)],
                     fill=(100, 100, 100), width=2)

            # Draw arrowhead
            arrow_size = 10
            angle = np.arctan2(dy, dx)
            arrow_left = (
                arrow_end_x - arrow_size * np.cos(angle - np.pi/6),
                arrow_end_y - arrow_size * np.sin(angle - np.pi/6)
            )
            arrow_right = (
                arrow_end_x - arrow_size * np.cos(angle + np.pi/6),
                arrow_end_y - arrow_size * np.sin(angle + np.pi/6)
            )
            draw.polygon(
                [(arrow_end_x, arrow_end_y), arrow_left, arrow_right],
                fill=(100, 100, 100)
            )
```

### Tree Rendering Pattern (Hierarchical Layout)
```python
def render(self) -> Image.Image:
    """Render tree with hierarchical layout."""
    # Use graphviz_layout for tree-like structure
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(self._graph, prog='dot')
    except:
        # Fallback to spring layout
        pos = nx.spring_layout(self._graph, seed=42)

    # Rest similar to graph rendering
    # May want to highlight root node, show levels with different colors
```

### Binary Tree Rendering Pattern
```python
def render(self) -> Image.Image:
    """Render binary tree with level-order layout."""
    # Custom positioning for binary trees
    # Level 0: 1 node at center
    # Level 1: 2 nodes evenly spaced
    # Level 2: 4 nodes evenly spaced
    # etc.

    def get_tree_positions(root, width, height):
        positions = {}
        levels = {}  # node -> level

        # BFS to assign levels
        from collections import deque
        queue = deque([(root, 0)])
        max_level = 0
        while queue:
            node, level = queue.popleft()
            levels[node] = level
            max_level = max(max_level, level)
            for child in tree.successors(node):
                queue.append((child, level + 1))

        # Assign positions based on level
        level_counts = {}
        level_indices = {}
        for node, level in levels.items():
            level_counts[level] = level_counts.get(level, 0) + 1

        for node in sorted(levels.keys(), key=lambda n: (levels[n], n)):
            level = levels[node]
            idx = level_indices.get(level, 0)
            level_indices[level] = idx + 1

            x = width * (idx + 1) / (level_counts[level] + 1)
            y = height * (level + 1) / (max_level + 2)
            positions[node] = (x, y)

        return positions
```

---

## Phase-Specific Guidelines

### Phase 1: Graph Structures (9 environments)

**Common patterns:**
- Use NetworkX Graph or DiGraph
- Spring layout or Kamada-Kawai for general graphs
- Highlight special paths/edges/nodes based on problem
- Edge weights for weighted problems
- Color coding for different edge types (mixed graphs)

**Key considerations:**
- Hamiltonian paths: highlight the path in a different color
- Spanning trees: highlight tree edges vs non-tree edges
- Matching: highlight matched edges
- Eulerian circuits: show edge directions clearly

### Phase 2: Tree Structures (10 environments)

**Common patterns:**
- Use NetworkX DiGraph (trees are directed from root)
- Hierarchical layout (graphviz_layout with 'dot' program)
- Root node highlighted differently
- Level-based coloring for some problems

**Key considerations:**
- Tree center: highlight center node(s)
- Diameter: highlight longest path
- Distance problems: show distances as edge labels or node colors
- Binary trees: use custom binary tree layout
- Weighted trees: show weights on edges

### Phase 3: Board & Placement (2 environments)

**Common patterns:**
- Grid-based layout with PIL
- Cell colors for different states
- Symbols or icons for game pieces

**Key considerations:**
- Knights and knaves: show positions and identities
- Whack-a-mole: show grid, hammer coverage range

### Phase 4: Coloring & Pattern (4 environments)

**Common patterns:**
- Graph visualization with node/edge colors
- Legend showing color meanings
- Clear color palette

**Key considerations:**
- Node coloring: different colors for different color classes
- Edge coloring: different edge colors
- Chromatic number: minimal color palette visualization

---

## Execution Strategy

**Option 1: Subagent-Driven Development (Recommended)**
- Stay in current session
- Spawn 3-4 subagents per phase in parallel
- Each subagent implements 1-3 environments
- Review each phase before proceeding

**Option 2: Sequential with Code Review**
- Implement one environment at a time
- Full testing and visual verification
- Code review after each implementation

**Which approach do you prefer?**

---

## Success Criteria

After completing Batch 2:

1. **All 25 environments implemented** with:
   - [ ] Generation logic ported from RLVE
   - [ ] Scoring logic ported from RLVE
   - [ ] Beautiful NetworkX-based rendering
   - [ ] Vision-appropriate descriptions
   - [ ] Passing tests with oracle answers

2. **Code quality**:
   - [ ] Google Python Style Guide compliance
   - [ ] Consistent with existing RLVE environments
   - [ ] No forward compatibility hacks

3. **Visual quality**:
   - [ ] Clear graph/tree structure
   - [ ] Readable labels and weights
   - [ ] Appropriate colors and highlights
   - [ ] Works across different seeds

4. **Documentation**:
   - [ ] All environments registered
   - [ ] Test registry updated
   - [ ] Commit messages clear and descriptive

---

## Next Steps After Batch 2

After completing all 25 environments in Batch 2:

1. **Code Review**: Review all implementations for consistency
2. **Performance Test**: Run full test suite to ensure no regressions
3. **Documentation**: Update main README with new environment count
4. **Commit Batch**: Create summary commit for the entire batch
5. **Move to Batch 3**: Proceed with remaining environments (matrices, spatial, games, etc.)

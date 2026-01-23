# RLVE Vision-Friendly Environments - Batch 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Implement the first batch of high-priority vision-friendly RLVE environments (8 grid puzzles + 9 grid graph + 7 geometry + 3 blocks = 27 environments)

**Architecture:**
- Each environment is self-contained in `gym_v/envs/rlve/<env_name>.py`
- Reuse RLVE original `_generate()` and `_score_answer()` logic
- Implement new `render()` function for beautiful visual representation
- Adjust `description` property to match visual rendering
- Follow existing patterns from `hitori_puzzle.py`, `light_up_puzzle.py`, `skyscraper_puzzle.py`

**Tech Stack:**
- Python 3.11+, PIL/Pillow for rendering
- NumPy for random generation
- Type hints with modern Python syntax
- Google Python Style Guide

**Strategy:** A + C combined approach
- Prioritize by urgency (🔥 highest first)
- Group by category for code reuse
- One environment at a time with full testing before moving to next

---

## Batch 1 Overview (27 Environments - 🔥 Highest Priority)

### Phase 1: Grid Puzzles (8 environments)
1. numbrix
2. magic_square_puzzle
3. campsite_puzzle
4. eight_digit_puzzle
5. nine_puzzle
6. twiddle_puzzle
7. skyscraper_sum_puzzle
8. binario_no_adjacency_requirement

### Phase 2: Grid Graph & Path (9 environments)
9. grid_bfs
10. grid_component
11. grid_coloring_counting
12. grid_local_minimum_counting
13. grid_parity_construction
14. grid_triangle_counting
15. circulating_grid
16. minimum_dominating_set_grid
17. maximum_independent_set_grid

### Phase 3: Geometry (7 environments)
18. convex_hull
19. largest_convex_polygon
20. largest_rectangle_among_points
21. smallest_circle
22. sum_triangle_area
23. sum_manhattan_curved_surface
24. landform_generation_counting

### Phase 4: Block & Image (3 environments)
25. block_image
26. klo_blocks
27. monochrome_block_counting

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
- Identify key data structures (grid, graph, points, etc.)
- Note the expected input/output format

**Step 2: Create environment file with class structure**
```python
"""<Environment name> environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVE<EnvName>Env(Env):
    """RLVE <Environment name> as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        max_size: int = 5,  # Adjust based on env
        cell_px: int = 56,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        # Initialize parameters
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Environment-specific state
        self._puzzle_data: Any = None
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
        """Generate puzzle instance - PORT FROM RLVE."""
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
        """Render puzzle as beautiful, clear image - WRITE NEW."""
        # TODO: Implement visual rendering
        pass
```

**Step 3: Port _generate() from RLVE**
- Copy generation logic from RLVE source
- Adapt to use self.np_random instead of random/np.random
- Keep algorithm logic identical
- Store generated puzzle in instance variables

**Step 4: Port _score_answer() and _process() from RLVE**
- Copy scoring logic exactly
- May need minor adaptations for structure
- Ensure scoring matches RLVE behavior

**Step 5: Implement render() function**
- Design visual representation that clearly shows puzzle state
- Use PIL/ImageDraw for clean, professional rendering
- Follow visual style of existing environments:
  - Clean grid lines (2px width, dark color)
  - Clear text with DejaVuSans.ttf font
  - Appropriate colors for different cell types
  - Padding and sizing consistent with other envs
  - Use helper functions for common patterns (grid drawing, text centering)

**Step 6: Write/adjust description property**
- Explain rules clearly
- Describe what's shown in the image
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
- Verify multiple seeds produce different valid puzzles
- Ensure rendering is clear, beautiful, and matches description

**Step 10: Commit**
```bash
git add gym_v/envs/rlve/<env_name>.py gym_v/envs/rlve/__init__.py
git commit -m "feat(rlve): add <environment name> environment

- Port generation and scoring logic from RLVE
- Implement visual rendering with clear grid/graph/geometry display
- Add comprehensive description for vision input
- Test passes with multiple seeds
"
```

---

## Environment Registration

After implementing environments, they must be registered in `gym_v/envs/registration.py`:

```python
# RLVE environments
register(
    id="RLVE/<EnvName>-v0",
    entry_point="gym_v.envs.rlve:<EnvName>Env",
    kwargs={},
)
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
- [ ] Grid/structure is clearly visible
- [ ] Text is readable and well-positioned
- [ ] Colors distinguish different element types
- [ ] Layout is balanced and professional
- [ ] Matches description of visual elements
- [ ] Works correctly across different puzzle sizes/seeds

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

### Grid Rendering Pattern
```python
def render(self) -> Image.Image:
    rows, cols = len(self._grid), len(self._grid[0])
    cell_px = self._cell_px
    padding = self._padding

    width = padding * 2 + cols * cell_px
    height = padding * 2 + rows * cell_px
    img = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)

    # Load font
    font_path = self.assets_dir / "DejaVuSans.ttf"
    if font_path.exists():
        font = ImageFont.truetype(str(font_path), int(cell_px * 0.45))
    else:
        font = ImageFont.load_default()

    # Draw grid lines
    for r in range(rows + 1):
        y = padding + r * cell_px
        draw.line((padding, y, padding + cols * cell_px, y),
                  fill=(30, 30, 30), width=2)
    for c in range(cols + 1):
        x = padding + c * cell_px
        draw.line((x, padding, x, padding + rows * cell_px),
                  fill=(30, 30, 30), width=2)

    # Draw cell contents
    for r in range(rows):
        for c in range(cols):
            val = str(self._grid[r][c])
            bbox = draw.textbbox((0, 0), val, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx = padding + c * cell_px + cell_px // 2
            cy = padding + r * cell_px + cell_px // 2
            draw.text((cx - tw // 2, cy - th // 2), val,
                     fill=(10, 10, 10), font=font)

    return img
```

### Graph Rendering Pattern
```python
def render(self) -> Image.Image:
    # Use spring layout or circular layout for node positions
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from(self._edges)
    pos = nx.spring_layout(G, seed=42)

    # Scale positions to image coordinates
    # Draw edges as lines
    # Draw nodes as circles
    # Add labels
```

### Geometry Rendering Pattern
```python
def render(self) -> Image.Image:
    # Determine bounding box of all points
    # Scale to fit image with padding
    # Draw points as circles
    # Draw lines/polygons with PIL.ImageDraw.polygon()
    # Add coordinate axes if helpful
```

---

## Next Steps After Batch 1

After completing all 27 environments in Batch 1:

1. **Code Review**: Review all implementations for consistency
2. **Performance Test**: Run full test suite to ensure no regressions
3. **Documentation**: Update main README with new environment count
4. **Commit Batch**: Create summary commit for the entire batch
5. **Move to Batch 2**: Proceed with next 25 environments (graph/tree structures)

---

## Execution Options

**Option 1: Subagent-Driven Development (Recommended)**
- Stay in current session
- Spawn fresh subagent per environment
- Review each implementation before proceeding
- Fast iteration with quality checks

**Option 2: Parallel Execution**
- Use separate Claude session
- Batch execute with checkpoints
- Faster but less oversight

**Which approach do you prefer?**

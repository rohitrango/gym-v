# RLVE Vision-Friendly Environments - Batch 3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Implement the final batch of vision-friendly RLVE environments (5 matrix + 3 spatial + 5 sequence + 6 game + 5 other = 24 environments)

**Architecture:**
- Each environment is self-contained in `gym_v/envs/rlve/<env_name>.py`
- Reuse RLVE original `_generate()` and `_score_answer()` logic
- Implement new `render()` function for beautiful visual representation
- Adjust `description` property to match visual rendering
- Follow existing patterns from implemented environments

**Tech Stack:**
- Python 3.11+, PIL/Pillow for rendering
- NumPy for matrices and arrays
- NetworkX for graph-based problems
- Type hints with modern Python syntax
- Google Python Style Guide

**Strategy:** Sequential implementation by category
- Implement one environment at a time
- Test thoroughly before moving to next
- Ensure visual quality and clarity

---

## Batch 3 Overview (24 Environments - 💡 Medium Priority)

### Phase 1: Matrix & 2D Array (5 environments)
1. matrix_pooling
2. addition_table
3. matrix_rmq_counting
4. matrix_permutation_main_diagonal_one
5. matrix_permutation_both_diagonal_one

### Phase 2: Spatial Layout & Arrangement (3 environments)
6. pipeline_arrangement
7. roundtable_assignment
8. warehouse_construction

### Phase 3: Sequence & Permutation Visualization (5 environments)
9. tetris_attack
10. jug_puzzle
11. quantum_lock_puzzle
12. face_right_way
13. preorder_traversal

### Phase 4: Games & Interactive (6 environments)
14. stone_game
15. stone_intervals_game
16. coin_square_game
17. gra_minima_game
18. new_nim_game
19. money_charging_game

### Phase 5: Other Visualization Problems (5 environments)
20. ska_rock_garden
21. spy_network
22. patrol
23. visible_line
24. max_grid_path_intersection

---

## Implementation Pattern

For each environment:

1. **Read RLVE source**: `/Users/moonshot/Desktop/project/RLVE/Gym/environments/<env_name>/environment.py`
2. **Create environment file**: `gym_v/envs/rlve/<env_name>.py`
3. **Port logic from RLVE**:
   - `_generate()` using `self.np_random`
   - `_score_answer()` and `_process()`
   - Keep algorithms identical
4. **Implement beautiful render()**:
   - Clear visual representation
   - Appropriate for problem type
   - PIL/ImageDraw based
5. **Write vision-appropriate description**:
   - Reference `/Users/moonshot/Desktop/project/gym-v/gym_v/envs/reasongym` for format
   - Explain what's shown in image
   - Specify exact output format
6. **Update registrations**:
   - `gym_v/envs/rlve/__init__.py`
   - `gym_v/envs/__init__.py`
   - `tests/test_single_turn_envs.py`
7. **Test**: `pytest tests/test_single_turn_envs.py -k RLVE/<EnvName> -v`
8. **Visual verification**: Check saved images for quality
9. **Commit**: Individual commit per environment

---

## Phase-Specific Guidelines

### Phase 1: Matrix & 2D Array

**Visual patterns:**
- Heatmaps with color gradients for value intensity
- Grid cells with clear boundaries
- Diagonal lines highlighted where relevant
- Pool windows/regions clearly marked
- Text labels for values when needed

**Key considerations:**
- matrix_pooling: Show pooling windows and operation type
- addition_table: Grid with row/column headers
- matrix_rmq_counting: Heatmap + region highlighting
- matrix_permutation: Highlight diagonal constraints

### Phase 2: Spatial Layout & Arrangement

**Visual patterns:**
- Top-down 2D layouts
- Clear spatial relationships
- Connection lines for flow/paths
- Labeled positions/stations
- Legend for symbols

**Key considerations:**
- pipeline_arrangement: Flow diagram with directional arrows
- roundtable_assignment: Circular seating arrangement
- warehouse_construction: Grid-based floor plan

### Phase 3: Sequence & Permutation Visualization

**Visual patterns:**
- State diagrams or sequences
- Container/vessel representations
- Arrows for operations/transformations
- Before/after states
- Tree structures for traversal

**Key considerations:**
- tetris_attack: Block grid with colors
- jug_puzzle: Cylinder representations with fill levels
- quantum_lock_puzzle: State machine diagram
- face_right_way: Arrow directions on a line
- preorder_traversal: Tree with traversal order

### Phase 4: Games & Interactive

**Visual patterns:**
- Game state visualization
- Piles/heaps of items
- Board positions
- Score/counter displays
- Turn indicators where relevant

**Key considerations:**
- stone_game: Vertical piles with stone counts
- stone_intervals_game: Number line with intervals
- coin_square_game: Grid with coin markers
- gra_minima_game: State visualization
- new_nim_game: Multiple piles with counts
- money_charging_game: Progress bars or meters

### Phase 5: Other Visualization Problems

**Visual patterns:**
- Problem-specific diagrams
- Network/graph structures
- Geometric layouts
- Path visualizations
- Spatial relationships

**Key considerations:**
- ska_rock_garden: 2D layout with rocks
- spy_network: Network graph with connections
- patrol: Path on a map/graph
- visible_line: Geometric visibility diagram
- max_grid_path_intersection: Grid with multiple paths

---

## Testing Strategy

```bash
# Test specific environment
pytest tests/test_single_turn_envs.py -k "RLVE/<EnvName>" -v

# Check visual outputs
ls tests/test_output_rlve_<env_name>/
open tests/test_output_rlve_<env_name>/0_reset.png

# Test all new environments
pytest tests/test_single_turn_envs.py -k "RLVE" -v | grep "Phase[1-5]"
```

---

## Coding Standards

- Google Python Style Guide
- Type hints on all functions
- Docstrings for complex methods
- No forward compatibility code
- Use `self.np_random` for all randomization
- Consistent with existing RLVE environments
- Clean, readable code

---

## Execution Strategy

**Subagent-Driven Approach:**
- Launch 3-4 subagents per phase in parallel
- Each subagent implements 1-2 environments
- Full testing before moving to next phase
- Visual quality verification for all outputs

---

## Success Criteria

After completing Batch 3:

- ✅ All 24 environments implemented
- ✅ All tests passing
- ✅ Clear, beautiful visualizations
- ✅ Vision-appropriate descriptions
- ✅ Google style compliance
- ✅ No rendering issues (text truncation, cutoffs, etc.)

---

## Final Environment Count

After Batch 3 completion:
- **Batch 1**: 27 environments
- **Batch 2**: 25 environments
- **Batch 3**: 24 environments
- **Total new**: 76 RLVE environments
- **Total in gym-v**: 82 RLVE environments (6 pre-existing + 76 new)

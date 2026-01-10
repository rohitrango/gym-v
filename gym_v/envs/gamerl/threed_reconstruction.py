"""3D Reconstruction QA Environment

This module implements a 3D voxel reconstruction question-answering environment for the gym-v framework.
Players must reconstruct a 3D structure by adding voxels to match target projections.

The environment supports 6 question types testing understanding of:
- Spatial reasoning (counting, position identification)
- Projection matching
- Action outcomes
- Optimal path finding
- Strategic planning

Source: /mnt/petrelfs/gujiawei/jiawei/env-v/Game-RL/src/3DReconstruction/
"""

from __future__ import annotations

from itertools import combinations
import random
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation

GAME_RULES = dedent("""
    You are in the middle of a 3D reconstruction puzzle.
The current structure has some initial voxels, and your goal is to complete it as the game rules.

Game Rules:
1. Goal: Reconstruct a 3D structure by adding voxels to match given projections.
2. Grid Space: The game is played on a 3x3x3 cube grid.
3. Coordinates: Position (x,y,z) ranges from 1 to 3, with (1,1,1) at front-left-bottom.
4. Position Rule: Each position can contain at most one voxel.
5. Connectivity: All voxels must be connected face-to-face.
6. Voxel Limit: You have a maximum of n additional voxels available.
7. Placement Rule: New voxels can only be placed adjacent to existing ones.
8. Front View (Y-Z): Shows structure when viewed along the negative X-axis direction (front to back), with Y as horizontal axis and Z as vertical axis. Projection coordinates are in (y,z) format.
9. Side View (X-Z): Shows structure when viewed along the positive Y-axis direction (left to right), with X as horizontal axis and Z as vertical axis. Projection coordinates are in (x,z) format.
10. Projection Rule: A cell shows '1' if any voxel exists along that line of sight, and '0' if no voxel exists along that line.
11. Victory: Match both projections using available voxels.
""").strip()


class ThreeDReconstructionGame:
    """Core game logic for 3D voxel reconstruction.

    Manages a 3x3x3 voxel grid where players must reconstruct a target structure
    by adding voxels that maintain connectivity and match target projections.
    """

    def __init__(self, target_voxels_count: int = 7, initial_voxels_count: int = 4):
        """Initialize the 3D reconstruction game.

        Args:
            target_voxels_count: Number of voxels in target structure (1-27)
            initial_voxels_count: Number of initial voxels (< target_voxels_count)
        """
        if not (1 <= target_voxels_count <= 27):
            raise ValueError("Target voxels count must be between 1 and 27")
        if initial_voxels_count >= target_voxels_count:
            raise ValueError(
                "Initial voxels count must be less than target voxels count"
            )

        self.target_count = target_voxels_count
        self.target_voxels = self._generate_connected_structure_with_base(
            target_voxels_count
        )
        self.current_voxels = self._generate_initial_state(
            self.target_voxels, initial_voxels_count
        )
        self.target_yz_projection, self.target_xz_projection = (
            self._calculate_projections(self.target_voxels)
        )
        self.minimal_addition = self._find_minimal_solution()
        self.complete_solution = list(
            set(self.current_voxels) | set(self.minimal_addition)
        )

    def _get_adjacent_neighbors(
        self, voxel: tuple[int, int, int]
    ) -> list[tuple[int, int, int]]:
        """Get face-adjacent neighbors of a voxel within 3x3x3 grid."""
        x, y, z = voxel
        neighbors = []
        directions = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
        for dx, dy, dz in directions:
            neighbor = (x + dx, y + dy, z + dz)
            if all(1 <= coord <= 3 for coord in neighbor):
                neighbors.append(neighbor)
        return neighbors

    def _generate_connected_structure_with_base(
        self, voxel_count: int
    ) -> list[tuple[int, int, int]]:
        """Generate a connected set of voxels with at least one in base layer."""
        base_voxel = random.choice(
            [(x, y, 1) for x in range(1, 4) for y in range(1, 4)]
        )
        structure = {base_voxel}

        while len(structure) < voxel_count:
            all_neighbors = set()
            for voxel in structure:
                all_neighbors.update(self._get_adjacent_neighbors(voxel))
            available_neighbors = all_neighbors - structure
            if not available_neighbors:
                break
            structure.add(random.choice(list(available_neighbors)))

        return list(structure)

    def _generate_initial_state(
        self, target_structure: list, initial_count: int
    ) -> list[tuple[int, int, int]]:
        """Generate initial game state from target structure."""
        base_layer = [v for v in target_structure if v[2] == 1]
        if not base_layer:
            raise ValueError(
                "Target structure must have at least one voxel in base layer"
            )

        initial_state = {random.choice(base_layer)}
        while len(initial_state) < initial_count:
            all_neighbors = set()
            for voxel in initial_state:
                all_neighbors.update(self._get_adjacent_neighbors(voxel))
            valid_neighbors = all_neighbors & set(target_structure) - initial_state
            if not valid_neighbors:
                break
            initial_state.add(random.choice(list(valid_neighbors)))

        return list(initial_state)

    def _calculate_projections(self, voxels: list) -> tuple[np.ndarray, np.ndarray]:
        """Calculate front (Y-Z) and side (X-Z) plane projections."""
        yz_grid = np.zeros((3, 3), dtype=int)
        xz_grid = np.zeros((3, 3), dtype=int)
        for x, y, z in voxels:
            yz_grid[y - 1, z - 1] = 1  # Front view (Y-Z)
            xz_grid[x - 1, z - 1] = 1  # Side view (X-Z)
        return yz_grid, xz_grid

    def _is_connected(self, voxel_set: set) -> bool:
        """Check if a set of voxels forms a connected structure."""
        if not voxel_set:
            return False
        visited = set()
        to_visit = {next(iter(voxel_set))}
        while to_visit:
            current = to_visit.pop()
            visited.add(current)
            neighbors = set(self._get_adjacent_neighbors(current)) & voxel_set
            to_visit.update(neighbors - visited)
        return visited == voxel_set

    def _find_minimal_solution(self) -> list[tuple[int, int, int]]:
        """Find minimal set of voxels to add to satisfy projections."""
        current_set = set(self.current_voxels)
        all_positions = [
            (x, y, z) for x in range(1, 4) for y in range(1, 4) for z in range(1, 4)
        ]
        candidates = [pos for pos in all_positions if pos not in current_set]

        def validates_solution(additional_voxels):
            test_structure = current_set.union(set(additional_voxels))
            if not self._is_connected(test_structure):
                return False
            yz_proj, xz_proj = self._calculate_projections(list(test_structure))
            return np.array_equal(
                yz_proj, self.target_yz_projection
            ) and np.array_equal(xz_proj, self.target_xz_projection)

        def can_connect(voxel, structure):
            return bool(set(self._get_adjacent_neighbors(voxel)) & structure)

        if validates_solution([]):
            return []

        max_additional = len(self.target_voxels) - len(self.current_voxels)
        for size in range(1, max_additional + 1):
            for combination in combinations(candidates, size):
                combination = list(combination)
                test_structure = current_set.copy()

                valid = True
                for voxel in combination:
                    if not can_connect(voxel, test_structure):
                        valid = False
                        break
                    test_structure.add(voxel)

                if not valid:
                    continue

                if validates_solution(combination):
                    return combination

        return [v for v in self.target_voxels if v not in current_set]

    def generate_random_connected_voxels(
        self, count: int, respect_remaining: bool = False
    ) -> tuple[list, list]:
        """Generate random connected voxels based on current state.

        Args:
            count: Number of voxels to generate
            respect_remaining: If True, won't generate more than remaining voxels

        Returns:
            Tuple of (current_voxels, new_voxels)
        """
        if respect_remaining:
            remaining = len(self.target_voxels) - len(self.current_voxels)
            count = min(count, remaining)

        current_structure = set(self.current_voxels)
        new_voxels = []

        while len(new_voxels) < count:
            all_neighbors = set()
            for voxel in current_structure:
                neighbors = set(self._get_adjacent_neighbors(voxel))
                all_neighbors.update(neighbors - current_structure)

            available_positions = all_neighbors - current_structure - set(new_voxels)

            if not available_positions:
                break

            new_voxel = random.choice(list(available_positions))
            new_voxels.append(new_voxel)

        return list(current_structure), new_voxels

    def get_game_state(self) -> dict:
        """Get current game state information."""
        return {
            "current_state": {
                "count": len(self.current_voxels),
                "positions": self.current_voxels,
            },
            "minimal_addition": {
                "count": len(self.minimal_addition),
                "positions": self.minimal_addition,
            },
            "complete_solution": {
                "count": len(self.complete_solution),
                "positions": self.complete_solution,
            },
            "target_count": self.target_count,
        }

    def get_projections(self, structure: list) -> tuple[np.ndarray, np.ndarray]:
        """Calculate Y-Z (front) and X-Z (side) projections for structure."""
        for x, y, z in structure:
            if not (1 <= x <= 3 and 1 <= y <= 3 and 1 <= z <= 3):
                raise ValueError(f"Invalid voxel position: ({x}, {y}, {z})")
        return self._calculate_projections(structure)


class GameRL3DReconstructionQAEnv(Env):
    """3D Reconstruction Question-Answering Environment.

    A single-turn QA environment testing 3D spatial reasoning and projection matching.

    Question Types:
    - Type 0 (Easy): Count voxels in structure (fill-in)
    - Type 1 (Easy): Which position has a voxel? (MCQ 6 options)
    - Type 2 (Medium): Do projections match target? (MCQ 4 options)
    - Type 3 (Medium): Projection after adding voxels (fill-in matrix)
    - Type 4 (Hard): Sequence to match projection (MCQ 8 options)
    - Type 5 (Hard): Minimum voxels needed (fill-in)

    Args:
        plot_level: Structure difficulty ("Easy"/"Medium"/"Hard")
        question_type: Specific question type (0-5), or None for random
    """

    QUESTION_TYPES = [
        {
            "id": "count",
            "name": "Count Voxels",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "StateInfo",
            "options": None,
        },
        {
            "id": "position",
            "name": "Position Check",
            "level": "Easy",
            "answer_format": "single_choice",
            "qa_type": "StateInfo",
            "options": 6,
        },
        {
            "id": "projection",
            "name": "Projection Match",
            "level": "Medium",
            "answer_format": "single_choice",
            "qa_type": "StateInfo",
            "options": 4,
        },
        {
            "id": "action_outcome",
            "name": "Action Outcome",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "ActionOutcome",
            "options": None,
        },
        {
            "id": "transition_path",
            "name": "Transition Path",
            "level": "Hard",
            "answer_format": "single_choice",
            "qa_type": "TransitionPath",
            "options": 8,
        },
        {
            "id": "strategy_optimization",
            "name": "Strategy Optimization",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "StrategyOptimization",
            "options": None,
        },
    ]

    def __init__(
        self, plot_level: str = "Easy", question_type: int | None = None, **kwargs
    ):
        """Initialize 3D Reconstruction QA environment."""
        super().__init__(**kwargs)
        self._plot_level = plot_level
        self._question_type = question_type
        self._game = None
        self._game_state = None
        self._current_question = None

    @property
    def description(self) -> str:
        """Return environment description with game rules."""
        return "3D Reconstruction QA\n\n" + GAME_RULES

    def _get_state_text(self) -> str:
        """Generate text description of current 3D reconstruction state."""
        current_voxels = self._game.current_voxels
        target_count = self._game.target_count
        remaining = target_count - len(current_voxels)

        text = "3D Voxel Reconstruction Puzzle\n"
        text += "Grid Size: 3x3x3\n"
        text += f"Current Voxels: {len(current_voxels)}\n"
        text += f"Remaining Available Voxels: {remaining}\n\n"

        text += "Current Structure Positions:\n"
        for x, y, z in sorted(current_voxels):
            text += f"  Voxel at ({x},{y},{z})\n"

        # Add projection info
        yz_proj, xz_proj = self._game.get_projections(current_voxels)
        text += "\nCurrent Front View (Y-Z) Projection:\n"
        for z in range(2, -1, -1):  # Top to bottom
            row = []
            for y in range(3):
                row.append(str(yz_proj[y, z]))
            text += f"  {' '.join(row)}\n"

        text += "\nCurrent Side View (X-Z) Projection:\n"
        for z in range(2, -1, -1):  # Top to bottom
            row = []
            for x in range(3):
                row.append(str(xz_proj[x, z]))
            text += f"  {' '.join(row)}\n"

        return text.strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset environment and generate a new question."""
        super().reset(seed=seed)

        # Generate game instance based on difficulty
        self._game = self._generate_game_instance(self._plot_level)
        self._game_state = self._game.get_game_state()

        # Generate question
        q_type = (
            self._question_type
            if self._question_type is not None
            else random.randint(0, 5)
        )
        self._current_question = self._generate_question(q_type)

        # Generate text state
        text_state = self._get_state_text()

        obs = Observation(
            image=self.render(),
            text=text_state,
            metadata={
                "question": self._current_question["question"],
            },
        )

        info = {
            "oracle_answer": self._current_question["answer"],
            "question_type": self.QUESTION_TYPES[q_type]["id"],
        }

        return obs, info

    def _generate_game_instance(self, plot_level: str) -> ThreeDReconstructionGame:
        """Generate game instance based on difficulty level."""
        if plot_level == "Easy":
            target_count = random.randint(3, 5)
            initial_count = max(1, target_count - 3)
        elif plot_level == "Medium":
            target_count = random.randint(6, 10)
            initial_count = max(2, target_count - 4)
        elif plot_level == "Hard":
            target_count = random.randint(11, 15)
            initial_count = max(3, target_count - 5)
        else:
            target_count = random.randint(3, 15)
            if target_count <= 5:
                initial_count = max(1, target_count - 3)
            elif target_count <= 10:
                initial_count = max(2, target_count - 4)
            else:
                initial_count = max(3, target_count - 5)

        initial_count = min(initial_count, target_count - 1)
        return ThreeDReconstructionGame(target_count, initial_count)

    def _generate_question(self, q_type: int) -> dict[str, Any]:
        """Generate question of specified type."""
        generators = {
            0: self._generate_count_question,
            1: self._generate_position_question,
            2: self._generate_projection_question,
            3: self._generate_action_outcome_question,
            4: self._generate_transition_path_question,
            5: self._generate_strategy_optimization_question,
        }

        if q_type not in generators:
            raise ValueError(f"Invalid question type: {q_type}")

        return generators[q_type]()

    def _generate_count_question(self) -> dict[str, Any]:
        """Type 0: Count voxels in structure."""
        structure = self._game.current_voxels
        voxel_count = len(structure)
        voxel_positions = sorted(structure)

        question = (
            "This is a state in a 3D reconstruction game.\n\n"
            "Given:\n"
            "- A 3x3x3 grid structure containing voxels\n"
            "- An image containing the voxel structure and its target projections\n"
            "  (Note: The projections shown in the image are not relevant for this question)\n\n"
            "Game Rules:\n"
            "1. Grid Space: The game is played on a 3x3x3 cube grid.\n"
            "2. Coordinates: Position (x,y,z) ranges from 1 to 3, with (1,1,1) at front-left-bottom.\n"
            "3. Position Rule: Each position can contain at most one voxel.\n"
            "4. Connectivity: All voxels must be connected face-to-face.\n\n"
            "Question:\n"
            "How many voxels are there in the given structure?\n\n"
            "Please answer with a number."
        )

        position_list = ", ".join([f"({x},{y},{z})" for x, y, z in voxel_positions])
        analysis = (
            f"The structure contains voxels at the following positions: {position_list}. "
            f"By counting these positions, we can see there are {voxel_count} voxels in total. "
            f"Therefore the answer is {voxel_count}."
        )

        return {"question": question, "answer": str(voxel_count), "analysis": analysis}

    def _generate_position_question(self) -> dict[str, Any]:
        """Type 1: Which position contains a voxel?"""
        structure = self._game.current_voxels
        if not structure:
            raise ValueError("Structure is empty")

        correct_pos = random.choice(list(structure))

        # Generate 5 wrong options
        all_positions = [
            (x, y, z) for x in range(1, 4) for y in range(1, 4) for z in range(1, 4)
        ]
        wrong_positions = [pos for pos in all_positions if pos not in structure]
        wrong_options = random.sample(wrong_positions, 5)

        # Shuffle all options
        all_options = [f"({x},{y},{z})" for x, y, z in [correct_pos] + wrong_options]
        random.shuffle(all_options)

        correct_index = (
            all_options.index(f"({correct_pos[0]},{correct_pos[1]},{correct_pos[2]})")
            + 1
        )

        question = (
            "This is a state in a 3D reconstruction game.\n\n"
            "Given:\n"
            "- A 3x3x3 grid structure containing voxels\n"
            "- An image containing the voxel structure and its target projections\n"
            "  (Note: The projections shown in the image are not relevant for this question)\n\n"
            "Game Rules:\n"
            "1. Grid Space: The game is played on a 3x3x3 cube grid.\n"
            "2. Coordinates: Position (x,y,z) ranges from 1 to 3, with (1,1,1) at front-left-bottom.\n"
            "3. Position Rule: Each position can contain at most one voxel.\n"
            "4. Connectivity: All voxels must be connected face-to-face.\n\n"
            "Question:\n"
            "Which of the following positions contains a voxel?\n\n"
            "Choose the correct position from the options below.\n"
            "Options:\n"
        )

        for i, option in enumerate(all_options, 1):
            question += f"{i}: {option}\n"

        analysis = "Let's analyze each option:\n\n"
        for i, option in enumerate(all_options, 1):
            x, y, z = map(int, option.strip("()").split(","))
            pos = (x, y, z)
            analysis += f"Option {i} - Position {option}:\n"
            if pos in structure:
                analysis += (
                    "- This position contains a voxel.\n- This is the correct answer.\n"
                )
            else:
                analysis += "- This position is empty.\n"
            analysis += "\n"

        analysis += f"Therefore, the correct answer is option {correct_index}."

        return {
            "question": question,
            "answer": str(correct_index),
            "analysis": analysis,
        }

    def _generate_projection_question(self) -> dict[str, Any]:
        """Type 2: Do projections match target?"""
        current_structure = self._game.current_voxels
        solution_structure = self._game_state["complete_solution"]["positions"]

        target_yz_proj, target_xz_proj = self._game.get_projections(solution_structure)
        current_yz_proj, current_xz_proj = self._game.get_projections(current_structure)

        yz_match = np.array_equal(current_yz_proj, target_yz_proj)
        xz_match = np.array_equal(current_xz_proj, target_xz_proj)

        if not yz_match and not xz_match:
            correct_index = 1
        elif yz_match and not xz_match:
            correct_index = 2
        elif not yz_match and xz_match:
            correct_index = 3
        else:
            correct_index = 4

        all_options = [
            "Neither Y-Z projection nor X-Z projection matches the target",
            "Only Y-Z projection matches the target",
            "Only X-Z projection matches the target",
            "Both Y-Z and X-Z projections match the target",
        ]

        question = (
            "This is a state in a 3D reconstruction game.\n\n"
            "Given:\n"
            "- A 3x3x3 grid structure containing voxels\n"
            "- An image containing the voxel structure and its target projections\n\n"
            "Game Rules:\n"
            "1. Grid Space: The game is played on a 3x3x3 cube grid.\n"
            "2. Coordinates: Position (x,y,z) ranges from 1 to 3, with (1,1,1) at front-left-bottom.\n"
            "3. Position Rule: Each position can contain at most one voxel.\n"
            "4. Connectivity: All voxels must be connected face-to-face.\n"
            "5. Front View (Y-Z): Shows structure when viewed along the negative X-axis direction, with Y as horizontal and Z as vertical.\n"
            "6. Side View (X-Z): Shows structure when viewed along the positive Y-axis direction, with X as horizontal and Z as vertical.\n"
            "7. Projection Rule: A cell shows '1' if any voxel exists along that line of sight.\n\n"
            "Question:\n"
            "How does the voxel structure's projections match with the target projections?\n\n"
            "Choose the correct description from the options below.\n"
            "Options:\n"
        )

        for i, option in enumerate(all_options, 1):
            question += f"{i}: {option}\n"

        analysis = "Let's analyze the projections:\n\n"
        analysis += "1. Front View (Y-Z) analysis:\n"
        if yz_match:
            analysis += "   - The Y-Z projection matches the target exactly\n\n"
        else:
            analysis += "   - The Y-Z projection does not match the target\n\n"

        analysis += "2. Side View (X-Z) analysis:\n"
        if xz_match:
            analysis += "   - The X-Z projection matches the target exactly\n\n"
        else:
            analysis += "   - The X-Z projection does not match the target\n\n"

        match_desc = (
            "neither projection matches"
            if not yz_match and not xz_match
            else "only Y-Z projection matches"
            if yz_match and not xz_match
            else "only X-Z projection matches"
            if not yz_match and xz_match
            else "both projections match"
        )
        analysis += f"Based on the above analysis, {match_desc} the target.\n"
        analysis += f"Therefore, the correct answer is option {correct_index}."

        return {
            "question": question,
            "answer": str(correct_index),
            "analysis": analysis,
        }

    def _generate_action_outcome_question(self) -> dict[str, Any]:
        """Type 3: Projection after adding voxels."""
        current_structure = self._game.current_voxels
        solution_structure = self._game_state["complete_solution"]["positions"]

        remaining = len(solution_structure) - len(current_structure)
        if remaining <= 0:
            raise ValueError("No remaining voxels")

        add_count = random.randint(1, min(3, remaining))
        _, new_voxels = self._game.generate_random_connected_voxels(
            add_count, respect_remaining=True
        )

        combined_structure = current_structure + new_voxels
        new_yz_proj, new_xz_proj = self._game.get_projections(combined_structure)

        check_yz = random.choice([True, False])
        proj_name = "Y-Z" if check_yz else "X-Z"

        question = (
            f"{GAME_RULES}\n"
            f"Action:\n"
            f"Add {len(new_voxels)} voxels at positions: {sorted(new_voxels)}\n\n"
            f"Question:\n"
            f"After adding these voxels, what will be the {proj_name} projection of the new structure?\n\n"
            "Answer Format:\n"
            "1. Write the answer as a list of three lists: [[row1], [row2], [row3]]\n"
            "2. Each row should contain three numbers (0 or 1)\n"
            "3. Rows are ordered from top to bottom of the projection\n"
            "4. Example format: [[0, 1, 0], [1, 1, 0], [0, 1, 1]]\n"
        )

        if check_yz:
            answer = str(
                [
                    [
                        int(new_yz_proj.T[2, 0]),
                        int(new_yz_proj.T[2, 1]),
                        int(new_yz_proj.T[2, 2]),
                    ],
                    [
                        int(new_yz_proj.T[1, 0]),
                        int(new_yz_proj.T[1, 1]),
                        int(new_yz_proj.T[1, 2]),
                    ],
                    [
                        int(new_yz_proj.T[0, 0]),
                        int(new_yz_proj.T[0, 1]),
                        int(new_yz_proj.T[0, 2]),
                    ],
                ]
            )
        else:
            answer = str(
                [
                    [
                        int(new_xz_proj.T[2, 0]),
                        int(new_xz_proj.T[2, 1]),
                        int(new_xz_proj.T[2, 2]),
                    ],
                    [
                        int(new_xz_proj.T[1, 0]),
                        int(new_xz_proj.T[1, 1]),
                        int(new_xz_proj.T[1, 2]),
                    ],
                    [
                        int(new_xz_proj.T[0, 0]),
                        int(new_xz_proj.T[0, 1]),
                        int(new_xz_proj.T[0, 2]),
                    ],
                ]
            )

        analysis = f"After adding the voxels, the {proj_name} projection matrix (from top to bottom) is:\n{answer}"

        return {"question": question, "answer": answer, "analysis": analysis}

    def _generate_transition_path_question(self) -> dict[str, Any]:
        """Type 4: Which sequence to match projection?"""
        current_voxels = self._game_state["current_state"]["positions"]
        minimal_addition = self._game_state["minimal_addition"]["positions"]
        target_voxels = self._game_state["complete_solution"]["positions"]
        remaining = len(target_voxels) - len(current_voxels)

        check_both = random.choice([True, False])
        proj_type = (
            "both target projections"
            if check_both
            else random.choice(["Y-Z", "X-Z"]) + " target projection"
        )

        question = (
            f"{GAME_RULES}\n"
            f"Question:\n"
            f"Which sequence of voxel additions will make the structure match the {proj_type}?\n"
            "Choose the correct sequence from the options below.\n\n"
            "Options:\n"
        )

        correct_option = f"Add voxels at positions: {sorted(minimal_addition)}"

        # Generate wrong options
        wrong_options = []
        for _ in range(7):
            extra_count = random.randint(1, min(5, remaining + 2))
            _, extra_voxels = self._game.generate_random_connected_voxels(extra_count)
            wrong_options.append(f"Add voxels at positions: {sorted(extra_voxels)}")

        all_options = [correct_option] + wrong_options
        random.shuffle(all_options)

        for i, option in enumerate(all_options, 1):
            question += f"{i}: {option}\n"

        correct_index = all_options.index(correct_option) + 1

        analysis = f"The minimal solution to match the {proj_type} is to add voxels at {sorted(minimal_addition)}.\nTherefore, the correct answer is option {correct_index}."

        return {
            "question": question,
            "answer": str(correct_index),
            "analysis": analysis,
        }

    def _generate_strategy_optimization_question(self) -> dict[str, Any]:
        """Type 5: Minimum voxels needed."""
        minimal_addition = self._game_state["minimal_addition"]["positions"]

        question = (
            f"{GAME_RULES}\n"
            "Question:\n"
            "What is the minimum number of voxels needed to add to the current structure\n"
            "to make it match both target projections?\n\n"
            "Please answer with a number."
        )

        answer = str(len(minimal_addition))
        analysis = f"The minimum number of voxels needed is {len(minimal_addition)}."

        return {"question": question, "answer": answer, "analysis": analysis}

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Process answer to current question."""
        if self._current_question is None:
            raise RuntimeError("No question generated. Call reset() first.")

        # Check answer
        correct = action.strip() == self._current_question["answer"].strip()
        reward = 1.0 if correct else 0.0

        if correct:
            response = "Correct!"
        else:
            response = (
                f"Incorrect. The correct answer is: {self._current_question['answer']}\n\n"
                f"{self._current_question['analysis']}"
            )

        obs = Observation(image=self.render(), text=response)
        return obs, reward, True, False, {}

    def render(self) -> Image.Image:
        """Render the 3D reconstruction game as a PIL Image.

        Creates a composite image showing:
        - 3D voxel structure (isometric view)
        - Front View (Y-Z projection)
        - Side View (X-Z projection)

        Returns:
            800x600 PIL Image
        """
        img_width = 800
        img_height = 600
        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Try to load font
        try:
            title_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
            )
            label_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
            )
        except OSError:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        # Title
        draw.text(
            (img_width // 2, 20),
            "3D Voxel Reconstruction Game",
            fill=(0, 0, 0),
            font=title_font,
            anchor="mm",
        )

        # Draw 3D isometric view (left side)
        self._draw_isometric_view(draw, 50, 100, self._game.current_voxels, label_font)

        # Calculate projections of current state
        current_yz, current_xz = self._game._calculate_projections(
            self._game.current_voxels
        )

        # Draw projections (right side)
        self._draw_projection(
            draw,
            500,
            100,
            current_yz,
            "Front View (Y-Z)",
            label_font,
        )
        self._draw_projection(
            draw,
            500,
            350,
            current_xz,
            "Side View (X-Z)",
            label_font,
        )

        # Show remaining voxels
        remaining = len(self._game.target_voxels) - len(self._game.current_voxels)
        draw.text(
            (400, 560),
            f"Remaining Available Voxels: {remaining}",
            fill=(255, 100, 0),
            font=label_font,
        )

        return img

    def _draw_isometric_view(self, draw, x_offset, y_offset, voxels, font):
        """Draw isometric 3D view of voxels."""
        # Isometric projection constants
        cube_size = 40

        # Draw grid lines first
        draw.text(
            (x_offset + 150, y_offset - 20),
            "3D Structure",
            fill=(0, 0, 0),
            font=font,
            anchor="mm",
        )

        # Draw voxels
        for vx, vy, vz in voxels:
            # Convert 3D coordinates to isometric 2D
            iso_x = x_offset + (vx - vy) * cube_size * 0.5
            iso_y = y_offset + (vx + vy) * cube_size * 0.25 - vz * cube_size * 0.5

            # Draw cube faces
            # Top face
            points_top = [
                (iso_x, iso_y),
                (iso_x + cube_size * 0.5, iso_y + cube_size * 0.25),
                (iso_x, iso_y + cube_size * 0.5),
                (iso_x - cube_size * 0.5, iso_y + cube_size * 0.25),
            ]
            draw.polygon(points_top, fill=(100, 180, 255), outline=(0, 0, 0))

            # Right face
            points_right = [
                (iso_x + cube_size * 0.5, iso_y + cube_size * 0.25),
                (iso_x, iso_y + cube_size * 0.5),
                (iso_x, iso_y + cube_size),
                (iso_x + cube_size * 0.5, iso_y + cube_size * 0.75),
            ]
            draw.polygon(points_right, fill=(70, 140, 200), outline=(0, 0, 0))

            # Left face
            points_left = [
                (iso_x - cube_size * 0.5, iso_y + cube_size * 0.25),
                (iso_x, iso_y + cube_size * 0.5),
                (iso_x, iso_y + cube_size),
                (iso_x - cube_size * 0.5, iso_y + cube_size * 0.75),
            ]
            draw.polygon(points_left, fill=(50, 120, 180), outline=(0, 0, 0))

    def _draw_projection(self, draw, x_offset, y_offset, projection, title, font):
        """Draw a 2D projection grid."""
        cell_size = 60

        draw.text(
            (x_offset + cell_size * 1.5, y_offset - 20),
            title,
            fill=(0, 0, 0),
            font=font,
            anchor="mm",
        )

        # Draw grid
        for row in range(3):
            for col in range(3):
                x = x_offset + col * cell_size
                y = y_offset + (2 - row) * cell_size  # Flip vertically

                # Fill cell
                # Note: row is flipped (2-row maps to screen position), so we need to access projection[col, 2-row]
                # to correctly map: col -> horizontal (y or x), row -> vertical (z with flip)
                color = (
                    (100, 150, 255)
                    if projection[col, 2 - row] == 1
                    else (240, 240, 240)
                )
                draw.rectangle(
                    [x, y, x + cell_size, y + cell_size],
                    fill=color,
                    outline=(0, 0, 0),
                    width=2,
                )

                # Draw value
                value = str(projection[col, 2 - row])
                draw.text(
                    (x + cell_size // 2, y + cell_size // 2),
                    value,
                    fill=(0, 0, 0),
                    font=font,
                    anchor="mm",
                )

        # Draw axis labels
        for i in range(3):
            # Horizontal axis labels (Y or X)
            draw.text(
                (
                    x_offset + i * cell_size + cell_size // 2,
                    y_offset + 3 * cell_size + 15,
                ),
                str(i + 1),
                fill=(0, 0, 0),
                font=font,
                anchor="mm",
            )
            # Vertical axis labels (Z)
            draw.text(
                (x_offset - 15, y_offset + (2 - i) * cell_size + cell_size // 2),
                str(i + 1),
                fill=(0, 0, 0),
                font=font,
                anchor="mm",
            )

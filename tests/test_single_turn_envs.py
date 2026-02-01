"""Unified tests for all single-turn gym-v environments.

This module consolidates tests for single-turn Q&A and puzzle environments across
multiple benchmark suites: ReasoningGym, GameRL, Perception, VGRP, Sphinx, and RLVE.

All environments follow the multi-agent dictionary interface with max_episode_steps=1.
Tests verify image generation, oracle answer validity, and reward grading for both
correct and incorrect answers.
"""

from __future__ import annotations

import base64
import html
import json
import os
from pathlib import Path
import random
import string
from typing import Any
import unittest

import numpy as np

import gym_v
from gym_v.core import Observation
from gym_v.envs.registration import registry as ENV_REGISTRY

# Environment registries organized by benchmark suite
REASONING_GYM_ENVS = {
    "ReasoningGym/Arc1D-v0": "arc1_d",
    "ReasoningGym/BinaryMatrix-v0": "binary_matrix",
    "ReasoningGym/CircuitLogic-v0": "circuit_logic",
    "ReasoningGym/GameOfLife-v0": "game_of_life",
    "ReasoningGym/Kakurasu-v0": "kakurasu",
    "ReasoningGym/KnightSwap-v0": "knight_swap",
    "ReasoningGym/LargestIsland-v0": "largest_island",
    "ReasoningGym/Maze-v0": "maze",
    "ReasoningGym/MiniSudoku-v0": "mini_sudoku",
    "ReasoningGym/NQueens-v0": "n_queens",
    "ReasoningGym/RectangleCount-v0": "rectangle_count",
    "ReasoningGym/RotateMatrix-v0": "rotate_matrix",
    "ReasoningGym/RottenOranges-v0": "rotten_oranges",
    "ReasoningGym/ShortestPath-v0": "shortest_path",
    "ReasoningGym/SpiralMatrix-v0": "spiral_matrix",
    "ReasoningGym/Sudoku-v0": "sudoku",
    "ReasoningGym/Survo-v0": "survo",
    "ReasoningGym/TowerOfHanoi-v0": "tower_of_hanoi",
    "ReasoningGym/Tsumego-v0": "tsumego",
}

GAMERL_ENVS = {
    "GameRL/3DReconstruction-QA-v0": "3_d_reconstruction_qa",
    "GameRL/ChessRanger-QA-v0": "chess_ranger_qa",
    "GameRL/Freecell-QA-v0": "freecell_qa",
    "GameRL/Hue-QA-v0": "hue_qa",
    "GameRL/Jewel2-QA-v0": "jewel2_qa",
    "GameRL/Klondike-QA-v0": "klondike_qa",
    "GameRL/LangtonAnt-QA-v0": "langton_ant_qa",
    "GameRL/Lifegame-QA-v0": "lifegame_qa",
    "GameRL/Maze-QA-v0": "maze_qa",
    "GameRL/Maze3D-QA-v0": "maze3_d_qa",
    "GameRL/Minecraft-QA-v0": "minecraft_qa",
    "GameRL/Minesweeper-QA-v0": "minesweeper_qa",
    "GameRL/Pacman-QA-v0": "pacman_qa",
    "GameRL/PyramidChess-QA-v0": "pyramid_chess_qa",
    "GameRL/RhythmGame-QA-v0": "rhythm_game_qa",
    "GameRL/RubiksCube-QA-v0": "rubiks_cube_qa",
    "GameRL/Snake-QA-v0": "snake_qa",
    "GameRL/Sokoban-QA-v0": "sokoban_qa",
    "GameRL/SpaceInvaders-QA-v0": "space_invaders_qa",
    "GameRL/SpiderSolitaire-QA-v0": "spider_solitaire_qa",
    "GameRL/StarBattle-QA-v0": "star_battle_qa",
    "GameRL/Sudoku-QA-v0": "sudoku_qa",
    "GameRL/Tangram-QA-v0": "tangram_qa",
    "GameRL/Tents-QA-v0": "tents_qa",
    "GameRL/Tetris-QA-v0": "tetris_qa",
    "GameRL/TicTacToe-QA-v0": "tic_tac_toe_qa",
    "GameRL/TuringMachine2d-QA-v0": "turing_machine2d_qa",
    "GameRL/UltraTicTacToe-QA-v0": "ultra_tic_tac_toe_qa",
    "GameRL/WordSearch-QA-v0": "word_search_qa",
    "GameRL/Zuma-QA-v0": "zuma_qa",
}

PERCEPTION_ENVS = {
    "Perception/ChartToTable-v0": "chart_to_table",
    "Perception/ContourPlot-v0": "contour_plot",
    "Perception/DAGToTopoOrder-v0": "dag_to_topo_order",
    "Perception/FlowNetwork-v0": "flow_network",
    "Perception/FunctionGraph-v0": "function_graph",
    "Perception/GraphToAdjacency-v0": "graph_to_adjacency",
    "Perception/GraphToMST-v0": "graph_to_mst",
    "Perception/ParametricCurve-v0": "parametric_curve",
    "Perception/PolarPlot-v0": "polar_plot",
    "Perception/TreeToTraversal-v0": "tree_to_traversal",
    "Perception/VectorField-v0": "vector_field",
}

VGRP_ENVS = {
    "VGRP/Battleships-v0": "battleships",
    "VGRP/Binairo-v0": "binairo",
    "VGRP/Futoshiki-v0": "futoshiki",
    "VGRP/Hitori-v0": "hitori",
    "VGRP/Renzoku-v0": "renzoku",
    "VGRP/StarBattle-v0": "star_battle",
    "VGRP/Thermometers-v0": "thermometers",
}

SPHINX_ENVS = {
    "Sphinx/OddOneOut-v0": "odd_one_out",
    "Sphinx/OddOneOutPoly-v0": "odd_one_out_poly",
    "Sphinx/SequenceCompletion-v0": "sequence_completion",
    "Sphinx/SequenceCompletionPoly-v0": "sequence_completion_poly",
    "Sphinx/SymmetryFill-v0": "symmetry_fill",
    "Sphinx/SymmetryFillPoly-v0": "symmetry_fill_poly",
    "Sphinx/TransformResult-v0": "transform_result",
    "Sphinx/TransformResultPoly-v0": "transform_result_poly",
}

RLVE_ENVS = {
    "RLVE/AdditionTable-v0": "addition_table",
    "RLVE/BinaryTreeLeafNumExpectation-v0": "binary_tree_leaf_num_expectation",
    "RLVE/BinarioNoAdjacencyRequirement-v0": "binario_no_adjacency_requirement",
    "RLVE/FbiBinaryTree-v0": "fbi_binary_tree",
    "RLVE/BlockImage-v0": "block_image",
    "RLVE/CampsitePuzzle-v0": "campsite_puzzle",
    "RLVE/CardColoringCounting-v0": "card_coloring_counting",
    "RLVE/CirculatingGrid-v0": "circulating_grid",
    "RLVE/CoinSquareGame-v0": "coin_square_game",
    "RLVE/ColoringCounting-v0": "coloring_counting",
    "RLVE/ConvexHull-v0": "convex_hull",
    "RLVE/EightDigitPuzzle-v0": "eight_digit_puzzle",
    "RLVE/FaceRightWay-v0": "face_right_way",
    "RLVE/GraMinimaGame-v0": "gra_minima_game",
    "RLVE/GraphContainTreeCounting-v0": "graph_contain_tree_counting",
    "RLVE/GridBFS-v0": "grid_bfs",
    "RLVE/GridComponent-v0": "grid_component",
    "RLVE/GridLocalMinimumCounting-v0": "grid_local_minimum_counting",
    "RLVE/GridParityConstruction-v0": "grid_parity_construction",
    "RLVE/GridTriangleCounting-v0": "grid_triangle_counting",
    "RLVE/HamiltonianPath-v0": "hamiltonian_path",
    "RLVE/HamiltonianPathExistence-v0": "hamiltonian_path_existence",
    "RLVE/HitoriPuzzle-v0": "hitori_puzzle",
    "RLVE/JugPuzzle-v0": "jug_puzzle",
    "RLVE/KloBlocks-v0": "klo_blocks",
    "RLVE/LandformGenerationCounting-v0": "landform_generation_counting",
    "RLVE/LargestConvexPolygon-v0": "largest_convex_polygon",
    "RLVE/LargestRectangleAmongPoints-v0": "largest_rectangle_among_points",
    "RLVE/SkyscraperPuzzle-v0": "skyscraper_puzzle",
    "RLVE/SkyscraperSumPuzzle-v0": "skyscraper_sum_puzzle",
    "RLVE/SumTriangleArea-v0": "sum_triangle_area",
    "RLVE/SumManhattanCurvedSurface-v0": "sum_manhattan_curved_surface",
    "RLVE/LongestPath-v0": "longest_path",
    "RLVE/MaximumAchromaticNumber-v0": "maximum_achromatic_number",
    "RLVE/MaximumIndependentSetGrid-v0": "maximum_independent_set_grid",
    "RLVE/MaximumIndependentSetTree-v0": "maximum_independent_set_tree",
    "RLVE/MaximumWeightMatching-v0": "maximum_weight_matching",
    "RLVE/MinimumChromaticNumber-v0": "minimum_chromatic_number",
    "RLVE/MinimumDirectedSpanningTree-v0": "minimum_directed_spanning_tree",
    "RLVE/MinimumDominatingSetGrid-v0": "minimum_dominating_set_grid",
    "RLVE/MinimumSpanningTreeCounting-v0": "minimum_spanning_tree_counting",
    "RLVE/MinimumWeightedSpanningTree-v0": "minimum_weighted_spanning_tree",
    "RLVE/MixedGraphEulerianCircuit-v0": "mixed_graph_eulerian_circuit",
    "RLVE/MonochromeBlockCounting-v0": "monochrome_block_counting",
    "RLVE/NinePuzzle-v0": "nine_puzzle",
    "RLVE/Numbrix-v0": "numbrix",
    "RLVE/PipelineArrangement-v0": "pipeline_arrangement",
    "RLVE/MagicSquarePuzzle-v0": "magic_square_puzzle",
    "RLVE/MatrixPermutationBothDiagonalOne-v0": "matrix_permutation_both_diagonal_one",
    "RLVE/MatrixPermutationMainDiagonalOne-v0": "matrix_permutation_main_diagonal_one",
    "RLVE/MatrixPooling-v0": "matrix_pooling",
    "RLVE/MatrixRmqCounting-v0": "matrix_rmq_counting",
    "RLVE/MaxGridPathIntersection-v0": "max_grid_path_intersection",
    "RLVE/MoneyChargingGame-v0": "money_charging_game",
    "RLVE/SmallestCircle-v0": "smallest_circle",
    "RLVE/TreeCenter-v0": "tree_center",
    "RLVE/TreeAddOneEdgeDiameter-v0": "tree_add_one_edge_diameter",
    "RLVE/TreeChangeOneEdgeDiameter-v0": "tree_change_one_edge_diameter",
    "RLVE/TreeColoring-v0": "tree_coloring",
    "RLVE/TreeDistanceEqualTriadCounting-v0": "tree_distance_equal_triad_counting",
    "RLVE/TreeEvenPartitioning-v0": "tree_even_partitioning",
    "RLVE/TreeTopologicalSequenceCounting-v0": "tree_topological_sequence_counting",
    "RLVE/TetrisAttack-v0": "tetris_attack",
    "RLVE/TwiddlePuzzle-v0": "twiddle_puzzle",
    "RLVE/WarehouseConstruction-v0": "warehouse_construction",
    "RLVE/WeightedBinarytree-v0": "weighted_binarytree",
    "RLVE/WhackAMole-v0": "whack_a_mole",
    "RLVE/NewNimGame-v0": "new_nim_game",
    "RLVE/Patrol-v0": "patrol",
    "RLVE/StoneGame-v0": "stone_game",
    "RLVE/StoneIntervalsGame-v0": "stone_intervals_game",
    "RLVE/VisibleLine-v0": "visible_line",
    "RLVE/SkaRockGarden-v0": "ska_rock_garden",
    "RLVE/SpyNetwork-v0": "spy_network",
}

# Environments that use partial credit scoring (from reasoning-gym library)
# These environments give partial scores based on correctness ratio, not strict 0/1
PARTIAL_CREDIT_ENVS = {
    # Grid-based environments with cell-by-cell scoring
    "ReasoningGym/MiniSudoku-v0": {
        "reason": "4x4 grid, scores by correct cells / 16",
        "max_wrong_reward": 0.99,  # Allow up to 99% partial credit
    },
    "ReasoningGym/Sudoku-v0": {
        "reason": "9x9 grid, scores by correct cells / 81 with constraint penalties",
        "max_wrong_reward": 0.99,
    },
    # Substring matching environments
    "ReasoningGym/RotateMatrix-v0": {
        "reason": "Substring matching: len(correct) / len(answer)",
        "max_wrong_reward": 0.99,
    },
    # Format-based partial credit
    "ReasoningGym/Tsumego-v0": {
        "reason": "Go coordinates: 0.05 for valid format, 0.01 otherwise",
        "max_wrong_reward": 0.1,  # Only allow small partial credit for format
    },
    # Environments that accept multiple valid solutions
    "VGRP/StarBattle-v0": {
        "reason": "Accepts any solution satisfying constraints; invalid chars normalize to 'e'",
        "max_wrong_reward": 1.0,  # Allow full credit for alternative valid solutions
        "allow_alternative_solutions": True,
    },
    "VGRP/Thermometers-v0": {
        "reason": "Accepts any solution satisfying constraints; invalid chars normalize to 'e'",
        "max_wrong_reward": 1.0,  # Allow full credit for alternative valid solutions
        "allow_alternative_solutions": True,
    },
    # RLVE puzzle with power-based scoring
    "RLVE/TwiddlePuzzle-v0": {
        "reason": "Grid transformation puzzle: scores by (matching_cells / total_cells)^5",
        "max_wrong_reward": 0.99,  # Allow up to 99% partial credit
    },
    # RLVE optimization problems with multiple valid solutions
    "RLVE/CirculatingGrid-v0": {
        "reason": "Optimization puzzle: minimizes cell changes to create circulation, scores by (gold/answer)^5",
        "max_wrong_reward": 1.0,  # Allow full credit for any valid solution
        "allow_alternative_solutions": True,
    },
    "RLVE/MinimumDominatingSetGrid-v0": {
        "reason": "Optimization puzzle: minimizes total cost of dominating set, scores by (gold/answer)^5",
        "max_wrong_reward": 1.0,  # Allow full credit for any valid optimal solution
        "allow_alternative_solutions": True,
    },
    "RLVE/LargestConvexPolygon-v0": {
        "reason": "Optimization puzzle: maximizes convex polygon size, scores by (answer/gold)^5",
        "max_wrong_reward": 1.0,  # Allow full credit for any valid optimal solution
        "allow_alternative_solutions": True,
    },
    "RLVE/LargestRectangleAmongPoints-v0": {
        "reason": "Optimization puzzle: maximizes rectangle area, scores by (answer/gold)^5",
        "max_wrong_reward": 1.0,  # Allow full credit for any valid optimal solution
        "allow_alternative_solutions": True,
    },
    "RLVE/MaximumIndependentSetTree-v0": {
        "reason": "Optimization puzzle: maximizes total weight of independent set in tree, scores by (answer/gold)^3",
        "max_wrong_reward": 1.0,  # Allow full credit for any valid optimal solution
        "allow_alternative_solutions": True,
    },
    "RLVE/MaximumWeightMatching-v0": {
        "reason": "Optimization puzzle: maximizes total weight of matching, scores by (answer/gold)^5",
        "max_wrong_reward": 1.0,  # Allow full credit for any valid optimal solution
        "allow_alternative_solutions": True,
    },
    "RLVE/MinimumWeightedSpanningTree-v0": {
        "reason": "Optimization puzzle: minimizes weighted depth of spanning tree, scores by (gold/answer)^5",
        "max_wrong_reward": 1.0,  # Allow full credit for any valid optimal solution
        "allow_alternative_solutions": True,
    },
    "RLVE/TreeAddOneEdgeDiameter-v0": {
        "reason": "Optimization puzzle: minimizes tree diameter by adding edge, scores by (gold/answer)^5",
        "max_wrong_reward": 1.0,  # Allow full credit for any valid optimal solution
        "allow_alternative_solutions": True,
    },
    "RLVE/WarehouseConstruction-v0": {
        "reason": "Optimization puzzle: minimizes warehouse + transport costs, scores by (gold/answer)^5",
        "max_wrong_reward": 1.0,  # Allow full credit for any valid optimal solution
        "allow_alternative_solutions": True,
    },
    "RLVE/WeightedBinarytree-v0": {
        "reason": "Optimization puzzle: maximizes binary tree score, scores by (answer/gold)^5",
        "max_wrong_reward": 1.0,  # Allow full credit for any valid optimal solution
        "allow_alternative_solutions": True,
    },
    "RLVE/TetrisAttack-v0": {
        "reason": "Optimization puzzle: minimizes swaps to clear array, scores by (gold/answer)^5",
        "max_wrong_reward": 1.0,  # Allow full credit for any valid optimal solution
        "allow_alternative_solutions": True,
    },
    "RLVE/SpyNetwork-v0": {
        "reason": "Optimization puzzle: minimizes vertex cover cost in directed graph, scores by (gold/answer)^3",
        "max_wrong_reward": 1.0,  # Allow full credit for any valid optimal solution
        "allow_alternative_solutions": True,
    },
    "GameRL/Pacman-QA-v0": {
        "reason": "Q&A with multiple valid answers depending on question type",
        "max_wrong_reward": 1.0,
        "allow_alternative_solutions": True,
    },
    "ReasoningGym/GameOfLife-v0": {
        "reason": "Partial credit based on cell-wise correctness",
        "max_wrong_reward": 0.99,
    },
    "ReasoningGym/RectangleCount-v0": {
        "reason": "Partial credit based on count proximity",
        "max_wrong_reward": 0.99,
    },
    "VGRP/Hitori-v0": {
        "reason": "Multiple valid solutions accepted by grader",
        "max_wrong_reward": 1.0,
        "allow_alternative_solutions": True,
    },
}

# Environments where oracle_answer may not align with scoring format
ORACLE_MAY_BE_INVALID = {
    "RLVE/NewNimGame-v0",
}


class TestSingleTurnEnvironments(unittest.TestCase):
    """Unified test suite for all single-turn gym-v environments.

    Tests verify:
    1. Environment creation and reset
    2. Image generation and saving
    3. Oracle answer retrieval
    4. Reward grading for correct answers (expect 1.0)
    5. Reward grading for wrong answers:
       - Empty string ""
       - Perturbed correct answer
    6. Multi-seed stability
    """

    def _get_output_dir(self, env_id: str) -> Path:
        """Get output directory for environment test artifacts.

        Args:
            env_id: Environment ID (e.g., "ReasoningGym/MMLU-v0")

        Returns:
            Path to output directory
        """
        suite, name = env_id.split("/")
        name_clean = name.replace("-v0", "")
        snake_name = "".join(
            f"_{c.lower()}" if c.isupper() else c for c in name_clean
        ).lstrip("_")
        root = Path(__file__).resolve().parent
        return root / f"test_output_{suite.lower()}_{snake_name}"

    def _image_to_data_uri(self, image: Any) -> list[str]:
        """Convert PIL Image(s) to base64 data URIs."""
        if image is None:
            return []
        images = image if isinstance(image, list) else [image]
        uris: list[str] = []
        for idx, img in enumerate(images):
            with (self._tmp_dir / f"__tmp_{idx}.png").open("wb") as f:
                img.save(f, format="PNG")
            raw = (self._tmp_dir / f"__tmp_{idx}.png").read_bytes()
            encoded = base64.b64encode(raw).decode("ascii")
            uris.append(f"data:image/png;base64,{encoded}")
        return uris

    def _write_html_report(
        self, output_dir: Path, env_id: str, cases: list[dict[str, Any]]
    ) -> None:
        """Write an HTML report with descriptions and observations."""
        html_path = output_dir / "report.html"
        sections = []
        for case in cases:
            images_html = ""
            for uri in case["image_uris"]:
                images_html += (
                    f'<img src="{uri}" style="max-width: 420px; margin: 8px;" />'
                )
            sections.append(
                f"""
                <section>
                  <h2>{html.escape(case["label"])}</h2>
                  <p><strong>difficulty</strong>: {case["difficulty"]}</p>
                  <p><strong>seed</strong>: {case["seed"]}</p>
                  <pre><strong>description</strong>\n{html.escape(case["description"])}</pre>
                  <pre><strong>observation.text</strong>\n{html.escape(case["obs_text"])}</pre>
                  <pre><strong>observation.metadata</strong>\n{html.escape(case["obs_metadata"])}</pre>
                  <pre><strong>info</strong>\n{html.escape(case["info"])}</pre>
                  <pre><strong>oracle</strong>\n{html.escape(case["oracle"])}</pre>
                  <pre><strong>reward</strong>\n{html.escape(case["reward"])}</pre>
                  <div>{images_html}</div>
                </section>
                """
            )

        content = f"""
        <html>
          <head>
            <meta charset="utf-8" />
            <title>{html.escape(env_id)} report</title>
            <style>
              body {{ font-family: Arial, sans-serif; margin: 24px; }}
              section {{ margin-bottom: 32px; border-bottom: 1px solid #ddd; padding-bottom: 24px; }}
              pre {{ background: #f6f6f6; padding: 12px; white-space: pre-wrap; }}
              img {{ border: 1px solid #ddd; }}
            </style>
          </head>
          <body>
            <h1>{html.escape(env_id)} test report</h1>
            {''.join(sections)}
          </body>
        </html>
        """
        html_path.write_text(content, encoding="utf-8")

    def _setup_output_dir(self, output_dir: Path) -> None:
        """Create or clean output directory.

        Args:
            output_dir: Directory to setup
        """
        if output_dir.exists():
            for p in output_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

    def _get_oracle_answer(self, env: Any, info: dict[str, Any]) -> str:
        """Retrieve oracle answer from environment or info dict.

        Different benchmark suites store oracle answers in different locations:
        - ReasoningGym: env.get_wrapper_attr("_oracle_answer")
        - RLVE: info["reference_answer"]
        - Others: info["oracle_answer"]

        Args:
            env: The gym environment
            info: Info dictionary from reset/step

        Returns:
            Oracle answer string
        """
        # Try wrapper attribute first (ReasoningGym)
        try:
            oracle = env.get_wrapper_attr("_oracle_answer")
            if oracle:
                return self._normalize_oracle(oracle)
        except (AttributeError, KeyError):
            pass

        # Try info dict (most common)
        oracle = info.get("oracle_answer")
        if oracle is not None and oracle != "":
            return self._normalize_oracle(oracle)

        # Try reference_answer (RLVE)
        oracle = info.get("reference_answer")
        if oracle is not None and oracle != "":
            return self._normalize_oracle(oracle)

        raise ValueError(f"Could not find oracle answer in env or info: {info.keys()}")

    def _normalize_oracle(self, oracle: Any) -> str:
        """Normalize oracle answers to a string format accepted by envs."""
        if isinstance(oracle, str):
            return oracle
        if isinstance(oracle, int | float | np.integer | np.floating):
            return str(oracle)
        if isinstance(oracle, list | tuple | np.ndarray):
            if len(oracle) == 0:
                return ""
            if all(isinstance(x, list | tuple | np.ndarray) for x in oracle):
                rows = [" ".join(str(v) for v in row) for row in oracle]
                return "\n".join(rows)
            return " ".join(str(v) for v in oracle)
        if isinstance(oracle, dict):
            return json.dumps(oracle, ensure_ascii=False)
        return str(oracle)

    def _safe_json(self, payload: Any) -> str:
        """Serialize metadata with numpy-friendly defaults."""

        def _default(obj: Any):
            if isinstance(obj, np.integer | np.floating):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        return json.dumps(payload, ensure_ascii=False, indent=2, default=_default)

    def _perturb_answer(self, answer: str) -> str:
        """Create a perturbed version of the correct answer.

        Applies random perturbation to create an incorrect but non-empty answer:
        - If multi-choice format like "(a)", change to different option
        - Otherwise, randomly modify 1-3 characters

        Args:
            answer: Correct answer string

        Returns:
            Perturbed answer string
        """
        if not answer:
            return "WRONG"

        # Handle multi-choice format: (a), (b), etc.
        if len(answer) == 3 and answer[0] == "(" and answer[2] == ")":
            choices = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
            other_choices = [c for c in choices if c != answer]
            return random.choice(other_choices)

        # For other formats, randomly perturb the string
        answer_list = list(answer)
        num_changes = min(random.randint(1, 3), len(answer_list))

        for _ in range(num_changes):
            idx = random.randint(0, len(answer_list) - 1)
            # Replace with random alphanumeric character
            answer_list[idx] = random.choice(string.ascii_letters + string.digits)

        perturbed = "".join(answer_list)

        # Ensure it's actually different
        if perturbed == answer:
            return answer + "_WRONG"

        return perturbed

    def _test_env(self, env_id: str, env_name: str) -> None:
        """Test a single environment comprehensively.

        Args:
            env_id: Environment ID for gym_v.make()
            env_name: Human-readable environment name
        """
        output_dir = self._get_output_dir(env_id)
        self._setup_output_dir(output_dir)
        self._tmp_dir = output_dir

        test_seed = random.randint(0, 9999)
        print(f"\n[{env_id}] Using random seed: {test_seed}")

        if env_id not in ENV_REGISTRY:
            self.skipTest(f"{env_id} is not registered in this build")
        env = gym_v.make(env_id)

        # Reset environment
        obs_dict, info_dict = env.reset(seed=test_seed)

        # Extract agent_0 data (all single-turn envs use agent_0)
        agent_id = "agent_0"
        self.assertIn(agent_id, obs_dict, f"{env_id}: agent_0 not in obs_dict")
        self.assertIn(agent_id, info_dict, f"{env_id}: agent_0 not in info_dict")

        obs: Observation = obs_dict[agent_id]
        info = info_dict[agent_id]

        # Test 1: Save image
        self.assertIsNotNone(obs.image, f"{env_id}: obs.image is None")
        obs.image.save(output_dir / "0_reset.png")

        # Test 2: Get oracle answer
        oracle = self._get_oracle_answer(env, info)
        self.assertIsInstance(oracle, str, f"{env_id}: oracle is not string")
        self.assertGreater(len(oracle), 0, f"{env_id}: oracle is empty")

        # Print diagnostic info
        print("\n" + "=" * 80)
        print(f"[{env_id}] SEED: {test_seed}")
        print(f"[{env_id}] DESCRIPTION:")
        desc = env.description[:500] if len(env.description) > 500 else env.description
        print(desc)
        print(f"\n[{env_id}] OBS.TEXT:")
        text = obs.text or "No text"
        print(text[:500] if len(text) > 500 else text)
        print(f"\n[{env_id}] ORACLE ANSWER:")
        print(oracle[:300] + "..." if len(oracle) > 300 else oracle)
        print("=" * 80 + "\n")

        # Test 3: Verify correct answer reward
        action_dict = {agent_id: oracle}
        _, reward_dict, terminated_dict, _, _ = env.step(action_dict)

        self.assertIn(agent_id, reward_dict)
        self.assertIn(agent_id, terminated_dict)
        self.assertTrue(
            terminated_dict[agent_id], f"{env_id}: not terminated after step"
        )
        self.assertIsInstance(reward_dict[agent_id], float)

        # For environments with alternative solutions, skip exact reward check
        if env_id in PARTIAL_CREDIT_ENVS and PARTIAL_CREDIT_ENVS[env_id].get(
            "allow_alternative_solutions", False
        ):
            # Just verify reward is positive for valid solutions
            self.assertGreater(
                reward_dict[agent_id],
                0.0,
                f"{env_id}: Expected positive reward for valid solution, got {reward_dict[agent_id]}",
            )
        elif env_id in ORACLE_MAY_BE_INVALID:
            self.assertIsInstance(
                reward_dict[agent_id],
                float,
                msg=f"{env_id}: Expected float reward for oracle answer",
            )
        else:
            self.assertAlmostEqual(
                reward_dict[agent_id],
                1.0,
                places=6,
                msg=f"{env_id}: Expected reward 1.0 for correct answer, got {reward_dict[agent_id]}",
            )

        # Test 4: Verify empty string reward (should be 0.0 or negative)
        obs_dict, info_dict = env.reset(seed=test_seed)
        action_empty = {agent_id: ""}
        _, reward_empty, terminated_empty, _, _ = env.step(action_empty)

        self.assertTrue(terminated_empty[agent_id])
        self.assertIsInstance(reward_empty[agent_id], float)

        # RLVE environments return negative rewards for wrong answers
        if "RLVE" not in env_id:
            self.assertEqual(
                reward_empty[agent_id],
                0.0,
                f"{env_id}: Expected reward 0.0 for empty answer, got {reward_empty[agent_id]}",
            )

        # Test 5: Verify perturbed answer reward
        obs_dict, info_dict = env.reset(seed=test_seed)
        info_reset = info_dict[agent_id]
        oracle_reset = self._get_oracle_answer(env, info_reset)

        perturbed = self._perturb_answer(oracle_reset)
        action_perturbed = {agent_id: perturbed}
        _, reward_perturbed, terminated_perturbed, _, _ = env.step(action_perturbed)

        self.assertTrue(terminated_perturbed[agent_id])
        self.assertIsInstance(reward_perturbed[agent_id], float)

        # Check reward based on environment type
        if env_id in PARTIAL_CREDIT_ENVS:
            # Partial credit environments from reasoning-gym
            # These give scores based on correctness ratio, not strict 0/1
            config = PARTIAL_CREDIT_ENVS[env_id]
            max_allowed = config["max_wrong_reward"]
            allow_alternative_solutions = config.get(
                "allow_alternative_solutions", False
            )

            # For environments that don't accept alternative solutions,
            # perturbed answer should not get full credit
            if not allow_alternative_solutions:
                self.assertLess(
                    reward_perturbed[agent_id],
                    1.0,
                    f"{env_id}: Perturbed answer should not get full credit (got {reward_perturbed[agent_id]})",
                )

            # Should not exceed the maximum allowed partial credit
            self.assertLessEqual(
                reward_perturbed[agent_id],
                max_allowed,
                f"{env_id}: Perturbed answer reward {reward_perturbed[agent_id]} exceeds max {max_allowed}. "
                f"Reason: {config['reason']}. Perturbed: '{perturbed}'",
            )
        elif "RLVE" not in env_id:
            # Standard environments expect 0.0 for wrong answers
            self.assertEqual(
                reward_perturbed[agent_id],
                0.0,
                f"{env_id}: Expected reward 0.0 for perturbed answer '{perturbed}', got {reward_perturbed[agent_id]}",
            )

        # Test 6: Multi-seed stability
        print(f"[{env_id}] Testing with 3 additional seeds...")
        for i in range(3):
            seed = random.randint(0, 9999)
            obs_test_dict, info_test_dict = env.reset(seed=seed)

            obs_test = obs_test_dict[agent_id]
            info_test = info_test_dict[agent_id]

            # Save image
            self.assertIsNotNone(obs_test.image, f"{env_id}: image None (seed={seed})")
            obs_test.image.save(output_dir / f"{i + 1}_seed_{seed}.png")

            # Get oracle
            oracle_test = self._get_oracle_answer(env, info_test)
            self.assertIsNotNone(oracle_test, f"{env_id}: oracle None (seed={seed})")
            self.assertIsInstance(oracle_test, str)
            self.assertGreater(len(oracle_test), 0)

            # Verify correct answer gives reward 1.0 (or positive for optimization envs)
            _, reward_test_dict, _, _, _ = env.step({agent_id: oracle_test})
            if env_id in PARTIAL_CREDIT_ENVS and PARTIAL_CREDIT_ENVS[env_id].get(
                "allow_alternative_solutions", False
            ):
                self.assertGreater(
                    reward_test_dict[agent_id],
                    0.0,
                    msg=f"{env_id}: Expected positive reward (seed={seed})",
                )
            elif env_id in ORACLE_MAY_BE_INVALID:
                self.assertIsInstance(
                    reward_test_dict[agent_id],
                    float,
                    msg=f"{env_id}: Expected float reward (seed={seed})",
                )
            else:
                self.assertAlmostEqual(
                    reward_test_dict[agent_id],
                    1.0,
                    places=6,
                    msg=f"{env_id}: Expected reward 1.0 (seed={seed})",
                )

            print(f"  ✓ Seed {seed}: Generated valid puzzle with oracle answer")

        # Difficulty tests (make-time)
        cases: list[dict[str, Any]] = []
        cases.append(
            {
                "label": "default",
                "difficulty": "None",
                "seed": test_seed,
                "description": env.description,
                "obs_text": obs.text or "",
                "obs_metadata": self._safe_json(obs.metadata),
                "info": self._safe_json(info),
                "oracle": oracle,
                "reward": str(reward_dict[agent_id]),
                "image_uris": self._image_to_data_uri(obs.image),
            }
        )

        for difficulty in [0, 5]:
            env_d = gym_v.make(env_id, difficulty=difficulty)
            obs_d_dict, info_d_dict = env_d.reset(seed=test_seed)
            obs_d: Observation = obs_d_dict[agent_id]
            info_d = info_d_dict[agent_id]
            oracle_d = self._get_oracle_answer(env_d, info_d)
            _, reward_d, _, _, _ = env_d.step({agent_id: oracle_d})
            if obs_d.image is not None:
                obs_d.image.save(output_dir / f"difficulty_{difficulty}.png")
            cases.append(
                {
                    "label": f"difficulty={difficulty}",
                    "difficulty": str(difficulty),
                    "seed": test_seed,
                    "description": env_d.description,
                    "obs_text": obs_d.text or "",
                    "obs_metadata": self._safe_json(obs_d.metadata),
                    "info": self._safe_json(info_d),
                    "oracle": oracle_d,
                    "reward": str(reward_d[agent_id]),
                    "image_uris": self._image_to_data_uri(obs_d.image),
                }
            )
            env_d.close()

        self._write_html_report(output_dir, env_id, cases)

        env.close()
        print(f"✅ {env_id}: All tests passed (primary_seed={test_seed})")


def _make_test_method(env_id: str, env_name: str):
    """Factory function to dynamically create test methods.

    Args:
        env_id: Environment ID
        env_name: Environment name for test method naming

    Returns:
        Test method function
    """

    def test_method(self):
        self._test_env(env_id, env_name)

    test_method.__name__ = f"test_{env_name}"
    test_method.__doc__ = f"Test {env_id} environment."
    return test_method


# Dynamically generate test methods for all environments
ALL_ENVS = {
    **REASONING_GYM_ENVS,
    **GAMERL_ENVS,
    **PERCEPTION_ENVS,
    **VGRP_ENVS,
    **SPHINX_ENVS,
    **RLVE_ENVS,
}

selected_envs = ALL_ENVS
suite_filter = os.environ.get("SINGLE_TURN_SUITES")
env_filter = os.environ.get("SINGLE_TURN_ENV_IDS")
if suite_filter:
    suites = {s.strip() for s in suite_filter.split(",") if s.strip()}
    selected_envs = {
        env_id: env_name
        for env_id, env_name in selected_envs.items()
        if env_id.split("/")[0] in suites
    }
if env_filter:
    env_ids = {s.strip() for s in env_filter.split(",") if s.strip()}
    selected_envs = {
        env_id: env_name
        for env_id, env_name in selected_envs.items()
        if env_id in env_ids
    }

for _env_id, _env_name in selected_envs.items():
    _test_method = _make_test_method(_env_id, _env_name)
    setattr(TestSingleTurnEnvironments, _test_method.__name__, _test_method)


if __name__ == "__main__":
    unittest.main()

"""Unified tests for all single-turn gym-v environments.

All environments follow the multi-agent dictionary interface with max_episode_steps=1.
Tests verify image generation, oracle answer validity, and reward grading for both
correct and incorrect answers. Rewards must be in [0, 1].
"""

from __future__ import annotations

from pathlib import Path
import random
import string
from typing import Any
import unittest

import gym_v
from gym_v.core import Observation

# ---------------------------------------------------------------------------
# Environment registry – organised by category (matches directory structure)
# ---------------------------------------------------------------------------

ENVS_ALGORITHMIC = {
    "Algorithmic/AdditionTable-v0": "addition_table",
    "Algorithmic/BinaryMatrix-v0": "binary_matrix",
    "Algorithmic/CirculatingGrid-v0": "circulating_grid",
    "Algorithmic/CoinSquareGame-v0": "coin_square_game",
    "Algorithmic/FaceRightWay-v0": "face_right_way",
    "Algorithmic/GameOfLife-v0": "game_of_life",
    "Algorithmic/GraMinimaGame-v0": "gra_minima_game",
    "Algorithmic/GridBFS-v0": "grid_bfs",
    "Algorithmic/GridLocalMinimumCounting-v0": "grid_local_minimum_counting",
    "Algorithmic/LandformGenerationCounting-v0": "landform_generation_counting",
    "Algorithmic/LangtonAnt-QA-v0": "langton_ant",
    "Algorithmic/Lifegame-QA-v0": "lifegame",
    "Algorithmic/MatrixPermutationBothDiagonalOne-v0": "matrix_permutation_both_diagonal_one",
    "Algorithmic/MatrixPermutationMainDiagonalOne-v0": "matrix_permutation_main_diagonal_one",
    "Algorithmic/MaxGridPathIntersection-v0": "max_grid_path_intersection",
    "Algorithmic/MonochromeBlockCounting-v0": "monochrome_block_counting",
    "Algorithmic/RotateMatrix-v0": "rotate_matrix",
    "Algorithmic/RottenOranges-v0": "rotten_oranges",
    "Algorithmic/SpiralMatrix-v0": "spiral_matrix",
    "Algorithmic/StoneIntervalsGame-v0": "stone_intervals_game",
    "Algorithmic/TuringMachine2d-QA-v0": "turing_machine_2d",
}

ENVS_ARC = {
    "Arc/Arc1D-v0": "arc_1d",
    "Arc/ArcAgi-v0": "arc_agi",
    "Arc/ReArc-v0": "rearc",
}

ENVS_COGNITION = {
    "Cognition/Hue-QA-v0": "hue",
    "Cognition/Maze3D-QA-v0": "maze_3d",
    "Cognition/OddOneOutPoly-v0": "odd_one_out",
    "Cognition/RectangleCount-v0": "rectangle_count",
    "Cognition/RubiksCube-QA-v0": "rubiks_cube",
    "Cognition/SequenceCompletionPoly-v0": "sequence_completion",
    "Cognition/SymmetryFillPoly-v0": "symmetry_fill",
    "Cognition/TransformResultPoly-v0": "transform_result",
}

ENVS_GEOMETRY = {
    "Geometry/ConvexHull-v0": "convex_hull",
    "Geometry/LargestRectangleAmongPoints-v0": "largest_rectangle_among_points",
    "Geometry/SkaRockGarden-v0": "ska_rock_garden",
    "Geometry/SmallestCircle-v0": "smallest_circle",
    "Geometry/SumTriangleArea-v0": "sum_triangle_area",
    "Geometry/Tangram-QA-v0": "tangram",
    "Geometry/VisibleLine-v0": "visible_line",
}

ENVS_GRAPHS = {
    "Graphs/FbiBinaryTree-v0": "fbi_binary_tree",
    "Graphs/GraphContainTreeCounting-v0": "graph_contain_tree_counting",
    "Graphs/GraphIsomorphism-v0": "graph_isomorphism",
    "Graphs/GridComponent-v0": "grid_component",
    "Graphs/HamiltonianPath-v0": "hamiltonian_path",
    "Graphs/LargestIsland-v0": "largest_island",
    "Graphs/LongestPath-v0": "longest_path",
    "Graphs/MaximumAchromaticNumber-v0": "maximum_achromatic_number",
    "Graphs/MaximumClique-v0": "maximum_clique",
    "Graphs/MaximumIndependentSetGrid-v0": "maximum_independent_set_grid",
    "Graphs/MaximumIndependentSetTree-v0": "maximum_independent_set_tree",
    "Graphs/MaximumWeightMatching-v0": "maximum_weight_matching",
    "Graphs/MinimumChromaticNumber-v0": "minimum_chromatic_number",
    "Graphs/MinimumDirectedSpanningTree-v0": "minimum_directed_spanning_tree",
    "Graphs/MinimumSpanningTreeCounting-v0": "minimum_spanning_tree_counting",
    "Graphs/MixedGraphEulerianCircuit-v0": "mixed_graph_eulerian_circuit",
    "Graphs/Patrol-v0": "patrol",
    "Graphs/ShortestPath-v0": "shortest_path",
    "Graphs/TreeCenter-v0": "tree_center",
    "Graphs/TreeChangeOneEdgeDiameter-v0": "tree_change_one_edge_diameter",
    "Graphs/TreeColoring-v0": "tree_coloring",
    "Graphs/TreeDistanceEqualTriadCounting-v0": "tree_distance_equal_triad_counting",
    "Graphs/TreeEvenPartitioning-v0": "tree_even_partitioning",
    "Graphs/TreeTopologicalSequenceCounting-v0": "tree_topological_sequence_counting",
    "Graphs/WeightedBinarytree-v0": "weighted_binarytree",
}

ENVS_LOGIC = {
    "Logic/Binairo-v0": "binairo",
    "Logic/BinarioNoAdjacencyRequirement-v0": "binario_no_adjacency_requirement",
    "Logic/CampsitePuzzle-v0": "campsite_puzzle",
    "Logic/CircuitLogic-v0": "circuit_logic",
    "Logic/Futoshiki-v0": "futoshiki",
    "Logic/GridParityConstruction-v0": "grid_parity_construction",
    "Logic/Kakurasu-v0": "kakurasu",
    "Logic/MagicSquarePuzzle-v0": "magic_square_puzzle",
    "Logic/MiniSudoku-v0": "mini_sudoku",
    "Logic/NQueens-v0": "n_queens",
    "Logic/Numbrix-v0": "numbrix",
    "Logic/Renzoku-v0": "renzoku",
    "Logic/SkyscraperPuzzle-v0": "skyscraper_puzzle",
    "Logic/StarBattle-QA-v0": "star_battle",
    "Logic/Survo-v0": "survo",
    "Logic/Tents-QA-v0": "tents",
    "Logic/Thermometers-v0": "thermometers",
}

ENVS_PERCEPTION = {
    "Perception/3DReconstruction-QA-v0": "threed_reconstruction",
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

ENVS_PUZZLES = {
    "Puzzles/ChessRanger-QA-v0": "chess_ranger",
    "Puzzles/EightDigitPuzzle-v0": "eight_digit_puzzle",
    "Puzzles/Freecell-QA-v0": "freecell",
    "Puzzles/Jewel2-QA-v0": "jewel2",
    "Puzzles/KloBlocks-v0": "klo_blocks",
    "Puzzles/KnightSwap-v0": "knight_swap",
    "Puzzles/Maze-QA-v0": "maze_qa",
    "Puzzles/NinePuzzle-v0": "nine_puzzle",
    "Puzzles/Pacman-QA-v0": "pacman",
    "Puzzles/PyramidChess-QA-v0": "pyramidchess",
    "Puzzles/RhythmGame-QA-v0": "rhythm_game",
    "Puzzles/Snake-QA-v0": "snake",
    "Puzzles/SpaceInvaders-QA-v0": "space_invaders",
    "Puzzles/SpiderSolitaire-QA-v0": "spider_solitaire",
    "Puzzles/Tetris-QA-v0": "tetris",
    "Puzzles/TetrisAttack-v0": "tetris_attack",
    "Puzzles/TicTacToe-QA-v0": "tictactoe",
    "Puzzles/TowerOfHanoi-v0": "tower_of_hanoi",
    "Puzzles/Tsumego-v0": "tsumego",
    "Puzzles/TwiddlePuzzle-v0": "twiddle_puzzle",
    "Puzzles/UltraTicTacToe-QA-v0": "ultra_tictactoe",
    "Puzzles/WordSearch-QA-v0": "word_search",
    "Puzzles/Zuma-QA-v0": "zuma",
}

# All single-turn environments
ALL_ENVS = {
    **ENVS_ALGORITHMIC,
    **ENVS_ARC,
    **ENVS_COGNITION,
    **ENVS_GEOMETRY,
    **ENVS_GRAPHS,
    **ENVS_LOGIC,
    **ENVS_PERCEPTION,
    **ENVS_PUZZLES,
}

# ---------------------------------------------------------------------------
# Environments that use partial credit scoring
# ---------------------------------------------------------------------------
PARTIAL_CREDIT_ENVS = {
    "Arc/ArcAgi-v0": {
        "reason": "ARC-AGI grid: 0.05 for valid grid format, 0.0 for parse failure",
        "max_wrong_reward": 0.05,
        "max_empty_reward": 0.05,
    },
    "Arc/ReArc-v0": {
        "reason": "ReArc grid: 0.05 for valid grid format, 0.0 for parse failure",
        "max_wrong_reward": 0.05,
        "max_empty_reward": 0.05,
    },
    "Algorithmic/MonochromeBlockCounting-v0": {
        "reason": "Numeric answer: partial credit based on closeness",
        "max_wrong_reward": 0.99,
    },
    "Cognition/RectangleCount-v0": {
        "reason": "Numeric answer: partial credit based on closeness",
        "max_wrong_reward": 0.99,
    },
    "Geometry/ConvexHull-v0": {
        "reason": "Numeric answer: partial credit based on closeness",
        "max_wrong_reward": 0.99,
    },
    "Logic/Kakurasu-v0": {
        "reason": "Kakurasu: accepts alternative valid solutions",
        "max_wrong_reward": 1.0,
        "allow_alternative_solutions": True,
    },
    "Perception/FlowNetwork-v0": {
        "reason": "Flow network: accepts alternative valid solutions",
        "max_wrong_reward": 1.0,
        "allow_alternative_solutions": True,
    },
    "Graphs/MaximumClique-v0": {
        "reason": "Graph coloring: accepts alternative valid solutions",
        "max_wrong_reward": 1.0,
        "allow_alternative_solutions": True,
    },
    "Graphs/MinimumChromaticNumber-v0": {
        "reason": "Graph coloring: partial credit for partially correct coloring",
        "max_wrong_reward": 0.99,
    },
    "Perception/GraphToMST-v0": {
        "reason": "MST extraction: accepts alternative valid MSTs",
        "max_wrong_reward": 1.0,
        "allow_alternative_solutions": True,
    },
    "Puzzles/TicTacToe-QA-v0": {
        "reason": "TicTacToe QA: accepts alternative valid answers",
        "max_wrong_reward": 1.0,
        "allow_alternative_solutions": True,
    },
    "Logic/MiniSudoku-v0": {
        "reason": "4x4 grid, scores by correct cells / 16",
        "max_wrong_reward": 0.99,
    },
    "Algorithmic/RotateMatrix-v0": {
        "reason": "Substring matching: len(correct) / len(answer)",
        "max_wrong_reward": 0.99,
    },
    "Puzzles/Tsumego-v0": {
        "reason": "Go coordinates: 0.05 for valid format, 0.01 otherwise",
        "max_wrong_reward": 0.1,
    },
    "Logic/Thermometers-v0": {
        "reason": "Accepts any solution satisfying constraints",
        "max_wrong_reward": 1.0,
        "allow_alternative_solutions": True,
    },
    "Puzzles/TwiddlePuzzle-v0": {
        "reason": "Grid transformation puzzle: scores by (matching_cells / total_cells)^5",
        "max_wrong_reward": 0.99,
    },
    "Algorithmic/CirculatingGrid-v0": {
        "reason": "Optimization puzzle: minimizes cell changes",
        "max_wrong_reward": 1.0,
        "allow_alternative_solutions": True,
    },
    "Geometry/LargestRectangleAmongPoints-v0": {
        "reason": "Optimization puzzle: maximizes rectangle area",
        "max_wrong_reward": 1.0,
        "allow_alternative_solutions": True,
    },
    "Graphs/MaximumIndependentSetTree-v0": {
        "reason": "Optimization puzzle: maximizes total weight of independent set",
        "max_wrong_reward": 1.0,
        "allow_alternative_solutions": True,
    },
    "Graphs/MaximumWeightMatching-v0": {
        "reason": "Optimization puzzle: maximizes total weight of matching",
        "max_wrong_reward": 1.0,
        "allow_alternative_solutions": True,
    },
    "Graphs/WeightedBinarytree-v0": {
        "reason": "Optimization puzzle: maximizes binary tree score",
        "max_wrong_reward": 1.0,
        "allow_alternative_solutions": True,
    },
    "Puzzles/TetrisAttack-v0": {
        "reason": "Optimization puzzle: minimizes swaps to clear array",
        "max_wrong_reward": 1.0,
        "allow_alternative_solutions": True,
    },
}


class TestSingleTurnEnvironments(unittest.TestCase):
    """Unified test suite for all single-turn gym-v environments."""

    def _get_output_dir(self, env_id: str) -> Path:
        """Get output directory for environment test artifacts."""
        suite, name = env_id.split("/")
        name_clean = name.replace("-v0", "")
        snake_name = "".join(
            f"_{c.lower()}" if c.isupper() else c for c in name_clean
        ).lstrip("_")
        return (
            Path(__file__).resolve().parent
            / f"test_output_{suite.lower()}_{snake_name}"
        )

    def _setup_output_dir(self, output_dir: Path) -> None:
        """Create or clean output directory."""
        if output_dir.exists():
            for p in output_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

    def _get_oracle_answer(self, env: Any, info: dict[str, Any]) -> str:
        """Retrieve oracle answer from environment or info dict."""

        def _normalize_oracle(value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, str):
                return value
            try:
                import numpy as np

                if isinstance(value, np.generic):
                    return str(value.item())
            except Exception:
                pass
            if isinstance(value, list | tuple):
                if all(isinstance(row, list | tuple) for row in value):
                    return "\n".join(" ".join(map(str, row)) for row in value)
                return " ".join(map(str, value))
            return str(value)

        oracle = _normalize_oracle(info.get("oracle_answer"))
        if oracle:
            return oracle

        oracle = _normalize_oracle(info.get("reference_answer"))
        if oracle:
            return oracle

        try:
            oracle = env.get_wrapper_attr("_oracle_answer")
            oracle = _normalize_oracle(oracle)
            if oracle:
                return oracle
        except (AttributeError, KeyError):
            pass

        raise ValueError(f"Could not find oracle answer in env or info: {info.keys()}")

    def _perturb_answer(self, answer: str) -> str:
        """Create a perturbed version of the correct answer."""
        if not answer:
            return "WRONG"

        if len(answer) == 3 and answer[0] == "(" and answer[2] == ")":
            choices = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
            other_choices = [c for c in choices if c != answer]
            return random.choice(other_choices)

        answer_list = list(answer)
        num_changes = min(random.randint(1, 3), len(answer_list))

        for _ in range(num_changes):
            idx = random.randint(0, len(answer_list) - 1)
            answer_list[idx] = random.choice(string.ascii_letters + string.digits)

        perturbed = "".join(answer_list)

        if perturbed == answer:
            return answer + "_WRONG"

        return perturbed

    def _test_env(self, env_id: str, env_name: str) -> None:
        """Test a single environment comprehensively."""
        output_dir = self._get_output_dir(env_id)
        self._setup_output_dir(output_dir)

        test_seed = random.randint(0, 9999)
        print(f"\n[{env_id}] Using random seed: {test_seed}")

        env = gym_v.make(env_id)

        obs_dict, info_dict = env.reset(seed=test_seed)

        agent_id = "agent_0"
        self.assertIn(agent_id, obs_dict, f"{env_id}: agent_0 not in obs_dict")
        self.assertIn(agent_id, info_dict, f"{env_id}: agent_0 not in info_dict")

        obs: Observation = obs_dict[agent_id]
        info = info_dict[agent_id]

        self.assertIsNotNone(obs.image, f"{env_id}: obs.image is None")
        obs.image.save(output_dir / "0_reset.png")

        oracle = self._get_oracle_answer(env, info)
        self.assertIsInstance(oracle, str, f"{env_id}: oracle is not string")
        self.assertGreater(len(oracle), 0, f"{env_id}: oracle is empty")

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

        # --- correct answer → reward should be 1.0 (or >0 for alternative-solution envs) ---
        action_dict = {agent_id: oracle}
        _, reward_dict, terminated_dict, _, _ = env.step(action_dict)

        self.assertIn(agent_id, reward_dict)
        self.assertIn(agent_id, terminated_dict)
        self.assertTrue(
            terminated_dict[agent_id], f"{env_id}: not terminated after step"
        )
        self.assertIsInstance(reward_dict[agent_id], float)

        if env_id in PARTIAL_CREDIT_ENVS and PARTIAL_CREDIT_ENVS[env_id].get(
            "allow_alternative_solutions", False
        ):
            self.assertGreater(
                reward_dict[agent_id],
                0.0,
                f"{env_id}: Expected positive reward for valid solution, got {reward_dict[agent_id]}",
            )
        else:
            self.assertAlmostEqual(
                reward_dict[agent_id],
                1.0,
                places=6,
                msg=f"{env_id}: Expected reward 1.0 for correct answer, got {reward_dict[agent_id]}",
            )

        # --- empty answer → reward should be 0.0 ---
        obs_dict, info_dict = env.reset(seed=test_seed)
        action_empty = {agent_id: ""}
        _, reward_empty, terminated_empty, _, _ = env.step(action_empty)

        self.assertTrue(terminated_empty[agent_id])
        self.assertIsInstance(reward_empty[agent_id], float)

        if env_id in PARTIAL_CREDIT_ENVS and "max_empty_reward" in PARTIAL_CREDIT_ENVS[env_id]:
            max_empty = PARTIAL_CREDIT_ENVS[env_id]["max_empty_reward"]
            self.assertLessEqual(
                reward_empty[agent_id],
                max_empty,
                f"{env_id}: Empty answer reward {reward_empty[agent_id]} exceeds max {max_empty}",
            )
        else:
            self.assertEqual(
                reward_empty[agent_id],
                0.0,
                f"{env_id}: Expected reward 0.0 for empty answer, got {reward_empty[agent_id]}",
            )

        # --- perturbed answer → reward depends on partial-credit config ---
        obs_dict, info_dict = env.reset(seed=test_seed)
        info_reset = info_dict[agent_id]
        oracle_reset = self._get_oracle_answer(env, info_reset)

        perturbed = self._perturb_answer(oracle_reset)
        action_perturbed = {agent_id: perturbed}
        _, reward_perturbed, terminated_perturbed, _, _ = env.step(action_perturbed)

        self.assertTrue(terminated_perturbed[agent_id])
        self.assertIsInstance(reward_perturbed[agent_id], float)

        if env_id in PARTIAL_CREDIT_ENVS:
            config = PARTIAL_CREDIT_ENVS[env_id]
            max_allowed = config["max_wrong_reward"]
            allow_alternative_solutions = config.get(
                "allow_alternative_solutions", False
            )

            if not allow_alternative_solutions:
                self.assertLess(
                    reward_perturbed[agent_id],
                    1.0,
                    f"{env_id}: Perturbed answer should not get full credit (got {reward_perturbed[agent_id]})",
                )

            self.assertLessEqual(
                reward_perturbed[agent_id],
                max_allowed,
                f"{env_id}: Perturbed answer reward {reward_perturbed[agent_id]} exceeds max {max_allowed}. "
                f"Reason: {config['reason']}. Perturbed: '{perturbed}'",
            )
        else:
            self.assertEqual(
                reward_perturbed[agent_id],
                0.0,
                f"{env_id}: Expected reward 0.0 for perturbed answer '{perturbed}', got {reward_perturbed[agent_id]}",
            )

        # --- additional seeds (stability check) ---
        print(f"[{env_id}] Testing with 3 additional seeds...")
        for i in range(3):
            seed = random.randint(0, 9999)
            obs_test_dict, info_test_dict = env.reset(seed=seed)

            obs_test = obs_test_dict[agent_id]
            info_test = info_test_dict[agent_id]

            self.assertIsNotNone(obs_test.image, f"{env_id}: image None (seed={seed})")
            obs_test.image.save(output_dir / f"{i + 1}_seed_{seed}.png")

            oracle_test = self._get_oracle_answer(env, info_test)
            self.assertIsNotNone(oracle_test, f"{env_id}: oracle None (seed={seed})")
            self.assertIsInstance(oracle_test, str)
            self.assertGreater(len(oracle_test), 0)

            _, reward_test_dict, _, _, _ = env.step({agent_id: oracle_test})
            if env_id in PARTIAL_CREDIT_ENVS and PARTIAL_CREDIT_ENVS[env_id].get(
                "allow_alternative_solutions", False
            ):
                self.assertGreater(
                    reward_test_dict[agent_id],
                    0.0,
                    msg=f"{env_id}: Expected positive reward (seed={seed})",
                )
            else:
                self.assertAlmostEqual(
                    reward_test_dict[agent_id],
                    1.0,
                    places=6,
                    msg=f"{env_id}: Expected reward 1.0 (seed={seed})",
                )

            print(f"  Seed {seed}: Generated valid puzzle with oracle answer")

        env.close()
        print(f"[{env_id}]: All tests passed (primary_seed={test_seed})")


def _make_test_method(env_id: str, env_name: str):
    """Factory function to dynamically create test methods."""

    def test_method(self):
        self._test_env(env_id, env_name)

    test_method.__name__ = f"test_{env_name}"
    test_method.__doc__ = f"Test {env_id} environment."
    return test_method


for _env_id, _env_name in ALL_ENVS.items():
    _test_method = _make_test_method(_env_id, _env_name)
    setattr(TestSingleTurnEnvironments, _test_method.__name__, _test_method)


if __name__ == "__main__":
    unittest.main()

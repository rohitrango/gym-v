"""Unified tests for GameRL QA environments."""

from __future__ import annotations

from pathlib import Path
import random
import unittest

import gym_v

GAMERL_ENVS = {
    "GameRL/Snake-QA-v0": "snake",
    "GameRL/Maze-QA-v0": "maze",
    "GameRL/Maze3D-QA-v0": "maze_3d",
    "GameRL/Lifegame-QA-v0": "lifegame",
    "GameRL/Freecell-QA-v0": "freecell",
    "GameRL/Hue-QA-v0": "hue",
    "GameRL/Jewel2-QA-v0": "jewel2",
    "GameRL/LangtonAnt-QA-v0": "langton_ant",
    "GameRL/Minesweeper-QA-v0": "minesweeper",
    "GameRL/Minecraft-QA-v0": "minecraft",
    "GameRL/Sudoku-QA-v0": "sudoku",
    "GameRL/Pacman-QA-v0": "pacman",
    "GameRL/RhythmGame-QA-v0": "rhythm_game",
    "GameRL/RubiksCube-QA-v0": "rubiks_cube",
    "GameRL/Sokoban-QA-v0": "sokoban",
    "GameRL/SpiderSolitaire-QA-v0": "spider_solitaire",
    "GameRL/Tangram-QA-v0": "tangram",
    "GameRL/Tetris-QA-v0": "tetris",
    "GameRL/TicTacToe-QA-v0": "tictactoe",
    "GameRL/UltraTicTacToe-QA-v0": "ultra_tictactoe",
    "GameRL/TuringMachine2d-QA-v0": "turing_machine_2d",
    "GameRL/Tents-QA-v0": "tents",
    "GameRL/SpaceInvaders-QA-v0": "space_invaders",
    "GameRL/StarBattle-QA-v0": "star_battle",
    "GameRL/WordSearch-QA-v0": "word_search",
    "GameRL/Zuma-QA-v0": "zuma",
    "GameRL/3DReconstruction-QA-v0": "threed_reconstruction",
    "GameRL/ChessRanger-QA-v0": "chess_ranger",
    "GameRL/PyramidChess-QA-v0": "pyramidchess",
    "GameRL/Klondike-QA-v0": "klondike",
}


class TestGameRL(unittest.TestCase):
    """Test GameRL QA environments."""

    def _get_output_dir(self, env_id: str) -> Path:
        env_name = env_id.split("/")[1].replace("-v0", "")
        snake_name = "".join(
            f"_{c.lower()}" if c.isupper() else c for c in env_name
        ).lstrip("_")
        return Path(__file__).resolve().parent / f"test_output_gamerl_{snake_name}"

    def _setup_output_dir(self, output_dir: Path) -> None:
        if output_dir.exists():
            for p in output_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

    def _test_env(self, env_id: str, env_name: str) -> None:
        output_dir = self._get_output_dir(env_id)
        self._setup_output_dir(output_dir)

        test_seed = random.randint(0, 9999)
        print(f"\n[{env_id}] Using random seed: {test_seed}")

        env = gym_v.make(env_id)

        # 1. Reset
        obs_dict, info_dict = env.reset(seed=test_seed)

        # Check dictionary return structure
        self.assertIsInstance(
            obs_dict, dict, f"{env_id}: reset() should return dict of observations"
        )
        self.assertIsInstance(
            info_dict, dict, f"{env_id}: reset() should return dict of infos"
        )

        # Get agent_0 (default single player)
        agent_id = "agent_0"
        self.assertIn(agent_id, obs_dict)
        self.assertIn(agent_id, info_dict)

        obs = obs_dict[agent_id]
        info = info_dict[agent_id]

        self.assertIsNotNone(obs.image)
        obs.image.save(output_dir / "0_reset.png")

        oracle = info.get("oracle_answer")
        self.assertIsInstance(oracle, str)
        self.assertGreater(len(oracle), 0)

        print("\n" + "=" * 80)
        print(f"[{env_id}] SEED: {test_seed}")
        print(f"[{env_id}] DESCRIPTION:\n")
        print(env.description[:500] if len(env.description) > 500 else env.description)
        print(f"\n[{env_id}] OBS.TEXT:\n")
        text = obs.text or "No text"
        print(text[:500] if len(text) > 500 else text)
        print(f"\n[{env_id}] ORACLE ANSWER:\n")
        print(oracle[:300] + "..." if len(oracle) > 300 else oracle)
        print("=" * 80 + "\n")

        # 3. Verify reward with correct answer
        actions = {agent_id: oracle}
        _, reward_dict, terminated_dict, truncated_dict, _ = env.step(actions)

        self.assertIn(agent_id, reward_dict)
        self.assertIn(agent_id, terminated_dict)
        self.assertIn(agent_id, truncated_dict)

        self.assertTrue(terminated_dict[agent_id])
        self.assertTrue(truncated_dict[agent_id])
        self.assertIsInstance(reward_dict[agent_id], float)
        self.assertEqual(
            reward_dict[agent_id],
            1.0,
            f"{env_id}: Expected reward 1.0 for oracle answer",
        )

        # 4. Verify reward with wrong answer
        env.reset(seed=test_seed)
        actions_wrong = {agent_id: ""}
        _, reward_dict_wrong, terminated_dict_wrong, truncated_dict_wrong, _ = env.step(
            actions_wrong
        )

        self.assertTrue(terminated_dict_wrong[agent_id])
        self.assertTrue(truncated_dict_wrong[agent_id])
        self.assertIsInstance(reward_dict_wrong[agent_id], float)
        self.assertEqual(
            reward_dict_wrong[agent_id],
            0.0,
            f"{env_id}: Expected reward 0.0 for empty answer",
        )

        # 5. Test with multiple seeds
        print(f"[{env_id}] Testing with 3 additional seeds...")
        for i in range(3):
            seed = random.randint(0, 9999)
            obs_dict_test, info_dict_test = env.reset(seed=seed)
            info_test = info_dict_test[agent_id]
            obs_test = obs_dict_test[agent_id]

            oracle_test = info_test.get("oracle_answer")

            obs_test.image.save(output_dir / f"{i + 1}_seed_{seed}.png")

            self.assertIsNotNone(oracle_test)
            self.assertIsInstance(oracle_test, str)
            self.assertGreater(len(oracle_test), 0)

            _, reward_dict_test, _, _, _ = env.step({agent_id: oracle_test})
            self.assertEqual(
                reward_dict_test[agent_id],
                1.0,
                f"{env_id}: Expected reward 1.0 (seed={seed})",
            )
            print(f"  ✓ Seed {seed}: Generated valid puzzle with oracle answer")

        env.close()
        print(f"✅ {env_id}: All tests passed (primary_seed={test_seed})")


def _make_test_method(env_id: str, env_name: str):
    def test_method(self):
        self._test_env(env_id, env_name)

    test_method.__name__ = f"test_{env_name.lower()}"
    test_method.__doc__ = f"Test {env_id} environment."
    return test_method


for _env_id, _env_name in GAMERL_ENVS.items():
    _test_method = _make_test_method(_env_id, _env_name)
    setattr(TestGameRL, _test_method.__name__, _test_method)


if __name__ == "__main__":
    unittest.main()

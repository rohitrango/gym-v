"""Tests for Stable-Retro environment integration."""

from __future__ import annotations

from pathlib import Path
import unittest

import stable_retro

import gym_v
from gym_v.envs.stable_retro import RetroGymVEnv


def rom_available(game: str) -> bool:
    """Check if ROM file is available for a game."""
    try:
        stable_retro.data.get_romfile_path(game)
        return True
    except FileNotFoundError:
        return False


# Test game - Airstriker-Genesis is included in stable-retro data
TEST_GAME = "Airstriker-Genesis"
ROM_AVAILABLE = rom_available(TEST_GAME)


class TestRetroModuleImport(unittest.TestCase):
    """Test that the retro module imports correctly."""

    def test_module_import(self) -> None:
        """Test that RetroGymVEnv can be imported."""
        self.assertTrue(RetroGymVEnv)

    def test_env_registered(self) -> None:
        """Test that Retro environments are registered in gym_v."""
        self.assertIn("Retro/Airstriker-v0", gym_v.registry)


@unittest.skipUnless(ROM_AVAILABLE, f"ROM not available for {TEST_GAME}")
class TestRetroIntegration(unittest.TestCase):
    """Test Stable-Retro environment integration."""

    def _get_output_dir(self) -> Path:
        return Path(__file__).resolve().parent / "test_output_retro"

    def _setup_output_dir(self, output_dir: Path) -> None:
        if output_dir.exists():
            for p in output_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

    def test_retro_env_direct_import(self) -> None:
        """Test direct import and instantiation of RetroGymVEnv."""
        env = RetroGymVEnv(game=TEST_GAME)

        # Check basic properties
        self.assertIsNotNone(env.buttons)
        self.assertIsNotNone(env.available_actions)
        self.assertTrue(len(env.available_actions) > 0)

        print(f"Available buttons: {env.available_actions}")

        env.close()

    def test_retro_env_reset(self) -> None:
        """Test reset functionality."""
        env = RetroGymVEnv(game=TEST_GAME)
        obs_dict, info_dict = env.reset(seed=42)

        # Check observation structure
        self.assertIn("agent_0", obs_dict)
        obs = obs_dict["agent_0"]

        # Check Observation has image and text
        self.assertIsNotNone(obs.image)
        self.assertIsNotNone(obs.text)

        # Check image is a PIL Image
        from PIL import Image

        self.assertIsInstance(obs.image, Image.Image)

        print(f"Image size: {obs.image.size}")
        print(f"Observation text: {obs.text}")

        env.close()

    def test_retro_env_step(self) -> None:
        """Test step functionality with various actions."""
        env = RetroGymVEnv(game=TEST_GAME)
        env.reset(seed=42)

        # Test single button actions
        actions_to_test = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "NOOP"]

        for action in actions_to_test:
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = (
                env.step({"agent_0": action})
            )

            self.assertIn("agent_0", obs_dict)
            self.assertIn("agent_0", reward_dict)
            self.assertIn("__all__", terminated_dict)

            if terminated_dict["__all__"]:
                env.reset()

        # Test combined button actions
        combined_actions = ["A+UP", "B+DOWN", "LEFT+A"]
        for action in combined_actions:
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = (
                env.step({"agent_0": action})
            )

            self.assertIn("agent_0", obs_dict)
            if terminated_dict["__all__"]:
                env.reset()

        env.close()

    def test_retro_env_render_and_save(self) -> None:
        """Test rendering and saving frames."""
        output_dir = self._get_output_dir()
        self._setup_output_dir(output_dir)

        env = RetroGymVEnv(game=TEST_GAME)
        obs_dict, _ = env.reset(seed=42)

        # Save initial frame
        obs = obs_dict["agent_0"]
        if obs.image is not None:
            obs.image.save(output_dir / "frame_0_reset.png")

        # Take a few steps and save frames
        for i in range(5):
            obs_dict, _, terminated, _, _ = env.step({"agent_0": "A"})
            obs = obs_dict["agent_0"]
            if obs.image is not None:
                obs.image.save(output_dir / f"frame_{i + 1}_step.png")
            if terminated["__all__"]:
                break

        # Verify files were created
        saved_files = list(output_dir.glob("*.png"))
        self.assertTrue(len(saved_files) > 0, "No frames were saved")
        print(f"Saved {len(saved_files)} frames to {output_dir}")

        env.close()

    def test_retro_env_description(self) -> None:
        """Test environment description property."""
        env = RetroGymVEnv(game=TEST_GAME)

        description = env.description
        self.assertIsInstance(description, str)
        self.assertIn(TEST_GAME, description)
        self.assertTrue(len(description) > 0)

        print(f"Description: {description}")

        env.close()

    def test_retro_env_via_make(self) -> None:
        """Test creating environment via gym_v.make."""
        env = gym_v.make("Retro/Airstriker-v0")

        obs_dict, _ = env.reset(seed=42)
        self.assertIn("agent_0", obs_dict)

        # Take a step
        obs_dict, reward_dict, terminated, truncated, info = env.step({"agent_0": "A"})
        self.assertIn("agent_0", obs_dict)

        env.close()


@unittest.skipUnless(ROM_AVAILABLE, f"ROM not available for {TEST_GAME}")
class TestRetroActionMapping(unittest.TestCase):
    """Test action string to button mask conversion."""

    def test_single_button_actions(self) -> None:
        """Test single button action parsing."""
        env = RetroGymVEnv(game=TEST_GAME)

        # Test that valid buttons create non-zero masks
        for button in env.available_actions:
            mask = env._action_to_mask(button)
            self.assertEqual(
                mask.sum(), 1, f"Button {button} should activate exactly 1 button"
            )

        env.close()

    def test_combined_button_actions(self) -> None:
        """Test combined button action parsing."""
        env = RetroGymVEnv(game=TEST_GAME)

        # Test combined actions
        mask = env._action_to_mask("A+UP")
        self.assertEqual(mask.sum(), 2, "A+UP should activate 2 buttons")

        mask = env._action_to_mask("B+DOWN+LEFT")
        self.assertEqual(mask.sum(), 3, "B+DOWN+LEFT should activate 3 buttons")

        env.close()

    def test_noop_actions(self) -> None:
        """Test NOOP action parsing."""
        env = RetroGymVEnv(game=TEST_GAME)

        for noop in ["NOOP", "NONE", ""]:
            mask = env._action_to_mask(noop)
            self.assertEqual(mask.sum(), 0, f"{noop!r} should not activate any buttons")

        env.close()

    def test_case_insensitivity(self) -> None:
        """Test that action parsing is case-insensitive."""
        env = RetroGymVEnv(game=TEST_GAME)

        # Test various cases
        mask_upper = env._action_to_mask("UP")
        mask_lower = env._action_to_mask("up")
        mask_mixed = env._action_to_mask("Up")

        self.assertTrue((mask_upper == mask_lower).all())
        self.assertTrue((mask_upper == mask_mixed).all())

        env.close()


if __name__ == "__main__":
    unittest.main()

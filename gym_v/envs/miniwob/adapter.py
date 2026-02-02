"""Minimal adapter for BrowserGym MiniWoB environments."""
import gymnasium as gym
import browsergym.miniwob
from browsergym.core.action.highlevel import HighLevelActionSet
from PIL import Image

from gym_v import Env, Observation


class MiniWoBAdapter(Env):
    """Lightweight adapter for BrowserGym MiniWoB tasks.

    This adapter converts BrowserGym's MiniWoB environments to the gym-v interface.
    It handles observation conversion and action mapping while keeping the code minimal.

    Args:
        task_name: Name of the MiniWoB task (e.g., "click-test", "login-user")
        headless: Whether to run browser in headless mode
        action_subsets: List of action subsets to enable (default: ["bid"])
            - "bid": Browser ID actions (click, fill, etc. by element ID)
            - "coord": Coordinate actions (mouse_click, keyboard_type, etc.)
            - "nav": Navigation actions (goto, go_back, go_forward)
            - "tab": Tab management actions
        max_episode_steps: Maximum steps per episode
        num_players: Number of players (always 1 for MiniWoB)
    """

    def __init__(
        self,
        task_name: str,
        headless: bool = True,
        action_subsets: list[str] | None = None,
        max_episode_steps: int | None = None,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(max_episode_steps=max_episode_steps, **kwargs)

        if num_players != 1:
            raise ValueError(f"MiniWoB only supports 1 player, got {num_players}")

        action_subsets = action_subsets or ["bid"]
        self._action_set = HighLevelActionSet(subsets=action_subsets, multiaction=True)
        self._task_name = task_name

        self._env = gym.make(
            f"browsergym/miniwob.{task_name}",
            action_mapping=self._action_set.to_python_code,
            headless=headless,
        )
        self._agent_ids = {"agent_0"}
        self.num_players = num_players
        self._current_goal = ""  # Store current goal for description

    @property
    def description(self) -> str:
        """Return task description and available actions.

        This provides static information about:
        1. Task name and type
        2. Available actions with signatures and examples
        3. How to reference and interact with elements
        """
        # Get action descriptions from HighLevelActionSet
        action_desc = self._action_set.describe(
            with_long_description=False, with_examples=True
        )

        goal_text = f"Goal: {self._current_goal}" if self._current_goal else "Goal: (will be shown after reset)"

        return f"""MiniWoB Task: {self._task_name}

You are interacting with a web page to complete a task.

{goal_text}

Available Actions:
{action_desc}

Notes:
- Elements are identified by 'bid' (browser ID) attributes
- The observation text shows all visible element IDs you can interact with
- Use these element IDs in your actions
- Actions use Python function call syntax
- Multiple actions can be chained with newlines

Example action sequences:
  click('a5')
  fill('username', 'john')
  fill('password', 'secret123')
  click('login_button')
"""

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        # Update current goal for description
        self._current_goal = obs.get("goal", obs.get("last_action_error", ""))
        return {"agent_0": self._to_gym_v_obs(obs)}, info

    def inner_step(self, actions):
        """Execute one step in the environment.

        This implements the actual step logic while the base class `step` method
        handles episode step counting and automatic truncation.
        """
        action = actions["agent_0"]
        obs, reward, terminated, truncated, info = self._env.step(action)

        # Update current goal for description
        self._current_goal = obs.get("goal", obs.get("last_action_error", ""))

        agent_id = "agent_0"
        return (
            {agent_id: self._to_gym_v_obs(obs)},
            {agent_id: reward},
            {agent_id: terminated, "__all__": terminated},
            {agent_id: truncated, "__all__": truncated},
            {agent_id: info},
        )

    def close(self):
        if hasattr(self, "_env"):
            self._env.close()

    def _to_gym_v_obs(self, bg_obs) -> Observation:
        """Convert BrowserGym observation to gym-v Observation format.

        Returns observation with:
        - image: Screenshot of the web page (includes goal text at top)
        - text: List of all visible element IDs
        - metadata: Raw BrowserGym observation (includes goal)
        """
        # Extract screenshot from numpy array
        image = Image.fromarray(bg_obs["screenshot"])

        # Build observation text with ALL visible element IDs (no truncation)
        obs_text = ""
        if "extra_element_properties" in bg_obs:
            visible_elements = []
            for bid, props in bg_obs["extra_element_properties"].items():
                if props.get("visibility", 0) > 0.5:  # Only visible elements
                    visible_elements.append(bid)

            if visible_elements:
                obs_text = f"Visible elements: {', '.join(visible_elements)}"
            else:
                obs_text = "Visible elements: (none)"
        else:
            obs_text = "Visible elements: (none)"

        return Observation(
            image=image,
            text=obs_text,
            metadata={"raw_obs": bg_obs},
        )

from __future__ import annotations

from typing import Any

from PIL import Image

import gym_v
from gym_v import Env, Observation


class TextFeedbackEnv(Env):
    def __init__(self, *, image_on_step: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        self._image_on_step = image_on_step

    @property
    def description(self) -> str:
        return "A test environment with textual step feedback."

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        image = Image.new("RGB", (2, 2))
        return {"agent_0": Observation(image=image, text="reset text")}, {}

    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        image = Image.new("RGB", (2, 2)) if self._image_on_step else None
        obs = Observation(image=image, text="step text feedback")
        return (
            {"agent_0": obs},
            {"agent_0": 0.0},
            {"agent_0": False, "__all__": False},
            {"agent_0": False, "__all__": False},
            {"agent_0": {}},
        )

    def render(self) -> Image.Image | list[Image.Image] | None:
        return Image.new("RGB", (2, 2))


if "Testing/TextFeedback-v0" not in gym_v.registry:
    gym_v.register(
        id="Testing/TextFeedback-v0",
        entry_point=TextFeedbackEnv,
        disable_env_checker=True,
    )


def test_disable_text_feedback_strips_step_text_but_keeps_reset_text():
    env = gym_v.make("Testing/TextFeedback-v0", disable_text_feedback=True)
    reset_obs, _ = env.reset()
    step_obs, _, _, _, _ = env.step({"agent_0": "action"})

    assert reset_obs["agent_0"].text == "reset text"
    assert step_obs["agent_0"].image is not None
    assert step_obs["agent_0"].text is None


def test_disable_text_feedback_keeps_text_only_observations_valid():
    env = gym_v.make(
        "Testing/TextFeedback-v0",
        disable_text_feedback=True,
        image_on_step=False,
    )
    env.reset()
    step_obs, _, _, _, _ = env.step({"agent_0": "action"})

    assert step_obs["agent_0"].image is None
    assert step_obs["agent_0"].text == ""


def test_text_feedback_remains_by_default():
    env = gym_v.make("Testing/TextFeedback-v0")
    env.reset()
    step_obs, _, _, _, _ = env.step({"agent_0": "action"})

    assert step_obs["agent_0"].text == "step text feedback"

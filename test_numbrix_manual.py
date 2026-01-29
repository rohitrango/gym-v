#!/usr/bin/env python3
"""Manual test for RLVE Numbrix environment."""

import gym_v

def test_numbrix():
    print("Creating RLVE/Numbrix-v0 environment...")
    env = gym_v.make("RLVE/Numbrix-v0")

    print("\nResetting environment with seed 42...")
    obs_dict, info_dict = env.reset(seed=42)

    agent_id = "agent_0"
    obs = obs_dict[agent_id]
    info = info_dict[agent_id]

    print(f"\nEnvironment description:\n{env.description}")

    print(f"\nObservation text:\n{obs.text[:500] if obs.text else 'None'}")

    reference_answer = info.get("reference_answer")
    print(f"\nReference answer:\n{reference_answer}")

    # Save the image
    if obs.image:
        obs.image.save("/tmp/numbrix_test.png")
        print("\nImage saved to /tmp/numbrix_test.png")

    # Test with correct answer
    print("\nTesting with correct answer...")
    _, reward_dict, terminated_dict, _, _ = env.step({agent_id: reference_answer})
    print(f"Reward: {reward_dict[agent_id]}")
    print(f"Terminated: {terminated_dict[agent_id]}")

    # Reset and test with wrong answer
    print("\nResetting and testing with wrong answer...")
    env.reset(seed=42)
    _, reward_dict, terminated_dict, _, _ = env.step({agent_id: "0 1\n2 3"})
    print(f"Reward for wrong answer: {reward_dict[agent_id]}")

    # Reset and test with empty answer
    print("\nResetting and testing with empty answer...")
    env.reset(seed=42)
    _, reward_dict, terminated_dict, _, _ = env.step({agent_id: ""})
    print(f"Reward for empty answer: {reward_dict[agent_id]}")

    print("\n✅ All manual tests passed!")
    env.close()

if __name__ == "__main__":
    test_numbrix()

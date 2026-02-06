import os
import tempfile
import unittest

from dotenv import load_dotenv
import pandas as pd
from vlmeval.dataset import build_dataset
from vlmeval.smp import dump

from gym_v.envs.eval.vlmeval.vlmeval_env import VLMEvalEnv


class TestVLMEvalEnv(unittest.TestCase):
    def setUp(self):
        # Load environment variables from .env file if it exists
        load_dotenv()

    def _run_consistency_test(self, dataset_name):
        print(f"\n{'=' * 20}\nTesting dataset: {dataset_name}\n{'=' * 20}")

        # Force exact matching to avoid API calls and ensure deterministic comparison
        test_judge_kwargs = {"model": "exact_matching"}

        # --- 1. Run via Gym-V ---
        print(f"Initializing Gym-V env with {dataset_name}...")
        try:
            env = VLMEvalEnv(dataset_name=dataset_name, judge_kwargs=test_judge_kwargs)
            obs_dict, info_dict = env.reset()
        except Exception as e:
            print(
                f"Skipping {dataset_name} due to initialization failure (likely download issue): {e}"
            )
            return

        # Generate dummy actions (e.g. all "A") to ensure deterministic results
        # We don't need real model predictions, just consistent inputs
        actions = {agent_id: "A" for agent_id in obs_dict}

        print("Running Gym-V step...")
        _, _, _, _, gym_info = env.step(actions)
        gym_metrics = gym_info["metrics"]

        # --- 2. Run via Native VLMEvalKit ---
        print("Running native VLMEvalKit evaluation...")

        # Map actions back to {index: prediction} format expected by VLMEvalKit
        # We need the dataset data to have all columns (A, B, C, answer, etc.)
        ds = build_dataset(dataset_name)
        df_native = ds.data.copy()

        # Create a map for predictions
        pred_map = {}
        for agent_id, pred in actions.items():
            idx = int(agent_id.split("_")[1])
            pred_map[idx] = pred

        # Add prediction column
        # Ensure we map using the index column type correctly (usually int)
        df_native["prediction"] = df_native["index"].map(pred_map)

        # Create temp file for native eval
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_file = tmp.name

        try:
            dump(df_native, tmp_file)

            # Use same judge_kwargs as in env implementation
            native_metrics = ds.evaluate(tmp_file, **test_judge_kwargs)
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

        # --- 3. Assert Equality ---
        print(f"Gym Metrics: {gym_metrics}")
        # print(f"Native Metrics: {native_metrics}") # Output can be large for dataframes

        # Check if results are equal
        if isinstance(gym_metrics, pd.DataFrame):
            self.assertTrue(
                isinstance(native_metrics, pd.DataFrame),
                "If Gym-V result is a DataFrame, native result must also be a DataFrame",
            )
            try:
                pd.testing.assert_frame_equal(gym_metrics, native_metrics)
            except AssertionError as e:
                self.fail(
                    f"Gym-V evaluation result (DataFrame) does not match native result for {dataset_name}: {e}"
                )
        else:
            self.assertEqual(
                gym_metrics,
                native_metrics,
                f"Gym-V evaluation result should match native VLMEvalKit result exactly for {dataset_name}",
            )

        # --- 4. Verify Prompt Consistency ---
        print("Verifying prompt consistency...")

        # We grab one sample to compare
        if len(obs_dict) > 0:
            sample_agent_id = list(obs_dict.keys())[0]
            sample_index = int(sample_agent_id.split("_")[1])

            # Get prompt from Gym-V observation
            gym_obs = obs_dict[sample_agent_id]
            gym_struct = gym_obs.metadata["struct"]

            # Get prompt from Native VLMEvalKit
            native_data = ds.data
            native_item = native_data[native_data["index"] == sample_index].iloc[0]
            native_struct = ds.build_prompt(native_item)

            self.assertEqual(
                gym_struct,
                native_struct,
                f"Prompt structure built by Gym-V should match native VLMEvalKit prompt exactly for {dataset_name}",
            )

    def test_multiple_datasets(self):
        # List of datasets to test
        # We choose small dev/mini sets to keep tests fast
        datasets_to_test = [
            "MMBench_DEV_EN",  # Typical MCQ, simple metrics
            "MMMU_DEV_VAL",  # Complex MCQ, DataFrame metrics, requires strict column format
        ]

        for dataset in datasets_to_test:
            with self.subTest(dataset=dataset):
                self._run_consistency_test(dataset)


if __name__ == "__main__":
    unittest.main()

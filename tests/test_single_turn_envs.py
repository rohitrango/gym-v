"""Tests for all single-turn gym-v environments.

For every registered single-turn environment and 5 random seeds:
  1. oracle (ground truth) answer  → reward == 1.0
  2. empty answer ("")             → reward == 0.0
  3. perturbed answer              → reward < 1.0
Images are saved to tests/test_images/<env_id>/.
"""

from __future__ import annotations

import json
import random
import re
import string
from pathlib import Path

import pytest

import gym_v

# ---------------------------------------------------------------------------
# All single-turn environment IDs
# ---------------------------------------------------------------------------

ALL_ENV_IDS = [
    # Algorithmic
    "Algorithmic/AdditionTable-v0",
    "Algorithmic/BinaryMatrix-v0",
    "Algorithmic/CirculatingGrid-v0",
    "Algorithmic/CoinSquareGame-v0",
    "Algorithmic/FaceRightWay-v0",
    "Algorithmic/GameOfLife-v0",
    "Algorithmic/GraMinimaGame-v0",
    "Algorithmic/GridBFS-v0",
    "Algorithmic/GridLocalMinimumCounting-v0",
    "Algorithmic/LandformGenerationCounting-v0",
    "Algorithmic/LangtonAnt-QA-v0",
    "Algorithmic/Lifegame-QA-v0",
    "Algorithmic/MatrixPermutationBothDiagonalOne-v0",
    "Algorithmic/MatrixPermutationMainDiagonalOne-v0",
    "Algorithmic/MaxGridPathIntersection-v0",
    "Algorithmic/MonochromeBlockCounting-v0",
    "Algorithmic/RotateMatrix-v0",
    "Algorithmic/RottenOranges-v0",
    "Algorithmic/SpiralMatrix-v0",
    "Algorithmic/StoneIntervalsGame-v0",
    "Algorithmic/TuringMachine2d-QA-v0",
    # Arc
    "Arc/Arc1D-v0",
    "Arc/ArcAgi-v0",
    "Arc/ReArc-v0",
    # Cognition
    "Cognition/Hue-QA-v0",
    "Cognition/Maze3D-QA-v0",
    "Cognition/OddOneOutPoly-v0",
    "Cognition/RectangleCount-v0",
    "Cognition/RubiksCube-QA-v0",
    "Cognition/SequenceCompletionPoly-v0",
    "Cognition/SymmetryFillPoly-v0",
    "Cognition/TransformResultPoly-v0",
    # Geometry
    "Geometry/ConvexHull-v0",
    "Geometry/LargestRectangleAmongPoints-v0",
    "Geometry/SkaRockGarden-v0",
    "Geometry/SmallestCircle-v0",
    "Geometry/SumTriangleArea-v0",
    "Geometry/Tangram-QA-v0",
    "Geometry/VisibleLine-v0",
    # Graphs
    "Graphs/FbiBinaryTree-v0",
    "Graphs/GraphContainTreeCounting-v0",
    "Graphs/GraphIsomorphism-v0",
    "Graphs/GridComponent-v0",
    "Graphs/HamiltonianPath-v0",
    "Geometry/LargestIsland-v0",
    "Graphs/LongestPath-v0",
    "Graphs/MaximumAchromaticNumber-v0",
    "Graphs/MaximumClique-v0",
    "Graphs/MaximumIndependentSetGrid-v0",
    "Graphs/MaximumIndependentSetTree-v0",
    "Graphs/MaximumWeightMatching-v0",
    "Graphs/MinimumChromaticNumber-v0",
    "Graphs/MinimumDirectedSpanningTree-v0",
    "Graphs/MinimumSpanningTreeCounting-v0",
    "Graphs/MixedGraphEulerianCircuit-v0",
    "Graphs/Patrol-v0",
    "Graphs/ShortestPath-v0",
    "Graphs/TreeCenter-v0",
    "Graphs/TreeChangeOneEdgeDiameter-v0",
    "Graphs/TreeColoring-v0",
    "Graphs/TreeDistanceEqualTriadCounting-v0",
    "Graphs/TreeEvenPartitioning-v0",
    "Graphs/TreeTopologicalSequenceCounting-v0",
    "Graphs/WeightedBinarytree-v0",
    # Logic
    "Logic/Binairo-v0",
    "Logic/BinarioNoAdjacencyRequirement-v0",
    "Logic/CampsitePuzzle-v0",
    "Logic/CircuitLogic-v0",
    "Logic/Futoshiki-v0",
    "Logic/GridParityConstruction-v0",
    "Logic/Kakurasu-v0",
    "Logic/MagicSquarePuzzle-v0",
    "Logic/MiniSudoku-v0",
    "Logic/NQueens-v0",
    "Logic/Numbrix-v0",
    "Logic/Renzoku-v0",
    "Logic/SkyscraperPuzzle-v0",
    "Logic/StarBattle-QA-v0",
    "Logic/Survo-v0",
    "Logic/Tents-QA-v0",
    "Logic/Thermometers-v0",
    # Perception
    "Perception/3DReconstruction-QA-v0",
    "Perception/ChartToTable-v0",
    "Perception/ContourPlot-v0",
    "Perception/DAGToTopoOrder-v0",
    "Perception/FlowNetwork-v0",
    "Perception/FunctionGraph-v0",
    "Perception/GraphToAdjacency-v0",
    "Perception/GraphToMST-v0",
    "Perception/ParametricCurve-v0",
    "Perception/PolarPlot-v0",
    "Perception/TreeToTraversal-v0",
    "Perception/VectorField-v0",
    # Puzzles
    "Puzzles/ChessRanger-QA-v0",
    "Puzzles/EightDigitPuzzle-v0",
    "Puzzles/Freecell-QA-v0",
    "Puzzles/Jewel2-QA-v0",
    "Puzzles/KloBlocks-v0",
    "Puzzles/KnightSwap-v0",
    "Puzzles/Maze-QA-v0",
    "Puzzles/NinePuzzle-v0",
    "Puzzles/Pacman-QA-v0",
    "Puzzles/PyramidChess-QA-v0",
    "Puzzles/RhythmGame-QA-v0",
    "Puzzles/Snake-QA-v0",
    "Puzzles/SpaceInvaders-QA-v0",
    "Puzzles/SpiderSolitaire-QA-v0",
    "Puzzles/Tetris-QA-v0",
    "Puzzles/TetrisAttack-v0",
    "Puzzles/TicTacToe-QA-v0",
    "Puzzles/TowerOfHanoi-v0",
    "Puzzles/Tsumego-v0",
    "Puzzles/TwiddlePuzzle-v0",
    "Puzzles/UltraTicTacToe-QA-v0",
    "Puzzles/WordSearch-QA-v0",
    "Puzzles/Zuma-QA-v0",
]

NUM_SEEDS = 10

IMAGE_DIR = Path(__file__).resolve().parent / "test_images"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_oracle(env, info: dict) -> str:
    """Extract oracle answer from info dict or env attribute."""
    for key in ("oracle_answer", "reference_answer"):
        val = info.get(key)
        if val is not None:
            if isinstance(val, str):
                return val
            if isinstance(val, list | tuple):
                if all(isinstance(row, list | tuple) for row in val):
                    return "\n".join(" ".join(map(str, row)) for row in val)
                return " ".join(map(str, val))
            return str(val)
    try:
        val = env.get_wrapper_attr("_oracle_answer")
        if val is not None:
            return str(val)
    except (AttributeError, KeyError):
        pass
    raise ValueError(f"No oracle answer found (info keys: {list(info.keys())})")


def _perturb(answer: str, rng: random.Random) -> str:
    """Create a semantically perturbed answer that should score < 1.0.

    Unlike naive character-level mutation, this understands common answer
    formats (JSON, multiple-choice, numbers, grids) and perturbs them in ways
    that survive normalisation (case-folding, whitespace stripping, partial-
    field checking).
    """
    if not answer:
        return "WRONG_ANSWER"
    stripped = answer.strip()

    # ---- JSON object: change every numeric value ----
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return _perturb_json_obj(obj, rng)
    except (json.JSONDecodeError, ValueError):
        pass

    # ---- Multiple-choice like (a), (b), ..., (h) ----
    mc = re.fullmatch(r"\(([a-zA-Z])\)", stripped)
    if mc:
        letter = mc.group(1).lower()
        choices = [c for c in "abcdefgh" if c != letter]
        return f"({rng.choice(choices)})"

    # ---- Coordinate like (7, 3) ----
    coord = re.match(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", stripped)
    if coord:
        x, y = int(coord.group(1)), int(coord.group(2))
        return f"({x + 999}, {y + 999})"

    # ---- Single character ----
    if len(stripped) == 1:
        return _perturb_single_char(stripped, rng)

    # ---- Multi-line grid (rows of space-separated tokens) ----
    lines = stripped.split("\n")
    if len(lines) > 1:
        return _perturb_grid(lines, rng)

    # ---- Space-separated number list ----
    tokens = stripped.split()
    if len(tokens) > 1 and all(_looks_numeric(t) for t in tokens):
        return _perturb_number_list(tokens, rng)

    # ---- Single number ----
    if _looks_numeric(stripped):
        return _perturb_number(stripped, rng)

    # ---- Fallback: aggressive mutation ----
    return _perturb_aggressive(stripped, rng)


def _perturb_json_obj(obj: dict, rng: random.Random) -> str:
    """Perturb all numeric values in a JSON dict."""
    new = {}
    for k, v in obj.items():
        if isinstance(v, (int, float)):
            new[k] = v + rng.choice([-999, -100, -10, 10, 100, 999])
        elif isinstance(v, list):
            new[k] = [_perturb_json_value(item, rng) for item in v]
        elif isinstance(v, str):
            new[k] = v + "_WRONG"
        else:
            new[k] = v
    return json.dumps(new)


def _perturb_json_value(val, rng: random.Random):
    """Perturb a single value inside a JSON structure."""
    if isinstance(val, (int, float)):
        return val + rng.choice([-999, -100, -10, 10, 100, 999])
    if isinstance(val, str):
        return val + "_WRONG"
    if isinstance(val, list):
        return [_perturb_json_value(item, rng) for item in val]
    return val


def _perturb_single_char(ch: str, rng: random.Random) -> str:
    """Perturb a single character to a guaranteed-wrong value.

    Returns a string with no letters (avoids matching choice-scoring regexes)
    and no valid numbers (avoids matching numeric scoring). The '###' format
    is unparseable by JSON, coordinate, choice, and number scorers.
    """
    return "###"


def _perturb_grid(lines: list[str], rng: random.Random) -> str:
    """Perturb a multi-line grid answer by flipping values in every row."""
    new_lines = []
    for line in lines:
        tokens = line.strip().split()
        if tokens:
            idx = rng.randint(0, len(tokens) - 1)
            tok = tokens[idx]
            if _looks_numeric(tok):
                tokens[idx] = str(_flip_number(tok, rng))
            else:
                tokens[idx] = _flip_token(tok)
            new_lines.append(" ".join(tokens))
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def _flip_token(tok: str) -> str:
    """Flip a non-numeric grid token to a guaranteed-different value."""
    low = tok.strip().lower()
    if low == "e":
        return "s"
    if low == "s":
        return "e"
    if low in ("0", "false", "no", "n"):
        return "1"
    if low in ("1", "true", "yes", "y"):
        return "0"
    return "###"


def _perturb_number_list(tokens: list[str], rng: random.Random) -> str:
    """Perturb a space-separated list of numbers.

    Changes values AND appends an extra element to break length-based
    validation (e.g. coloring problems where relabelled colors are equivalent).
    """
    tokens = list(tokens)
    n_change = max(1, len(tokens) // 2)
    indices = rng.sample(range(len(tokens)), min(n_change, len(tokens)))
    for i in indices:
        tokens[i] = str(_flip_number(tokens[i], rng))
    tokens.append("999")
    return " ".join(tokens)


def _perturb_number(s: str, rng: random.Random) -> str:
    """Perturb a single number string."""
    return str(_flip_number(s, rng))


def _flip_number(s: str, rng: random.Random):
    """Change a numeric string to a significantly different value."""
    try:
        if "." in s:
            v = float(s)
            decimals = len(s.split(".")[-1])
            return round(v + rng.choice([-100.0, -10.0, 10.0, 100.0]), decimals)
        else:
            v = int(s)
            return v + rng.choice([-999, -100, -10, 10, 100, 999])
    except ValueError:
        return s + "_WRONG"


def _perturb_aggressive(s: str, rng: random.Random) -> str:
    """Aggressively mutate a string to be clearly wrong."""
    if len(s) <= 3:
        return "WRONG_" + s
    chars = list(s)
    n = max(len(chars) // 3, 2)
    for _ in range(n):
        i = rng.randint(0, len(chars) - 1)
        if chars[i].isdigit():
            chars[i] = rng.choice(string.ascii_uppercase)
        elif chars[i].isalpha():
            chars[i] = rng.choice(string.digits)
        else:
            chars[i] = rng.choice(string.digits)
    result = "".join(chars)
    return result if result.lower() != s.lower() else "WRONG_" + s


def _looks_numeric(s: str) -> bool:
    """Check if string looks like a number (int or float)."""
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id", ALL_ENV_IDS)
def test_single_turn_env(env_id: str):
    """Test a single-turn environment across 5 random seeds."""

    # Prepare image output dir
    safe_name = env_id.replace("/", "_").replace("-", "_")
    img_dir = IMAGE_DIR / safe_name
    img_dir.mkdir(parents=True, exist_ok=True)

    env = gym_v.make(env_id)
    rng = random.Random()
    seeds = [rng.randint(0, 99999) for _ in range(NUM_SEEDS)]

    for seed in seeds:
        # ---- reset & save image ----
        obs_dict, info_dict = env.reset(seed=seed)
        obs = obs_dict["agent_0"]
        info = info_dict["agent_0"]
        assert obs.image is not None, f"[{env_id} seed={seed}] obs.image is None"
        obs.image.save(img_dir / f"seed_{seed}.png")

        oracle = _get_oracle(env, info)
        assert isinstance(oracle, str) and len(oracle) > 0, (
            f"[{env_id} seed={seed}] oracle is empty or not a string"
        )

        # ---- 1. oracle answer → reward == 1.0 ----
        _, rwd, term, _, _ = env.step({"agent_0": oracle})
        r_oracle = rwd["agent_0"]
        assert term["agent_0"], f"[{env_id} seed={seed}] not terminated after step"
        assert r_oracle == pytest.approx(1.0, abs=1e-6), (
            f"[{env_id} seed={seed}] oracle reward should be 1.0, "
            f"got {r_oracle}. oracle='{oracle}'"
        )

        # ---- 2. empty answer → reward == 0.0 ----
        env.reset(seed=seed)
        _, rwd_e, _, _, _ = env.step({"agent_0": ""})
        r_empty = rwd_e["agent_0"]
        assert r_empty == 0.0, (
            f"[{env_id} seed={seed}] empty answer reward should be 0.0, "
            f"got {r_empty}"
        )

        # ---- 3. perturbed answer → reward < 1.0 ----
        _, info_p = env.reset(seed=seed)
        oracle_p = _get_oracle(env, info_p["agent_0"])
        perturbed = _perturb(oracle_p, rng)
        _, rwd_p, _, _, _ = env.step({"agent_0": perturbed})
        r_perturbed = rwd_p["agent_0"]
        assert r_perturbed < 1.0, (
            f"[{env_id} seed={seed}] perturbed reward should be < 1.0, "
            f"got {r_perturbed}. "
            f"oracle='{oracle}', perturbed='{perturbed}'"
        )

    env.close()

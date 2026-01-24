"""Generate HTML gallery for all RLVE environments in PR #44."""

import base64
import io
from pathlib import Path

import gym_v as gym

# All RLVE environments added in this PR (Batches 1, 2, 3)
BATCH_1_ENVS = [
    # Grid Puzzles (8)
    "RLVE/BinarioNoAdjacencyRequirement-v0",
    "RLVE/CampsitePuzzle-v0",
    "RLVE/HitoriPuzzle-v0",
    "RLVE/MagicSquarePuzzle-v0",
    "RLVE/SkyscraperPuzzle-v0",
    "RLVE/SkyscraperSumPuzzle-v0",
    "RLVE/Numbrix-v0",
    "RLVE/TwiddlePuzzle-v0",
    # Grid Graphs (6)
    "RLVE/GridBFS-v0",
    "RLVE/GridComponent-v0",
    "RLVE/GridLocalMinimumCounting-v0",
    "RLVE/GridParityConstruction-v0",
    "RLVE/GridTriangleCounting-v0",
    "RLVE/MaximumIndependentSetGrid-v0",
    # Geometry (7)
    "RLVE/ConvexHull-v0",
    "RLVE/LargestConvexPolygon-v0",
    "RLVE/SmallestCircle-v0",
    "RLVE/SumManhattanCurvedSurface-v0",
    "RLVE/SumTriangleArea-v0",
    "RLVE/LandformGenerationCounting-v0",
    "RLVE/MonochromeBlockCounting-v0",
    # Blocks & Spatial (6)
    "RLVE/BlockImage-v0",
    "RLVE/EightDigitPuzzle-v0",
    "RLVE/NinePuzzle-v0",
    "RLVE/KloBlocks-v0",
    "RLVE/WhackAMole-v0",
    "RLVE/CirculatingGrid-v0",
]

BATCH_2_ENVS = [
    # Graph Structures (10)
    "RLVE/GraphContainTreeCounting-v0",
    "RLVE/GraphIsomorphism-v0",
    "RLVE/HamiltonianPath-v0",
    "RLVE/HamiltonianPathExistence-v0",
    "RLVE/LongestPath-v0",
    "RLVE/MaximumAchromaticNumber-v0",
    "RLVE/MaximumClique-v0",
    "RLVE/MinimumChromaticNumber-v0",
    "RLVE/MaximumWeightMatching-v0",
    "RLVE/MixedGraphEulerianCircuit-v0",
    # Tree Structures (9)
    "RLVE/BinaryTreeLeafNumExpectation-v0",
    "RLVE/FbiBinaryTree-v0",
    "RLVE/WeightedBinarytree-v0",
    "RLVE/MaximumIndependentSetTree-v0",
    "RLVE/MinimumSpanningTreeCounting-v0",
    "RLVE/MinimumWeightedSpanningTree-v0",
    "RLVE/MinimumDirectedSpanningTree-v0",
    "RLVE/MinimumDominatingSetGrid-v0",
    "RLVE/TreeCenter-v0",
    # Tree Modifications (6)
    "RLVE/TreeAddOneEdgeDiameter-v0",
    "RLVE/TreeChangeOneEdgeDiameter-v0",
    "RLVE/TreeColoring-v0",
    "RLVE/TreeDistanceEqualTriadCounting-v0",
    "RLVE/TreeEvenPartitioning-v0",
    "RLVE/TreeTopologicalSequenceCounting-v0",
    # Card Game
    "RLVE/CardColoringCounting-v0",
]

BATCH_3_ENVS = [
    # Phase 1: Matrix & 2D Array (5)
    "RLVE/MatrixPooling-v0",
    "RLVE/AdditionTable-v0",
    "RLVE/MatrixRmqCounting-v0",
    "RLVE/MatrixPermutationMainDiagonalOne-v0",
    "RLVE/MatrixPermutationBothDiagonalOne-v0",
    # Phase 2: Spatial Layout (3)
    "RLVE/PipelineArrangement-v0",
    "RLVE/RoundtableAssignment-v0",
    "RLVE/WarehouseConstruction-v0",
    # Phase 3: Sequence & Permutation (5)
    "RLVE/TetrisAttack-v0",
    "RLVE/JugPuzzle-v0",
    "RLVE/QuantumLockPuzzle-v0",
    "RLVE/FaceRightWay-v0",
    "RLVE/PreorderTraversal-v0",
    # Phase 4: Games & Interactive (6)
    "RLVE/StoneGame-v0",
    "RLVE/StoneIntervalsGame-v0",
    "RLVE/CoinSquareGame-v0",
    "RLVE/GraMinimaGame-v0",
    "RLVE/NewNimGame-v0",
    "RLVE/MoneyChargingGame-v0",
    # Phase 5: Other Visualization (5)
    "RLVE/SkaRockGarden-v0",
    "RLVE/SpyNetwork-v0",
    "RLVE/Patrol-v0",
    "RLVE/VisibleLine-v0",
    "RLVE/MaxGridPathIntersection-v0",
]

ALL_ENVS = BATCH_1_ENVS + BATCH_2_ENVS + BATCH_3_ENVS


def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def generate_html_gallery():
    """Generate HTML gallery with all environments."""
    html_parts = [
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RLVE Environments Gallery - PR #44</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: white;
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            text-align: center;
            color: rgba(255,255,255,0.9);
            font-size: 1.2em;
            margin-bottom: 40px;
        }
        .stats {
            background: rgba(255,255,255,0.95);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        .stat-item {
            text-align: center;
            padding: 10px 20px;
        }
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            font-size: 1em;
            color: #666;
            margin-top: 5px;
        }
        .batch-section {
            background: rgba(255,255,255,0.95);
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .batch-title {
            font-size: 2em;
            color: #667eea;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        .env-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .env-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .env-name {
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }
        .env-description {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 0.95em;
            line-height: 1.6;
            white-space: pre-wrap;
            color: #555;
        }
        .images-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .image-container {
            position: relative;
            overflow: hidden;
            border-radius: 5px;
            background: #f0f0f0;
            aspect-ratio: 4/3;
        }
        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            transition: transform 0.3s;
        }
        .image-container:hover img {
            transform: scale(1.05);
        }
        .seed-label {
            position: absolute;
            top: 5px;
            right: 5px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .loading {
            text-align: center;
            padding: 40px;
            font-size: 1.2em;
            color: #666;
        }
        .error {
            background: #fee;
            border: 1px solid #fcc;
            color: #c33;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 RLVE Environments Gallery</h1>
        <div class="subtitle">PR #44: Vision-Friendly RLVE Environments</div>

        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">76</div>
                <div class="stat-label">Total Environments</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">27</div>
                <div class="stat-label">Batch 1</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">25</div>
                <div class="stat-label">Batch 2</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">24</div>
                <div class="stat-label">Batch 3</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">380</div>
                <div class="stat-label">Images (5 seeds each)</div>
            </div>
        </div>
"""
    ]

    batches = [
        ("Batch 1: Grid Puzzles, Graphs, Geometry & Blocks", BATCH_1_ENVS),
        ("Batch 2: Graph Structures, Trees & Card Games", BATCH_2_ENVS),
        ("Batch 3: Matrix, Spatial, Sequences, Games & Visualization", BATCH_3_ENVS),
    ]

    seeds = [42, 123, 456, 789, 1337]

    for batch_title, env_list in batches:
        html_parts.append(f'<div class="batch-section">')
        html_parts.append(f'<h2 class="batch-title">{batch_title}</h2>')

        for env_id in env_list:
            print(f"Processing {env_id}...")
            env_name = env_id.replace("RLVE/", "").replace("-v0", "")
            try:
                env = gym.make(env_id)

                # Get description
                description = env.unwrapped.description

                html_parts.append(f'<div class="env-card">')
                html_parts.append(f'<div class="env-name">{env_name}</div>')
                html_parts.append(
                    f'<div class="env-description">{description}</div>'
                )
                html_parts.append(f'<div class="images-grid">')

                # Generate images with different seeds
                for seed in seeds:
                    try:
                        obs, _ = env.reset(seed=seed)
                        agent_id = list(obs.keys())[0]
                        image = obs[agent_id].image

                        # Convert to base64
                        img_base64 = image_to_base64(image)

                        html_parts.append(f'<div class="image-container">')
                        html_parts.append(
                            f'<img src="data:image/png;base64,{img_base64}" alt="{env_name} seed {seed}">'
                        )
                        html_parts.append(f'<div class="seed-label">seed={seed}</div>')
                        html_parts.append(f"</div>")
                    except Exception as e:
                        print(f"  Error generating image for seed {seed}: {e}")
                        html_parts.append(
                            f'<div class="error">Error with seed {seed}: {str(e)}</div>'
                        )

                html_parts.append(f"</div>")  # images-grid
                html_parts.append(f"</div>")  # env-card

                env.close()

            except Exception as e:
                print(f"  Error processing {env_id}: {e}")
                html_parts.append(f'<div class="env-card">')
                html_parts.append(f'<div class="env-name">{env_name}</div>')
                html_parts.append(f'<div class="error">Error: {str(e)}</div>')
                html_parts.append(f"</div>")

        html_parts.append(f"</div>")  # batch-section

    html_parts.append(
        """
    </div>
</body>
</html>
"""
    )

    return "".join(html_parts)


if __name__ == "__main__":
    print("Generating RLVE environments gallery...")
    print(f"Total environments: {len(ALL_ENVS)}")
    print(f"Batch 1: {len(BATCH_1_ENVS)}")
    print(f"Batch 2: {len(BATCH_2_ENVS)}")
    print(f"Batch 3: {len(BATCH_3_ENVS)}")
    print()

    html_content = generate_html_gallery()

    output_path = Path("/Users/moonshot/Desktop/project/gym-v/rlve_gallery.html")
    output_path.write_text(html_content, encoding="utf-8")

    print(f"\n✅ Gallery generated: {output_path}")
    print(f"Opening in browser...")

    import subprocess
    import sys

    if sys.platform == "darwin":  # macOS
        subprocess.run(["open", str(output_path)])
    elif sys.platform == "win32":  # Windows
        subprocess.run(["start", str(output_path)], shell=True)
    else:  # Linux
        subprocess.run(["xdg-open", str(output_path)])

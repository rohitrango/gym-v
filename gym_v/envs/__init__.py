"""Registers the internal gym-v envs then loads the env plugins for module using the entry point."""

from gym_v.envs.registration import register

# ============================================================
# Single Turn
# ============================================================

# --- Arc (1 environment) ---

# --- Algorithmic (24 environments) ---

register(
    id="Algorithmic/AdditionTable-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.addition_table:AdditionTableEnv",
    max_episode_steps=1,
    kwargs=dict(
        min_n=3,
        max_n=10,
        num_players=1,
    ),
)

register(
    id="Arc/Arc1D-v0",
    entry_point="gym_v.envs.single_turn.arc.arc_1d:Arc1DEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=28,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Arc/ArcAgi-v0",
    entry_point="gym_v.envs.single_turn.arc.arc_agi:ArcAgiEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=16,
        padding=16,
        num_players=1,
    ),
)

register(
    id="Arc/ReArc-v0",
    entry_point="gym_v.envs.single_turn.arc.rearc:ReArcEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=16,
        padding=16,
        num_players=1,
    ),
)

register(
    id="Algorithmic/BinaryMatrix-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.binary_matrix:BinaryMatrixEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500, min_n=3, max_n=5),
        cell_px=40,
        padding=20,
        num_players=1,
    ),
)

register(
    id="Logic/CircuitLogic-v0",
    entry_point="gym_v.envs.single_turn.logic.circuit_logic:CircuitLogicEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        padding=24,
        num_players=1,
    ),
)

register(
    id="Algorithmic/CirculatingGrid-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.circulating_grid:CirculatingGridEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_r_c=5,
        cell_px=56,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Algorithmic/CoinSquareGame-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.coin_square_game:CoinSquareGameEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        weight_multiple=2,
        cell_px=70,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Geometry/ConvexHull-v0",
    entry_point="gym_v.envs.single_turn.geometry.convex_hull:ConvexHullEnv",
    max_episode_steps=1,
    kwargs=dict(
        N=10,
        num_players=1,
    ),
)

register(
    id="Algorithmic/FaceRightWay-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.face_right_way:FaceRightWayEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        cell_px=60,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Graphs/FbiBinaryTree-v0",
    entry_point="gym_v.envs.single_turn.graphs.fbi_binary_tree:FbiBinaryTreeEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=4,
        probability_same_as_before=0.7,
        base_image_width=1000,
        base_image_height=800,
        num_players=1,
    ),
)

register(
    id="Algorithmic/GameOfLife-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.game_of_life:GameOfLifeEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=32,
        padding=16,
        num_players=1,
    ),
)

register(
    id="Algorithmic/GraMinimaGame-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.gra_minima_game:GraMinimaGameEnv",
    max_episode_steps=1,
    kwargs=dict(
        n=8,
        cell_px=70,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Graphs/GraphContainTreeCounting-v0",
    entry_point="gym_v.envs.single_turn.graphs.graph_contain_tree_counting:GraphContainTreeCountingEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=8,
        edge_density=0.5,
        node_radius=20,
        image_size=800,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Graphs/GraphIsomorphism-v0",
    entry_point="gym_v.envs.single_turn.graphs.graph_isomorphism:GraphIsomorphismEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=8,
        edge_density=0.3,
        node_radius=20,
        image_size=800,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Algorithmic/GridBFS-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.grid_bfs:GridBFSEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n_m=8,
        cell_px=56,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Graphs/GridComponent-v0",
    entry_point="gym_v.envs.single_turn.graphs.grid_component:GridComponentEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n_m=8,
        cell_px=56,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Algorithmic/GridLocalMinimumCounting-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.grid_local_minimum_counting:GridLocalMinimumCountingEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n_m=4,
        cell_px=64,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Graphs/HamiltonianPath-v0",
    entry_point="gym_v.envs.single_turn.graphs.hamiltonian_path:HamiltonianPathEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        edge_density=0.5,
        node_radius=18,
        image_size=700,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Algorithmic/LandformGenerationCounting-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.landform_generation_counting:LandformGenerationCountingEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        max_mod=1000000000,
        cell_px=56,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Geometry/LargestIsland-v0",
    entry_point="gym_v.envs.single_turn.geometry.largest_island:LargestIslandEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(
            size=500,
            min_num_islands=2,    # 至少生成 2 个岛屿
            max_num_islands=5,    # 最多 5 个
            min_island_size=0,    # 岛屿至少由 3 个格子组成
            max_island_size=10,   # 最大 10 个格子
            ),
        cell_px=40,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Geometry/LargestRectangleAmongPoints-v0",
    entry_point="gym_v.envs.single_turn.geometry.largest_rectangle_among_points:LargestRectangleAmongPointsEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=15,
        rewarding_strategy="(answer/gold)^beta",
        rewarding_weight=1.0,
        rewarding_beta=5.0,
        fig_size=(8, 8),
        num_players=1,
    ),
)

register(
    id="Graphs/LongestPath-v0",
    entry_point="gym_v.envs.single_turn.graphs.longest_path:LongestPathEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=6,
        edge_density=0.3,
        node_radius=18,
        image_size=700,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Algorithmic/MatrixPermutationBothDiagonalOne-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.matrix_permutation_both_diagonal_one:MatrixPermutationBothDiagonalOneEnv",
    max_episode_steps=1,
    kwargs=dict(
        N=4,
        cell_px=64,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Algorithmic/MatrixPermutationMainDiagonalOne-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.matrix_permutation_main_diagonal_one:MatrixPermutationMainDiagonalOneEnv",
    max_episode_steps=1,
    kwargs=dict(
        N=4,
        cell_px=64,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Algorithmic/MaxGridPathIntersection-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.max_grid_path_intersection:MaxGridPathIntersectionEnv",
    max_episode_steps=1,
    kwargs=dict(
        n=5,
        cell_px=70,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Graphs/MaximumAchromaticNumber-v0",
    entry_point="gym_v.envs.single_turn.graphs.maximum_achromatic_number:MaximumAchromaticNumberEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=8,
        edge_density=0.5,
        node_radius=20,
        image_size=700,
        padding=60,
        num_players=1,
        rewarding_strategy="(answer/gold)^beta",
        rewarding_weight=1.0,
        rewarding_beta=5.0,
    ),
)

register(
    id="Graphs/MaximumClique-v0",
    entry_point="gym_v.envs.single_turn.graphs.maximum_clique:MaximumCliqueEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=12,
        edge_density=0.5,
        node_radius=18,
        image_size=700,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Graphs/MaximumIndependentSetGrid-v0",
    entry_point="gym_v.envs.single_turn.graphs.maximum_independent_set_grid:MaximumIndependentSetGridEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n_m=4,
        cell_px=64,
        padding=24,
        num_players=1,
        rewarding_strategy="(answer/gold)^beta",
        rewarding_weight=1.0,
        rewarding_beta=3.0,
    ),
)

register(
    id="Graphs/MaximumIndependentSetTree-v0",
    entry_point="gym_v.envs.single_turn.graphs.maximum_independent_set_tree:MaximumIndependentSetTreeEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        node_radius=22,
        image_size=700,
        padding=60,
        num_players=1,
        rewarding_strategy="(answer/gold)^beta",
        rewarding_weight=1.0,
        rewarding_beta=3.0,
    ),
)

register(
    id="Graphs/MaximumWeightMatching-v0",
    entry_point="gym_v.envs.single_turn.graphs.maximum_weight_matching:MaximumWeightMatchingEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        edge_density=0.5,
        node_radius=18,
        image_size=700,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Graphs/MinimumChromaticNumber-v0",
    entry_point="gym_v.envs.single_turn.graphs.minimum_chromatic_number:MinimumChromaticNumberEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        edge_density=0.5,
        node_radius=18,
        image_size=700,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Graphs/MinimumDirectedSpanningTree-v0",
    entry_point="gym_v.envs.single_turn.graphs.minimum_directed_spanning_tree:MinimumDirectedSpanningTreeEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        edge_density=0.5,
        node_radius=18,
        image_size=700,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Graphs/MinimumSpanningTreeCounting-v0",
    entry_point="gym_v.envs.single_turn.graphs.minimum_spanning_tree_counting:MinimumSpanningTreeCountingEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        edge_ratio=2.0,
        max_mod=10000,
        weight_range_divisor=10,
        node_radius=18,
        image_size=700,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Graphs/MixedGraphEulerianCircuit-v0",
    entry_point="gym_v.envs.single_turn.graphs.mixed_graph_eulerian_circuit:MixedGraphEulerianCircuitEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        image_size=800,
        padding=80,
        node_radius=20,
        num_players=1,
    ),
)

register(
    id="Algorithmic/MonochromeBlockCounting-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.monochrome_block_counting:MonochromeBlockCountingEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_a_b=10,
        num_players=1,
    ),
)



register(
    id="Graphs/Patrol-v0",
    entry_point="gym_v.envs.single_turn.graphs.patrol:PatrolEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        node_radius=22,
        image_size=800,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Cognition/RectangleCount-v0",
    entry_point="gym_v.envs.single_turn.cognition.rectangle_count:RectangleCountEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=8,
        padding=16,
        num_players=1,
    ),
)

register(
    id="Algorithmic/RottenOranges-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.rotten_oranges:RottenOrangesEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(
            size=500,
            min_n=5,
            max_n=6,
            ),
        cell_px=36,
        padding=20,
        num_players=1,
    ),
)

register(
    id="Graphs/ShortestPath-v0",
    entry_point="gym_v.envs.single_turn.graphs.shortest_path:ShortestPathEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=48,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Geometry/SkaRockGarden-v0",
    entry_point="gym_v.envs.single_turn.geometry.ska_rock_garden:SkaRockGardenEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        cell_px=60,
        padding=40,
        num_players=1,
    ),
)

register(
    id="Geometry/SmallestCircle-v0",
    entry_point="gym_v.envs.single_turn.geometry.smallest_circle:SmallestCircleEnv",
    max_episode_steps=1,
    kwargs=dict(
        n_points=10,
        rewarding_strategy="(gold/answer)^beta",
        rewarding_weight=1.0,
        rewarding_beta=10.0,
        num_players=1,
    ),
)

register(
    id="Algorithmic/SpiralMatrix-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.spiral_matrix:SpiralMatrixEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=48,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Algorithmic/StoneIntervalsGame-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.stone_intervals_game:StoneIntervalsGameEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        num_players=1,
    ),
)

register(
    id="Geometry/SumTriangleArea-v0",
    entry_point="gym_v.envs.single_turn.geometry.sum_triangle_area:SumTriangleAreaEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        num_players=1,
    ),
)

register(
    id="Graphs/TreeCenter-v0",
    entry_point="gym_v.envs.single_turn.graphs.tree_center:TreeCenterEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        node_radius=22,
        image_size=700,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Graphs/TreeChangeOneEdgeDiameter-v0",
    entry_point="gym_v.envs.single_turn.graphs.tree_change_one_edge_diameter:TreeChangeOneEdgeDiameterEnv",
    max_episode_steps=1,
    kwargs=dict(
        N=8,
        node_radius=22,
        image_size=700,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Graphs/TreeColoring-v0",
    entry_point="gym_v.envs.single_turn.graphs.tree_coloring:TreeColoringEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        node_radius=18,
        image_size=700,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Graphs/TreeDistanceEqualTriadCounting-v0",
    entry_point="gym_v.envs.single_turn.graphs.tree_distance_equal_triad_counting:TreeDistanceEqualTriadCountingEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        node_radius=22,
        image_size=700,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Graphs/TreeEvenPartitioning-v0",
    entry_point="gym_v.envs.single_turn.graphs.tree_even_partitioning:TreeEvenPartitioningEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=4,
        max_k=3,
        node_radius=20,
        image_size=800,
        padding=80,
        num_players=1,
    ),
)

register(
    id="Graphs/TreeTopologicalSequenceCounting-v0",
    entry_point="gym_v.envs.single_turn.graphs.tree_topological_sequence_counting:TreeTopologicalSequenceCountingEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        max_mod=1000000,
        node_radius=22,
        image_size=700,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Geometry/VisibleLine-v0",
    entry_point="gym_v.envs.single_turn.geometry.visible_line:VisibleLineEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=10,
        canvas_width=800,
        canvas_height=600,
        padding=60,
        num_players=1,
    ),
)

register(
    id="Graphs/WeightedBinarytree-v0",
    entry_point="gym_v.envs.single_turn.graphs.weighted_binarytree:WeightedBinarytreeEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=8,
        max_score=10,
        image_width=800,
        image_height=600,
        node_radius=25,
        num_players=1,
    ),
)

# --- Cognition (8 environments) ---

# --- Geometry (9 environments) ---

# --- Graphs (29 environments) ---

# --- Logic (20 environments) ---

# --- Puzzles (24 environments) ---

register(
    id="Logic/Binairo-v0",
    entry_point="gym_v.envs.single_turn.logic.binairo:BinairoEnv",
    max_episode_steps=1,
    kwargs=dict(
        size=6,
        num_hints=12,
        cell_px=60,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Logic/BinarioNoAdjacencyRequirement-v0",
    entry_point="gym_v.envs.single_turn.logic.binario_no_adjacency_requirement:BinarioNoAdjacencyRequirementEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n_m=4,
        sparsity=0.5,
        cell_px=56,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Logic/CampsitePuzzle-v0",
    entry_point="gym_v.envs.single_turn.logic.campsite_puzzle:CampsitePuzzleEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n_m=4,
        sparsity=0.5,
        cell_px=56,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Puzzles/ChessRanger-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.chess_ranger:ChessRangerQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        num_pieces=6,
        question_type=None,
        num_players=1,
    ),
)

register(
    id="Puzzles/EightDigitPuzzle-v0",
    entry_point="gym_v.envs.single_turn.puzzles.eight_digit_puzzle:EightDigitPuzzleEnv",
    max_episode_steps=1,
    kwargs=dict(
        n=3,
        m=3,
        steps=10,
        cell_px=72,
        padding=32,
        num_players=1,
    ),
)

register(
    id="Puzzles/Freecell-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.freecell:FreecellQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        cascade_number=None,
        num_players=1,
    ),
)

register(
    id="Logic/Futoshiki-v0",
    entry_point="gym_v.envs.single_turn.logic.futoshiki:FutoshikiEnv",
    max_episode_steps=1,
    kwargs=dict(
        size=5,
        cell_px=60,
        padding=30,
        num_players=1,
    ),
)

register(
    id="Logic/GridParityConstruction-v0",
    entry_point="gym_v.envs.single_turn.logic.grid_parity_construction:GridParityConstructionEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n_m=8,
        cell_px=56,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Cognition/Hue-QA-v0",
    entry_point="gym_v.envs.single_turn.cognition.hue:HueQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        board_size=None,
        num_lines=None,
        cell_size=60,
        num_players=1,
    ),
)

register(
    id="Puzzles/Jewel2-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.jewel2:Jewel2QAEnv",
    max_episode_steps=1,
    kwargs=dict(
        size=5,
        question_type=None,
        num_players=1,
    ),
)

register(
    id="Logic/Kakurasu-v0",
    entry_point="gym_v.envs.single_turn.logic.kakurasu:KakurasuEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=56,
        padding=40,
        num_players=1,
    ),
)

register(
    id="Puzzles/KloBlocks-v0",
    entry_point="gym_v.envs.single_turn.puzzles.klo_blocks:KloBlocksEnv",
    max_episode_steps=1,
    kwargs=dict(
        N=10,
        cell_px=60,
        padding=30,
        num_players=1,
    ),
)


register(
    id="Puzzles/KnightSwap-v0",
    entry_point="gym_v.envs.single_turn.puzzles.knight_swap:KnightSwapEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=64,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Algorithmic/LangtonAnt-QA-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.langton_ant:LangtonAntQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=None,
        cell_size=30,
        num_players=1,
    ),
)

register(
    id="Algorithmic/Lifegame-QA-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.lifegame:LifegameQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=None,
        cell_size=30,
        num_players=1,
    ),
)

register(
    id="Logic/MagicSquarePuzzle-v0",
    entry_point="gym_v.envs.single_turn.logic.magic_square_puzzle:MagicSquarePuzzleEnv",
    max_episode_steps=1,
    kwargs=dict(
        min_n=3,
        max_n=5,
        sparsity=0.5,
        cell_px=64,
        padding=32,
        num_players=1,
    ),
)

register(
    id="Puzzles/Maze-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.maze_qa:MazeQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        size="small",
        cell_size=40,
        num_players=1,
    ),
)

register(
    id="Logic/MiniSudoku-v0",
    entry_point="gym_v.envs.single_turn.logic.mini_sudoku:MiniSudokuEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=80,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Logic/NQueens-v0",
    entry_point="gym_v.envs.single_turn.logic.n_queens:NQueensEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=64,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Puzzles/NinePuzzle-v0",
    entry_point="gym_v.envs.single_turn.puzzles.nine_puzzle:NinePuzzleEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n_m=3,
        steps=5,
        cell_px=64,
        padding=32,
        num_players=1,
    ),
)

register(
    id="Logic/Numbrix-v0",
    entry_point="gym_v.envs.single_turn.logic.numbrix:NumbrixEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n_m=4,
        sparsity=0.5,
        cell_px=56,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Puzzles/Pacman-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.pacman:PacmanQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=16,
        wall_ratio=0.1,
        cell_size=25,
        num_players=1,
    ),
)

register(
    id="Puzzles/PyramidChess-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.pyramidchess:PyramidChessQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        plot_level="Easy",
        question_type=None,
        num_players=1,
    ),
)

register(
    id="Logic/Renzoku-v0",
    entry_point="gym_v.envs.single_turn.logic.renzoku:RenzokuEnv",
    max_episode_steps=1,
    kwargs=dict(
        size=9,
        cell_px=50,
        padding=30,
        num_players=1,
    ),
)

register(
    id="Puzzles/RhythmGame-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.rhythm_game:RhythmGameQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=None,
        difficulty=None,
        cell_size=40,
        num_players=1,
    ),
)

register(
    id="Algorithmic/RotateMatrix-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.rotate_matrix:RotateMatrixEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=48,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Logic/SkyscraperPuzzle-v0",
    entry_point="gym_v.envs.single_turn.logic.skyscraper_puzzle:SkyscraperPuzzleEnv",
    max_episode_steps=1,
    kwargs=dict(
        n=3,
        cell_px=52,
        padding=28,
        num_players=1,
    ),
)

register(
    id="Puzzles/Snake-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.snake:SnakeQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        width=10,
        height=10,
        initial_snake_length=(10, 20),
        cell_size=40,
        num_players=1,
    ),
)

register(
    id="Puzzles/SpaceInvaders-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.space_invaders:SpaceInvadersQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        enemy_rows=4,
        enemy_cols=6,
        enemy_area_rows=8,
        cell_width=50,
        cell_height=40,
        num_players=1,
    ),
)

register(
    id="Puzzles/SpiderSolitaire-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.spider_solitaire:SpiderSolitaireQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        num_waste=10,
        circular=False,
        open=False,
        question_type=None,
        num_players=1,
    ),
)

register(
    id="Logic/StarBattle-QA-v0",
    entry_point="gym_v.envs.single_turn.logic.star_battle:StarBattleQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=6,
        stars_per_region=1,
        cell_size=50,
        num_players=1,
    ),
)

register(
    id="Logic/Survo-v0",
    entry_point="gym_v.envs.single_turn.logic.survo:SurvoEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=64,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Geometry/Tangram-QA-v0",
    entry_point="gym_v.envs.single_turn.geometry.tangram:TangramQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=None,
        num_seeds=None,
        num_pieces_to_remove=None,
        num_players=1,
    ),
)

register(
    id="Logic/Tents-QA-v0",
    entry_point="gym_v.envs.single_turn.logic.tents:TentsQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=None,
        num_trees=None,
        cell_size=50,
        num_players=1,
    ),
)

register(
    id="Puzzles/Tetris-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.tetris:TetrisQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        rows=12,
        cols=8,
        cell_size=30,
        num_players=1,
    ),
)

register(
    id="Puzzles/TetrisAttack-v0",
    entry_point="gym_v.envs.single_turn.puzzles.tetris_attack:TetrisAttackEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n=8,
        cell_px=60,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Logic/Thermometers-v0",
    entry_point="gym_v.envs.single_turn.logic.thermometers:ThermometersEnv",
    max_episode_steps=1,
    kwargs=dict(
        size=5,
        num_hints=0,
        cell_px=60,
        padding=50,
        num_players=1,
    ),
)

register(
    id="Puzzles/TicTacToe-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.tictactoe:TicTacToeQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        num_players=1,
    ),
)

register(
    id="Puzzles/TowerOfHanoi-v0",
    entry_point="gym_v.envs.single_turn.puzzles.tower_of_hanoi:TowerOfHanoiEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500, min_disks=3, max_disks=4, min_pegs=3, max_pegs=4),
        peg_width=150,
        peg_height=250,
        padding=40,
        num_players=1,
    ),
)

register(
    id="Puzzles/Tsumego-v0",
    entry_point="gym_v.envs.single_turn.puzzles.tsumego:TsumegoEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=36,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Algorithmic/TuringMachine2d-QA-v0",
    entry_point="gym_v.envs.single_turn.algorithmic.turing_machine_2d:TuringMachineQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=None,
        num_states=2,
        num_symbols=2,
        max_steps=8,
        cell_size=50,
        num_players=1,
    ),
)

register(
    id="Puzzles/TwiddlePuzzle-v0",
    entry_point="gym_v.envs.single_turn.puzzles.twiddle_puzzle:TwiddlePuzzleEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n_m=4,
        steps=3,
        cell_px=56,
        padding=24,
        num_players=1,
    ),
)

register(
    id="Puzzles/UltraTicTacToe-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.ultra_tictactoe:UltraTicTacToeQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        plot_level="Easy",
        question_type=None,
        num_players=1,
    ),
)

register(
    id="Puzzles/WordSearch-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.word_search:WordSearchQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=None,
        cell_size=50,
        num_players=1,
    ),
)

register(
    id="Puzzles/Zuma-QA-v0",
    entry_point="gym_v.envs.single_turn.puzzles.zuma:ZumaQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        curve_type=None,
        num_balls=None,
        ball_radius=0.3,
        num_players=1,
    ),
)

# --- Cognition (moved from Perception) ---

register(
    id="Cognition/FlowNetwork-v0",
    entry_point="gym_v.envs.single_turn.cognition.flow_network:FlowNetworkEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=(640, 480),
        min_nodes=5,
        max_nodes=8,
        num_players=1,
    ),
)

register(
    id="Cognition/Maze3D-QA-v0",
    entry_point="gym_v.envs.single_turn.cognition.maze_3d:MazeQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=(8, 8, 7),
        num_players=1,
    ),
)

register(
    id="Cognition/OddOneOutPoly-v0",
    entry_point="gym_v.envs.single_turn.cognition.odd_one_out:OddOneOutPolyEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=200,
        num_points=8,
        line_width=3,
        grid_divisions=8,
        option_size=200,
        padding=15,
    ),
)


register(
    id="Cognition/RubiksCube-QA-v0",
    entry_point="gym_v.envs.single_turn.cognition.rubiks_cube:RubiksCubeQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        num_moves=None,
        num_players=1,
    ),
)

register(
    id="Cognition/SequenceCompletionPoly-v0",
    entry_point="gym_v.envs.single_turn.cognition.sequence_completion:SequenceCompletionPolyEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=200,
        num_points=6,
        line_width=3,
        grid_divisions=8,
        sequence_length=4,
        option_size=150,
        padding=10,
    ),
)

register(
    id="Cognition/SymmetryFillPoly-v0",
    entry_point="gym_v.envs.single_turn.cognition.symmetry_fill:SymmetryFillPolyEnv",
    max_episode_steps=1,
    kwargs=dict(
        cell_size=200,
        line_width=4,
        option_size=200,
        padding=15,
    ),
)

register(
    id="Cognition/TransformResultPoly-v0",
    entry_point="gym_v.envs.single_turn.cognition.transform_result:TransformResultPolyEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=300,
        num_points=8,
        line_width=3,
        grid_divisions=8,
        option_size=280,
        padding=20,
    ),
)

register(
    id="Cognition/TreeToTraversal-v0",
    entry_point="gym_v.envs.single_turn.cognition.tree_to_traversal:TreeToTraversalEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=(640, 480),
        min_nodes=5,
        max_nodes=12,
        num_players=1,
    ),
)

# ============================================================
# Multi Turn
# ============================================================

# --- Games (31 environments) ---

register(
    id="Games/Alquerque-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.alquerque:AlquerqueEnv",
    max_episode_steps=200,
    kwargs=dict(
        tile_size=80,
        num_players=2,
    ),
)

register(
    id="Games/Breakthrough-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.breakthrough:BreakthroughEnv",
    max_episode_steps=200,
    kwargs=dict(
        board_size=8,
        tile_size=60,
        num_players=2,
    ),
)

register(
    id="Games/Chess-v0",
    entry_point="gym_v.envs.multi_turn.games.multi_player.chess:ChessEnv",
    max_episode_steps=500,
    kwargs=dict(
        num_players=2,
    ),
)

register(
    id="Games/ConnectFour-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.connectfour:ConnectFourEnv",
    max_episode_steps=100,
    kwargs=dict(
        num_rows=6,
        num_cols=7,
        tile_size=80,
        num_players=2,
    ),
)

register(
    id="Games/ConnectFourMultiAgent-v0",
    entry_point="gym_v.envs.multi_turn.games.multi_player.connectfour:ConnectFourMultiAgentEnv",
    max_episode_steps=100,
    kwargs=dict(
        num_players=2,
    ),
)

register(
    id="Games/Crosswords-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.crosswords:CrosswordsEnv",
    max_episode_steps=100,
    kwargs=dict(
        hardcore=False,
        num_words=5,
        cell_size=48,
        num_players=1,
    ),
)

register(
    id="Games/Crusade-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.crusade:CrusadeEnv",
    max_episode_steps=100,
    kwargs=dict(
        tile_size=60,
        num_players=2,
    ),
)

register(
    id="Games/FifteenPuzzle-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.fifteenpuzzle:FifteenPuzzleEnv",
    max_episode_steps=100,
    kwargs=dict(
        tile_size=80,
        num_players=1,
    ),
)

register(
    id="Games/FrozenLake-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.frozenlake:FrozenLakeEnv",
    max_episode_steps=100,
    kwargs=dict(
        size=4,
        num_holes=3,
        randomize_start_goal=False,
        tile_size=64,
        num_players=1,
    ),
)

register(
    id="Games/Game2048-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.game2048:Game2048Env",
    max_episode_steps=100,
    kwargs=dict(
        target_tile=2048,
        tile_size=100,
        num_players=1,
    ),
)

register(
    id="Games/GinRummy-v0",
    entry_point="gym_v.envs.multi_turn.games.multi_player.gin_rummy:GinRummyEnv",
    max_episode_steps=200,
    kwargs=dict(
        num_players=2,
        knock_reward=0.5,
        gin_reward=1.0,
        opponents_hand_visible=False,
    ),
)

register(
    id="Games/Go-v0",
    entry_point="gym_v.envs.multi_turn.games.multi_player.go:GoEnv",
    max_episode_steps=1000,
    kwargs=dict(
        num_players=2,
        board_size=19,
        komi=7.5,
    ),
)

register(
    id="Games/LeducHoldem-v0",
    entry_point="gym_v.envs.multi_turn.games.multi_player.leduc_holdem:LeducHoldemEnv",
    max_episode_steps=100,
    kwargs=dict(
        num_players=2,
    ),
)

register(
    id="Games/LightsOut-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.lightsout:LightsOutEnv",
    max_episode_steps=100,
    kwargs=dict(
        size=5,
        cell_size=80,
        num_players=1,
    ),
)

register(
    id="Games/LinesOfAction-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.linesofaction:LinesOfActionEnv",
    max_episode_steps=200,
    kwargs=dict(
        tile_size=60,
        num_players=2,
    ),
)

register(
    id="Games/Minesweeper-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.minesweeper:MinesweeperEnv",
    max_episode_steps=100,
    kwargs=dict(
        rows=8,
        cols=8,
        num_mines=10,
        cell_size=64,
        num_players=1,
    ),
)

register(
    id="Games/Nim-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.nim:NimEnv",
    max_episode_steps=50,
    kwargs=dict(
        piles=[3, 4, 5],
        pile_width=100,
        num_players=2,
    ),
)

register(
    id="Games/Othello-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.othello:OthelloEnv",
    max_episode_steps=100,
    kwargs=dict(
        board_size=8,
        show_valid=True,
        tile_size=80,
        num_players=2,
    ),
)

register(
    id="Games/PegJump-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.pegjump:PegJumpEnv",
    max_episode_steps=100,
    kwargs=dict(
        initial_empty=1,
        peg_size=80,
        num_players=1,
    ),
)

register(
    id="Games/RushHour-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.rushhour:RushHourEnv",
    max_episode_steps=100,
    kwargs=dict(
        difficulty="easy",
        cell_size=80,
        num_players=1,
    ),
)

register(
    id="Games/SimpleTak-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.simpletak:SimpleTakEnv",
    max_episode_steps=50,
    kwargs=dict(
        board_size=5,
        cell_size=80,
        num_players=2,
    ),
)

register(
    id="Games/Sokoban-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.sokoban:SokobanEnv",
    max_episode_steps=100,
    kwargs=dict(
        dim_room=(6, 6),
        num_boxes=3,
        tile_size=48,
        num_players=1,
    ),
)

register(
    id="Games/Sudoku-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.sudoku:SudokuEnv",
    max_episode_steps=100,
    kwargs=dict(
        clues=30,
        cell_size=50,
        num_players=1,
    ),
)

register(
    id="Games/TexasHoldem-v0",
    entry_point="gym_v.envs.multi_turn.games.multi_player.texas_holdem:TexasHoldemEnv",
    max_episode_steps=200,
    kwargs=dict(
        num_players=2,
    ),
)

register(
    id="Games/TexasHoldemNoLimit-v0",
    entry_point="gym_v.envs.multi_turn.games.multi_player.texas_holdem_no_limit:TexasHoldemNoLimitEnv",
    max_episode_steps=200,
    kwargs=dict(
        num_players=2,
    ),
)

register(
    id="Games/TicTacToe-v0",
    entry_point="gym_v.envs.multi_turn.games.multi_player.tictactoe:TicTacToeEnv",
    max_episode_steps=20,
    kwargs=dict(
        num_players=2,
    ),
)

register(
    id="Games/TowerOfHanoiMultiTurn-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.towerofhanoi:TowerOfHanoiEnv",
    max_episode_steps=100,
    kwargs=dict(
        num_disks=3,
    ),
)

register(
    id="Games/UltimateTicTacToe-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.ultimatetictactoe:UltimateTicTacToeEnv",
    max_episode_steps=100,
    kwargs=dict(
        mini_board_size=200,
        num_players=2,
    ),
)

register(
    id="Games/WildTicTacToe-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.wildtictactoe:WildTicTacToeEnv",
    max_episode_steps=20,
    kwargs=dict(
        tile_size=120,
        num_players=2,
    ),
)

register(
    id="Games/WordSearch-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.wordsearch:WordSearchEnv",
    max_episode_steps=100,
    kwargs=dict(
        hardcore=False,
        cell_size=60,
        num_players=1,
    ),
)

register(
    id="Games/Wordle-v0",
    entry_point="gym_v.envs.multi_turn.games.single_player.wordle:WordleEnv",
    max_episode_steps=100,
    kwargs=dict(
        word_length=5,
        num_guesses=6,
        hardcore=False,
        cell_size=60,
        num_players=1,
    ),
)

# --- Spatial (30 environments) ---

register(
    id="Spatial/CollectHealth-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.collecthealth:CollectHealthEnv",
    max_episode_steps=1000,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/DoorKey-v0",
    entry_point="gym_v.envs.multi_turn.spatial.2d.doorkey:DoorKeyEnv",
    max_episode_steps=640,
    kwargs=dict(
        size=8,
        tile_size=32,
        num_players=1,
    ),
)

register(
    id="Spatial/DynamicObstacles-v0",
    entry_point="gym_v.envs.multi_turn.spatial.2d.dynamicobstacles:DynamicObstaclesEnv",
    max_episode_steps=256,
    kwargs=dict(
        size=8,
        n_obstacles=4,
        tile_size=32,
        num_players=1,
    ),
)

register(
    id="Spatial/Empty-v0",
    entry_point="gym_v.envs.multi_turn.spatial.2d.empty:EmptyEnv",
    max_episode_steps=256,
    kwargs=dict(
        size=8,
        agent_start_pos=(1, 1),
        tile_size=32,
        num_players=1,
    ),
)

register(
    id="Spatial/FourRooms2D-v0",
    entry_point="gym_v.envs.multi_turn.spatial.2d.fourrooms:FourRoomsEnv",
    max_episode_steps=100,
    kwargs=dict(
        tile_size=32,
        num_players=1,
    ),
)

register(
    id="Spatial/FourRooms3D-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.rooms:FourRoomsEnv",
    max_episode_steps=1000,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/Hallway-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.hallway:HallwayEnv",
    max_episode_steps=400,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/LavaGap-v0",
    entry_point="gym_v.envs.multi_turn.spatial.2d.lavagap:LavaGapEnv",
    max_episode_steps=100,
    kwargs=dict(
        size=7,
        tile_size=32,
        num_players=1,
    ),
)

register(
    id="Spatial/Maze-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.maze:MazeEnv",
    max_episode_steps=1000,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/MazeS2-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.maze:MazeS2Env",
    max_episode_steps=500,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/MazeS3-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.maze:MazeS3Env",
    max_episode_steps=800,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/MazeS3Fast-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.maze:MazeS3FastEnv",
    max_episode_steps=600,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/MultiRoom-v0",
    entry_point="gym_v.envs.multi_turn.spatial.2d.multiroom:MultiRoomEnv",
    max_episode_steps=100,
    kwargs=dict(
        min_num_rooms=6,
        max_num_rooms=6,
        max_room_size=10,
        tile_size=32,
        num_players=1,
    ),
)

register(
    id="Spatial/OneRoom-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.oneroom:OneRoomEnv",
    max_episode_steps=400,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/OneRoomS6-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.oneroom:OneRoomS6Env",
    max_episode_steps=600,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/OneRoomS6Fast-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.oneroom:OneRoomS6FastEnv",
    max_episode_steps=400,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/PickupObjects-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.pickup:PickupObjectsEnv",
    max_episode_steps=1000,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/PutNext-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.putnext:PutNextEnv",
    max_episode_steps=1000,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/RoomObjects-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.rooms:RoomObjectsEnv",
    max_episode_steps=800,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/Sidewalk-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.sidewalk:SidewalkEnv",
    max_episode_steps=800,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/Sign-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.sign:SignEnv",
    max_episode_steps=600,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/TMaze-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.tmaze:TMazeEnv",
    max_episode_steps=500,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/TMazeLeft-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.tmaze:TMazeLeftEnv",
    max_episode_steps=500,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/TMazeRight-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.tmaze:TMazeRightEnv",
    max_episode_steps=500,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/ThreeRooms-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.rooms:ThreeRoomsEnv",
    max_episode_steps=800,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/Unlock-v0",
    entry_point="gym_v.envs.multi_turn.spatial.2d.unlock:UnlockEnv",
    max_episode_steps=100,
    kwargs=dict(
        tile_size=32,
        num_players=1,
    ),
)

register(
    id="Spatial/WallGap-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.wallgap:WallGapEnv",
    max_episode_steps=400,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/YMaze-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.ymaze:YMazeEnv",
    max_episode_steps=500,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/YMazeLeft-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.ymaze:YMazeLeftEnv",
    max_episode_steps=500,
    kwargs=dict(num_players=1),
)

register(
    id="Spatial/YMazeRight-v0",
    entry_point="gym_v.envs.multi_turn.spatial.3d.ymaze:YMazeRightEnv",
    max_episode_steps=500,
    kwargs=dict(num_players=1),
)

# --- Temporal (13 environments) ---

register(
    id="Temporal/Airstriker-v0",
    entry_point="gym_v.envs.multi_turn.temporal.retro_env:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="Airstriker-Genesis",
        num_players=1,
    ),
)

register(
    id="Temporal/AlteredBeast-v0",
    entry_point="gym_v.envs.multi_turn.temporal.retro_env:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="AlteredBeast-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Temporal/CastleOfIllusion-v0",
    entry_point="gym_v.envs.multi_turn.temporal.retro_env:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="CastleOfIllusion-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Temporal/CastlevaniaBloodlines-v0",
    entry_point="gym_v.envs.multi_turn.temporal.retro_env:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="CastlevaniaBloodlines-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Temporal/Columns-v0",
    entry_point="gym_v.envs.multi_turn.temporal.retro_env:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="Columns-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Temporal/DynamiteHeaddy-v0",
    entry_point="gym_v.envs.multi_turn.temporal.retro_env:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="DynamiteHeaddy-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Temporal/GoldenAxe-v0",
    entry_point="gym_v.envs.multi_turn.temporal.retro_env:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="GoldenAxe-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Temporal/KidChameleon-v0",
    entry_point="gym_v.envs.multi_turn.temporal.retro_env:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="KidChameleon-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Temporal/MortalKombatII-v0",
    entry_point="gym_v.envs.multi_turn.temporal.retro_env:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="MortalKombatII-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Temporal/SpaceHarrierII-v0",
    entry_point="gym_v.envs.multi_turn.temporal.retro_env:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="SpaceHarrierII-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Temporal/StreetsOfRage2-v0",
    entry_point="gym_v.envs.multi_turn.temporal.retro_env:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="StreetsOfRage2-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Temporal/Strider-v0",
    entry_point="gym_v.envs.multi_turn.temporal.retro_env:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="Strider-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Temporal/ThunderForceIII-v0",
    entry_point="gym_v.envs.multi_turn.temporal.retro_env:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="ThunderForceIII-Genesis-v0",
        num_players=1,
    ),
)

# ============================================================
# Uncategorized
# ============================================================

register(
    id="Offline/SingleTurn-v0",
    entry_point="gym_v.envs.offline.single_turn:OfflineSingleTurnEnv",
    max_episode_steps=1,
    kwargs=dict(
        grader="exact_match",
        description=None,
        shuffle=True,
    ),
)

register(
    id="VLMEval-Base-v0",
    entry_point="gym_v.envs.eval.vlmeval:VLMEvalEnv",
    max_episode_steps=1,
    kwargs={},
)

register(
    id="GenEval-v0",
    entry_point="gym_v.envs.eval.t2ieval:GenevalEnv",
    max_episode_steps=1,
    kwargs=dict(),
)

# GenEval2 T2I environment
register(
    id="GenEval2-v0",
    entry_point="gym_v.envs.eval.t2ieval:Geneval2Env",
    max_episode_steps=1,
    kwargs=dict(),
)

# GenExam T2I environment
register(
    id="GenExam-v0",
    entry_point="gym_v.envs.eval.t2ieval:GenExamEnv",
    max_episode_steps=1,
    kwargs=dict(),
)

# WISE T2I environment
register(
    id="WISE-v0",
    entry_point="gym_v.envs.eval.t2ieval:WiseEnv",
    max_episode_steps=1,
    kwargs=dict(),
)

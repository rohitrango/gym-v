# gym-v

## 开发前可以先安装一下pre-commit (uv run pre-commit install)

## 已接入环境列表

**总计: 202 个环境**

- Single Turn (单轮): 125 个
- Multi Turn (多轮): 74 个
- 其他: 3 个

---

## Single Turn 环境 (125 个)

单轮环境仅需一步交互，智能体观察图像后给出答案，环境返回奖励。

### Arc (3 个)

ARC (Abstraction and Reasoning Corpus) 抽象推理任务。

| 环境 ID | 简介 |
|---------|------|
| `Arc/Arc1D-v0` | 一维 ARC 推理任务，根据输入输出示例推断变换规则 |
| `Arc/ArcAgi-v0` | ARC-AGI 基准推理任务，二维网格颜色变换推理 |
| `Arc/ReArc-v0` | ReArc 程序生成的 ARC 任务，支持无限量题目 |

### Algorithmic (24 个)

算法与模拟类任务，涵盖矩阵运算、博弈论、搜索、计数等。

| 环境 ID | 简介 |
|---------|------|
| `Algorithmic/AdditionTable-v0` | 加法表填空，根据部分信息推断缺失值 |
| `Algorithmic/BinaryMatrix-v0` | 二进制矩阵变换，预测下一步状态 |
| `Algorithmic/BinaryTreeLeafNumExpectation-v0` | 随机二叉树的叶子节点数量期望计算 |
| `Algorithmic/CirculatingGrid-v0` | 循环网格优化，最少修改使行列满足约束 |
| `Algorithmic/CoinSquareGame-v0` | 硬币方格博弈，判断先后手胜负 |
| `Algorithmic/FaceRightWay-v0` | 朝向调整，计算最少翻转次数使所有朝同一方向 |
| `Algorithmic/GameOfLife-v0` | 康威生命游戏，预测若干步后的网格状态 |
| `Algorithmic/GraMinimaGame-v0` | 图博弈，在网格上的极小值策略分析 |
| `Algorithmic/GridBFS-v0` | 网格 BFS 最短路径搜索 |
| `Algorithmic/GridLocalMinimumCounting-v0` | 网格局部最小值计数 |
| `Algorithmic/LandformGenerationCounting-v0` | 地形生成计数，计算满足高度约束的方案数 |
| `Algorithmic/LangtonAnt-QA-v0` | 兰顿蚂蚁模拟问答，预测蚂蚁行为 |
| `Algorithmic/Lifegame-QA-v0` | 生命游戏模拟问答，分析网格演化 |
| `Algorithmic/MatrixPermutationBothDiagonalOne-v0` | 矩阵排列使两条对角线均为 1 的方案计数 |
| `Algorithmic/MatrixPermutationMainDiagonalOne-v0` | 矩阵排列使主对角线均为 1 的方案计数 |
| `Algorithmic/MaxGridPathIntersection-v0` | 网格路径最大交叉点数 |
| `Algorithmic/MonochromeBlockCounting-v0` | 单色块计数 |
| `Algorithmic/RotateMatrix-v0` | 矩阵旋转，输出旋转后的矩阵 |
| `Algorithmic/RottenOranges-v0` | 腐烂橘子扩散模拟，计算全部腐烂所需时间 |
| `Algorithmic/SpiralMatrix-v0` | 螺旋矩阵遍历，按螺旋顺序输出元素 |
| `Algorithmic/StoneGame-v0` | 取石子博弈，判断先后手胜负 |
| `Algorithmic/StoneIntervalsGame-v0` | 石子区间博弈策略分析 |
| `Algorithmic/TuringMachine2d-QA-v0` | 2D 图灵机模拟问答，预测执行结果 |

### Cognition (8 个)

认知与空间推理任务，包含模式识别、3D 理解、视觉推理。

| 环境 ID | 简介 |
|---------|------|
| `Cognition/Hue-QA-v0` | 颜色匹配游戏问答 |
| `Cognition/Maze3D-QA-v0` | 3D 迷宫空间推理问答 |
| `Cognition/OddOneOutPoly-v0` | 多边形找不同，识别异类图形 |
| `Cognition/RectangleCount-v0` | 矩形计数，统计图中矩形数量 |
| `Cognition/RubiksCube-QA-v0` | 魔方状态问答，分析旋转后的状态 |
| `Cognition/SequenceCompletionPoly-v0` | 多边形序列补全，推断下一个图形 |
| `Cognition/SymmetryFillPoly-v0` | 对称填充，补全对称图形 |
| `Cognition/TransformResultPoly-v0` | 变换结果预测，推断几何变换后的图形 |

### Geometry (8 个)

计算几何任务，涉及凸包、面积、最小圆等经典问题。

| 环境 ID | 简介 |
|---------|------|
| `Geometry/ConvexHull-v0` | 凸包计算，找出点集的凸包顶点 |
| `Geometry/LargestRectangleAmongPoints-v0` | 点集中最大矩形面积 |
| `Geometry/PipelineArrangement-v0` | 管道排列优化 |
| `Geometry/SkaRockGarden-v0` | 枯山水庭院布局问题 |
| `Geometry/SmallestCircle-v0` | 最小覆盖圆，求包含所有点的最小圆 |
| `Geometry/SumTriangleArea-v0` | 三角形面积之和计算 |
| `Geometry/Tangram-QA-v0` | 七巧板拼图问答 |
| `Geometry/VisibleLine-v0` | 可见线段计算，判断从上方可见的线段 |

### Graphs (27 个)

图论任务，涵盖最短路、生成树、着色、匹配、同构等经典问题。

| 环境 ID | 简介 |
|---------|------|
| `Graphs/FbiBinaryTree-v0` | FBI 二叉树问题，分析树结构 |
| `Graphs/GraphContainTreeCounting-v0` | 图中包含的生成树计数 |
| `Graphs/GraphIsomorphism-v0` | 图同构判定 |
| `Graphs/GridComponent-v0` | 网格连通分量计数 |
| `Graphs/HamiltonianPath-v0` | 哈密顿路径求解 |
| `Graphs/HamiltonianPathExistence-v0` | 哈密顿路径存在性判定 |
| `Graphs/LargestIsland-v0` | 最大岛屿面积搜索 |
| `Graphs/LongestPath-v0` | 图中最长路径求解 |
| `Graphs/MaximumAchromaticNumber-v0` | 最大无色数计算 |
| `Graphs/MaximumClique-v0` | 最大团求解 |
| `Graphs/MaximumIndependentSetGrid-v0` | 网格图最大独立集 |
| `Graphs/MaximumIndependentSetTree-v0` | 树上最大权独立集 |
| `Graphs/MaximumWeightMatching-v0` | 最大权匹配 |
| `Graphs/MinimumChromaticNumber-v0` | 最小色数（图着色） |
| `Graphs/MinimumDirectedSpanningTree-v0` | 最小有向生成树 |
| `Graphs/MinimumSpanningTreeCounting-v0` | 最小生成树计数 |
| `Graphs/MixedGraphEulerianCircuit-v0` | 混合图欧拉回路判定 |
| `Graphs/Patrol-v0` | 巡逻路径规划 |
| `Graphs/ShortestPath-v0` | 加权图最短路径 |
| `Graphs/SpyNetwork-v0` | 间谍网络最小顶点覆盖 |
| `Graphs/TreeCenter-v0` | 树的中心节点求解 |
| `Graphs/TreeChangeOneEdgeDiameter-v0` | 改变一条边最小化树直径 |
| `Graphs/TreeColoring-v0` | 树着色方案数 |
| `Graphs/TreeDistanceEqualTriadCounting-v0` | 树上等距三元组计数 |
| `Graphs/TreeEvenPartitioning-v0` | 树的均匀划分 |
| `Graphs/TreeTopologicalSequenceCounting-v0` | 树拓扑序列计数 |
| `Graphs/WeightedBinarytree-v0` | 加权二叉树最大分值 |

### Logic (19 个)

逻辑推理与约束满足类谜题。

| 环境 ID | 简介 |
|---------|------|
| `Logic/Binairo-v0` | 二进制谜题 (Binairo)，填入 0/1 满足行列约束 |
| `Logic/BinarioNoAdjacencyRequirement-v0` | 无相邻要求的二进制填充谜题 |
| `Logic/CampsitePuzzle-v0` | 露营地谜题，在树旁放置帐篷 |
| `Logic/CircuitLogic-v0` | 电路逻辑门求解 |
| `Logic/Futoshiki-v0` | 不等号数独 |
| `Logic/GridParityConstruction-v0` | 网格奇偶性构造 |
| `Logic/HitoriPuzzle-v0` | Hitori 谜题变体 |
| `Logic/Kakurasu-v0` | Kakurasu 加法逻辑谜题 |
| `Logic/MagicSquarePuzzle-v0` | 幻方填数 |
| `Logic/MiniSudoku-v0` | 迷你数独 (4x4) |
| `Logic/NQueens-v0` | N 皇后问题 |
| `Logic/Numbrix-v0` | 数字路径连接谜题 |
| `Logic/Renzoku-v0` | 连续数字谜题 |
| `Logic/SkyscraperPuzzle-v0` | 摩天楼谜题 |
| `Logic/SkyscraperSumPuzzle-v0` | 摩天楼求和谜题 |
| `Logic/StarBattle-QA-v0` | 星星大战谜题问答 |
| `Logic/Survo-v0` | Survo 逻辑谜题 |
| `Logic/Tents-QA-v0` | 帐篷谜题问答 |
| `Logic/Thermometers-v0` | 温度计谜题 |

### Perception (12 个)

图表与数学图形的视觉理解任务。

| 环境 ID | 简介 |
|---------|------|
| `Perception/3DReconstruction-QA-v0` | 3D 重建视觉理解问答 |
| `Perception/ChartToTable-v0` | 图表转表格数据 |
| `Perception/ContourPlot-v0` | 等高线图函数理解 |
| `Perception/DAGToTopoOrder-v0` | 有向无环图拓扑排序 |
| `Perception/FlowNetwork-v0` | 流网络最大流分析 |
| `Perception/FunctionGraph-v0` | 函数图像识别与理解 |
| `Perception/GraphToAdjacency-v0` | 图转邻接矩阵 |
| `Perception/GraphToMST-v0` | 从图中求最小生成树 |
| `Perception/ParametricCurve-v0` | 参数曲线方程识别 |
| `Perception/PolarPlot-v0` | 极坐标图方程识别 |
| `Perception/TreeToTraversal-v0` | 树结构遍历序列 |
| `Perception/VectorField-v0` | 向量场函数识别 |

### Puzzles (24 个)

经典游戏与益智谜题的视觉问答。

| 环境 ID | 简介 |
|---------|------|
| `Puzzles/ChessRanger-QA-v0` | 象棋攻击范围问答 |
| `Puzzles/EightDigitPuzzle-v0` | 八数码推盘，求解复原步骤 |
| `Puzzles/Freecell-QA-v0` | 空当接龙问答 |
| `Puzzles/Jewel2-QA-v0` | 宝石消除问答 |
| `Puzzles/KloBlocks-v0` | 方块消除谜题 |
| `Puzzles/Klondike-QA-v0` | 克朗代克纸牌问答 |
| `Puzzles/KnightSwap-v0` | 骑士交换位置求解 |
| `Puzzles/Maze-QA-v0` | 迷宫问答 |
| `Puzzles/NinePuzzle-v0` | 九宫格推盘谜题 |
| `Puzzles/Pacman-QA-v0` | 吃豆人问答 |
| `Puzzles/PyramidChess-QA-v0` | 金字塔象棋问答 |
| `Puzzles/RhythmGame-QA-v0` | 节奏游戏问答 |
| `Puzzles/Snake-QA-v0` | 贪吃蛇问答 |
| `Puzzles/SpaceInvaders-QA-v0` | 太空侵略者问答 |
| `Puzzles/SpiderSolitaire-QA-v0` | 蜘蛛纸牌问答 |
| `Puzzles/Tetris-QA-v0` | 俄罗斯方块问答 |
| `Puzzles/TetrisAttack-v0` | 方块攻击，最少交换消除 |
| `Puzzles/TicTacToe-QA-v0` | 井字棋问答 |
| `Puzzles/TowerOfHanoi-v0` | 汉诺塔，求最优移动序列 |
| `Puzzles/Tsumego-v0` | 围棋死活题 (诘棋) |
| `Puzzles/TwiddlePuzzle-v0` | 旋转拼图，旋转子矩阵复原 |
| `Puzzles/UltraTicTacToe-QA-v0` | 终极井字棋问答 |
| `Puzzles/WordSearch-QA-v0` | 单词搜索问答 |
| `Puzzles/Zuma-QA-v0` | 祖玛问答 |

---

## Multi Turn 环境 (74 个)

多轮交互环境，智能体在多个步骤中与环境持续交互。

### Games (31 个)

经典棋牌与益智游戏的多轮交互版本。

| 环境 ID | 简介 |
|---------|------|
| `Games/Alquerque-v0` | 西班牙跳棋 |
| `Games/Breakthrough-v0` | 突围棋 |
| `Games/Chess-v0` | 国际象棋 |
| `Games/ConnectFour-v0` | 四子棋 (单智能体 vs 内置对手) |
| `Games/ConnectFourMultiAgent-v0` | 四子棋 (多智能体对战) |
| `Games/Crosswords-v0` | 英文填字游戏 |
| `Games/Crusade-v0` | 十字军棋 |
| `Games/FifteenPuzzle-v0` | 15 数字推盘 |
| `Games/FrozenLake-v0` | 冰湖导航 |
| `Games/Game2048-v0` | 2048 数字合并 |
| `Games/GinRummy-v0` | 金罗美扑克 |
| `Games/Go-v0` | 围棋 |
| `Games/LeducHoldem-v0` | Leduc 扑克 |
| `Games/LightsOut-v0` | 点灯谜题 |
| `Games/LinesOfAction-v0` | 行动路线棋 |
| `Games/Minesweeper-v0` | 扫雷 |
| `Games/Nim-v0` | 尼姆取子游戏 |
| `Games/Othello-v0` | 黑白棋 |
| `Games/PegJump-v0` | 孔明棋 |
| `Games/RushHour-v0` | 华容道 / 塞车时间 |
| `Games/SimpleTak-v0` | 简化 Tak 棋 |
| `Games/Sokoban-v0` | 推箱子 |
| `Games/Sudoku-v0` | 9x9 数独 |
| `Games/TexasHoldem-v0` | 德州扑克 |
| `Games/TexasHoldemNoLimit-v0` | 无限注德州扑克 |
| `Games/TicTacToe-v0` | 井字棋 |
| `Games/TowerOfHanoiMultiTurn-v0` | 汉诺塔 (多轮交互) |
| `Games/UltimateTicTacToe-v0` | 终极井字棋 |
| `Games/WildTicTacToe-v0` | 狂野井字棋 |
| `Games/WordSearch-v0` | 单词搜索 |
| `Games/Wordle-v0` | Wordle 猜单词 |

### Spatial (30 个)

2D/3D 空间导航与物体交互任务。

| 环境 ID | 简介 |
|---------|------|
| `Spatial/CollectHealth-v0` | 3D 空间收集生命值物品 |
| `Spatial/DoorKey-v0` | 2D 网格中找钥匙开门到达目标 |
| `Spatial/DynamicObstacles-v0` | 2D 动态障碍物躲避导航 |
| `Spatial/Empty-v0` | 2D 空房间导航基准 |
| `Spatial/FourRooms2D-v0` | 2D 四房间导航 |
| `Spatial/FourRooms3D-v0` | 3D 四房间导航 |
| `Spatial/Hallway-v0` | 3D 走廊导航 |
| `Spatial/LavaGap-v0` | 2D 跨越熔岩间隙 |
| `Spatial/Maze-v0` | 3D 迷宫导航 |
| `Spatial/MazeS2-v0` | 3D 迷宫 (中等规模) |
| `Spatial/MazeS3-v0` | 3D 迷宫 (大规模) |
| `Spatial/MazeS3Fast-v0` | 3D 迷宫 (大规模, 快速模式) |
| `Spatial/MultiRoom-v0` | 2D 多房间导航 |
| `Spatial/OneRoom-v0` | 3D 单房间导航 |
| `Spatial/OneRoomS6-v0` | 3D 单房间 (大规模) |
| `Spatial/OneRoomS6Fast-v0` | 3D 单房间 (大规模, 快速模式) |
| `Spatial/PickupObjects-v0` | 3D 空间拾取物体 |
| `Spatial/PutNext-v0` | 3D 空间将物体放到指定位置旁 |
| `Spatial/RoomObjects-v0` | 3D 多房间物体交互 |
| `Spatial/Sidewalk-v0` | 3D 人行道导航 |
| `Spatial/Sign-v0` | 3D 根据标识导航 |
| `Spatial/TMaze-v0` | 3D T 形迷宫 |
| `Spatial/TMazeLeft-v0` | 3D T 形迷宫 (左转) |
| `Spatial/TMazeRight-v0` | 3D T 形迷宫 (右转) |
| `Spatial/ThreeRooms-v0` | 3D 三房间导航 |
| `Spatial/Unlock-v0` | 2D 解锁房间 |
| `Spatial/WallGap-v0` | 3D 穿墙缺口导航 |
| `Spatial/YMaze-v0` | 3D Y 形迷宫 |
| `Spatial/YMazeLeft-v0` | 3D Y 形迷宫 (左转) |
| `Spatial/YMazeRight-v0` | 3D Y 形迷宫 (右转) |

### Temporal (13 个)

Sega Genesis 复古游戏，基于 Retro 模拟器。

| 环境 ID | 简介 |
|---------|------|
| `Temporal/Airstriker-v0` | 空中打击 |
| `Temporal/AlteredBeast-v0` | 兽王记 |
| `Temporal/CastleOfIllusion-v0` | 幻影城堡 |
| `Temporal/CastlevaniaBloodlines-v0` | 恶魔城血统 |
| `Temporal/Columns-v0` | 宝石方块 |
| `Temporal/DynamiteHeaddy-v0` | 炸弹人 |
| `Temporal/GoldenAxe-v0` | 战斧 |
| `Temporal/KidChameleon-v0` | 变色龙小子 |
| `Temporal/MortalKombatII-v0` | 真人快打 II |
| `Temporal/SpaceHarrierII-v0` | 太空哈利 II |
| `Temporal/StreetsOfRage2-v0` | 怒之铁拳 2 |
| `Temporal/Strider-v0` | 出击飞龙 |
| `Temporal/ThunderForceIII-v0` | 雷电战机 III |

---

## 其他环境 (3 个)

| 环境 ID | 简介 |
|---------|------|
| `Offline/SingleTurn-v0` | 通用离线单轮环境，支持加载 JSONL 多模态数据集 |
| `VLMEval-Base-v0` | VLMEval 评测基础环境 |
| `GenEval-v0` | 文生图评测环境 |

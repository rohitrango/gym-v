# gym-v

## 开发前可以先安装一下pre-commit (uv run pre-commit install)

## 阶段一

我们先给一些已经有的env集成进来，新增的env需要：
- 和原仓库的env的游戏行为一致
- 代码也尽量和原仓库保持一致（方便review）
- 如果原仓库是纯文本env，集成进来需要新增出图函数并且可以注意一下game description部分不要再包含原来的与纯文本caption / maze相关的内容
- 如果是纯文本env，在实现时候以注释形式写一下原本文本env的一个qa的case，新的game description还需要注意需要与文本env预期的回答format和现在的render的图对的上（可参考下rg在gym-v得写法

## 已接入环境列表

**总计: 137 个环境**

### GameRL (37 个环境)
包含经典游戏的强化学习版本和视觉问答(QA)版本。

#### 交互式游戏 (7个)
- `GameRL/Snake-v0`: 经典贪吃蛇
- `GameRL/Pacman-v0`: 经典吃豆人
- `GameRL/Tetris-v0`: 俄罗斯方块
- `GameRL/SpaceInvaders-v0`: 太空侵略者射击游戏
- `GameRL/Maze-v0`: 经典迷宫寻路
- `GameRL/Lifegame-v0`: 康威生命游戏模拟
- `GameRL/LangtonAnt-v0`: 兰顿蚂蚁移动行为模拟

#### 视觉问答单轮环境 (30个)
侧重于对当前游戏画面的视觉理解与单步决策：
- `GameRL/3DReconstruction-QA-v0`: 3D重建视觉理解
- `GameRL/ChessRanger-QA-v0`: 象棋场景问答
- `GameRL/Freecell-QA-v0`: 空当接龙
- `GameRL/Hue-QA-v0`: 颜色匹配游戏
- `GameRL/Jewel2-QA-v0`: 宝石消除
- `GameRL/Klondike-QA-v0`: 克朗代克纸牌
- `GameRL/LangtonAnt-QA-v0`: 兰顿蚂蚁
- `GameRL/Lifegame-QA-v0`: 生命游戏
- `GameRL/Maze-QA-v0`: 迷宫
- `GameRL/Maze3D-QA-v0`: 3D迷宫
- `GameRL/Minecraft-QA-v0`: 我的世界
- `GameRL/Minesweeper-QA-v0`: 扫雷
- `GameRL/Pacman-QA-v0`: 吃豆人
- `GameRL/PyramidChess-QA-v0`: 金字塔象棋
- `GameRL/RhythmGame-QA-v0`: 节奏游戏
- `GameRL/RubiksCube-QA-v0`: 魔方
- `GameRL/Snake-QA-v0`: 贪吃蛇
- `GameRL/Sokoban-QA-v0`: 推箱子
- `GameRL/SpaceInvaders-QA-v0`: 太空侵略者
- `GameRL/SpiderSolitaire-QA-v0`: 蜘蛛纸牌
- `GameRL/StarBattle-QA-v0`: 星星大战
- `GameRL/Sudoku-QA-v0`: 数独
- `GameRL/Tangram-QA-v0`: 七巧板
- `GameRL/Tents-QA-v0`: 帐篷谜题
- `GameRL/Tetris-QA-v0`: 俄罗斯方块
- `GameRL/TicTacToe-QA-v0`: 井字棋
- `GameRL/TuringMachine2d-QA-v0`: 2D图灵机
- `GameRL/UltraTicTacToe-QA-v0`: 终极井字棋
- `GameRL/WordSearch-QA-v0`: 单词搜索
- `GameRL/Zuma-QA-v0`: 祖玛

### TextArena (23 个环境)
文本游戏增强视觉渲染版本：
- `TextArena/Alquerque-v0`: 西班牙跳棋
- `TextArena/Breakthrough-v0`: 突围棋
- `TextArena/ConnectFour-v0`: 四子棋
- `TextArena/Crosswords-v0`: 英文填字游戏
- `TextArena/Crusade-v0`: 十字军棋
- `TextArena/FifteenPuzzle-v0`: 15数字推盘
- `TextArena/FrozenLake-v0`: 冰湖挑战
- `TextArena/Game2048-v0`: 2048数字合并游戏
- `TextArena/LightsOut-v0`: 点灯谜题
- `TextArena/LinesOfAction-v0`: 行动路线棋
- `TextArena/Minesweeper-v0`: 扫雷游戏
- `TextArena/Nim-v0`: 尼姆游戏
- `TextArena/Othello-v0`: 黑白棋
- `TextArena/PegJump-v0`: 孔明棋
- `TextArena/RushHour-v0`: 塞车时间
- `TextArena/SimpleTak-v0`: 简化Tak棋
- `TextArena/Sokoban-v0`: 推箱子
- `TextArena/Sudoku-v0`: 数独
- `TextArena/TowerOfHanoi-v0`: 汉诺塔
- `TextArena/UltimateTicTacToe-v0`: 终极井字棋
- `TextArena/WildTicTacToe-v0`: 狂野井字棋
- `TextArena/WordSearch-v0`: 单词搜索谜题
- `TextArena/Wordle-v0`: Wordle猜单词游戏

### ReasoningGym (19 个环境)
逻辑推理与算法类游戏：
- `ReasoningGym/Arc1D-v0`: 一维ARC推理任务
- `ReasoningGym/BinaryMatrix-v0`: 二进制矩阵逻辑
- `ReasoningGym/CircuitLogic-v0`: 电路逻辑模拟
- `ReasoningGym/GameOfLife-v0`: 康威生命游戏
- `ReasoningGym/Kakurasu-v0`: Kakurasu逻辑谜题
- `ReasoningGym/KnightSwap-v0`: 骑士交换
- `ReasoningGym/LargestIsland-v0`: 最大岛屿搜索
- `ReasoningGym/Maze-v0`: 迷宫寻路
- `ReasoningGym/MiniSudoku-v0`: 迷你数独(4x4/6x6)
- `ReasoningGym/NQueens-v0`: N皇后问题
- `ReasoningGym/RectangleCount-v0`: 矩形统计
- `ReasoningGym/RotateMatrix-v0`: 矩阵旋转
- `ReasoningGym/RottenOranges-v0`: 腐烂橘子模拟
- `ReasoningGym/ShortestPath-v0`: 最短路径规划
- `ReasoningGym/SpiralMatrix-v0`: 螺旋矩阵遍历
- `ReasoningGym/Sudoku-v0`: 9x9数独
- `ReasoningGym/Survo-v0`: Survo逻辑谜题
- `ReasoningGym/TowerOfHanoi-v0`: 汉诺塔
- `ReasoningGym/Tsumego-v0`: 围棋死活题(诘棋)

### Retro (13 个环境)
Sega Genesis 复古游戏：
- `Retro/Airstriker-v0`: 空中打击
- `Retro/AlteredBeast-v0`: 兽王记
- `Retro/CastleOfIllusion-v0`: 幻影城堡
- `Retro/CastlevaniaBloodlines-v0`: 恶魔城血统
- `Retro/Columns-v0`: 宝石方块
- `Retro/DynamiteHeaddy-v0`: 炸弹超人
- `Retro/GoldenAxe-v0`: 战斧
- `Retro/KidChameleon-v0`: 变色龙小子
- `Retro/MortalKombatII-v0`: 真人快打2
- `Retro/SpaceHarrierII-v0`: 太空哈利2
- `Retro/StreetsOfRage2-v0`: 怒之铁拳2
- `Retro/Strider-v0`: 忍者龙剑传
- `Retro/ThunderForceIII-v0`: 雷电战机3

### Perception (11 个环境)
图表与图形理解任务：
- `Perception/ChartToTable-v0`: 图表转表格
- `Perception/ContourPlot-v0`: 等高线图理解
- `Perception/DAGToTopoOrder-v0`: 有向无环图拓扑排序
- `Perception/FlowNetwork-v0`: 流网络分析
- `Perception/FunctionGraph-v0`: 函数图像理解
- `Perception/GraphToAdjacency-v0`: 图转邻接矩阵
- `Perception/GraphToMST-v0`: 图最小生成树
- `Perception/ParametricCurve-v0`: 参数曲线
- `Perception/PolarPlot-v0`: 极坐标图
- `Perception/TreeToTraversal-v0`: 树遍历
- `Perception/VectorField-v0`: 向量场可视化

### PettingZoo (8 个环境)
多智能体经典棋牌游戏，支持VLM友好的动作格式（需安装额外依赖: `uv sync --extra pettingzoo`）：
- `PettingZoo/Chess-v0`: 国际象棋（UCI格式: e2e4）
- `PettingZoo/ConnectFour-v0`: 四子棋（列索引: 0-6）
- `PettingZoo/GinRummy-v0`: 金罗美扑克（自然语言动作）
- `PettingZoo/Go-v0`: 围棋（坐标格式: A1-T19）
- `PettingZoo/LeducHoldem-v0`: Leduc扑克
- `PettingZoo/TexasHoldem-v0`: 德州扑克
- `PettingZoo/TexasHoldemNoLimit-v0`: 无限注德州扑克
- `PettingZoo/TicTacToe-v0`: 井字棋（坐标: 0-8）

### Sphinx (8 个环境)
视觉推理与模式识别：
- `Sphinx/OddOneOut-v0`: 找出不同项
- `Sphinx/OddOneOutPoly-v0`: 找出不同项（多边形）
- `Sphinx/SequenceCompletion-v0`: 序列补全
- `Sphinx/SequenceCompletionPoly-v0`: 序列补全（多边形）
- `Sphinx/SymmetryFill-v0`: 对称填充
- `Sphinx/SymmetryFillPoly-v0`: 对称填充（多边形）
- `Sphinx/TransformResult-v0`: 变换结果预测
- `Sphinx/TransformResultPoly-v0`: 变换结果预测（多边形）

### Minigrid (7 个环境)
简化的网格世界导航任务：
- `Minigrid/DoorKey-v0`: 钥匙开门
- `Minigrid/DynamicObstacles-v0`: 动态障碍物
- `Minigrid/Empty-v0`: 空房间
- `Minigrid/FourRooms-v0`: 四房间
- `Minigrid/LavaGap-v0`: 熔岩间隙
- `Minigrid/MultiRoom-v0`: 多房间
- `Minigrid/Unlock-v0`: 解锁房间

### VGRP (7 个环境)
视觉网格推理谜题：
- `VGRP/Battleships-v0`: 战舰谜题
- `VGRP/Binairo-v0`: 二进制谜题
- `VGRP/Futoshiki-v0`: 不等号数独
- `VGRP/Hitori-v0`: 数字消除
- `VGRP/Renzoku-v0`: 连续数字
- `VGRP/StarBattle-v0`: 星星大战
- `VGRP/Thermometers-v0`: 温度计谜题

### RLVE (3 个环境)
逻辑推理谜题环境：
- `RLVE/HitoriPuzzle-v0`: Hitori谜题
- `RLVE/LightUpPuzzle-v0`: 点灯谜题
- `RLVE/SkyscraperPuzzle-v0`: 摩天楼谜题

### Offline (1 个环境)
离线数据集环境：
- `Offline/SingleTurn-v0`: 通用单轮任务环境，支持加载JSONL离线多模态数据

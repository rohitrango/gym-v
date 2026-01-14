# gym-v

## 开发前可以先安装一下pre-commit (uv run pre-commit install)

## 阶段一

我们先给一些已经有的env集成进来，新增的env需要：
- 和原仓库的env的游戏行为一致
- 代码也尽量和原仓库保持一致（方便review）
- 如果原仓库是纯文本env，集成进来需要新增出图函数并且可以注意一下game description部分不要再包含原来的与纯文本caption / maze相关的内容
- 如果是纯文本env，在实现时候以注释形式写一下原本文本env的一个qa的case，新的game description还需要注意需要与文本env预期的回答format和现在的render的图对的上（可参考下rg在gym-v得写法

## 已接入环境列表

### ReasoningGym (推理类)
- `ReasoningGym/Sudoku-v0`: 经典的 9x9 数独游戏。
- `ReasoningGym/Maze-v0`: 迷宫寻路，寻找从起点到终点的路径。
- `ReasoningGym/NQueens-v0`: N皇后问题，在 NxN 棋盘上放置 N 个互不攻击的皇后。
- `ReasoningGym/GameOfLife-v0`: 康威生命游戏，模拟细胞种群演化。
- `ReasoningGym/TowerOfHanoi-v0`: 汉诺塔谜题，按规则移动圆盘。
- `ReasoningGym/KnightSwap-v0`: 骑士交换任务，在棋盘上移动并交换不同颜色的骑士。
- `ReasoningGym/MiniSudoku-v0`: 简化版数独游戏（4x4 或 6x6）。
- `ReasoningGym/Survo-v0`: Survo 逻辑谜题，根据行列之和填充数字。
- `ReasoningGym/Kakurasu-v0`: Kakurasu 逻辑谜题，根据行列权重选择方格。
- `ReasoningGym/Tsumego-v0`: 围棋死活题（诘棋），解决局部生存问题。
- `ReasoningGym/SpiralMatrix-v0`: 螺旋矩阵遍历任务。
- `ReasoningGym/RotateMatrix-v0`: 矩阵旋转，将矩阵按要求角度旋转。
- `ReasoningGym/BinaryMatrix-v0`: 二进制矩阵逻辑，处理 0/1 矩阵模式。
- `ReasoningGym/LargestIsland-v0`: 最大岛屿搜索，寻找最大的连通区域。
- `ReasoningGym/RottenOranges-v0`: 腐烂橘子模拟，计算完全腐烂所需时间。
- `ReasoningGym/ShortestPath-v0`: 最短路径规划，寻找两点间最小代价路径。
- `ReasoningGym/RectangleCount-v0`: 矩形统计，识别并计算图形中矩形数量。
- `ReasoningGym/CircuitLogic-v0`: 电路逻辑模拟，根据逻辑门关系解决谜题。
- `ReasoningGym/Arc1D-v0`: 一维 ARC 推理任务，模式识别与抽象逻辑推理。

### TextArena (文本类增强渲染)
- `TextArena/Crosswords-v0`: 英文填字游戏。
- `TextArena/FifteenPuzzle-v0`: 15 数字推盘。
- `TextArena/FrozenLake-v0`: 冰湖挑战，避开冰洞到达目标。
- `TextArena/Game2048-v0`: 2048 数字合并游戏。
- `TextArena/LightsOut-v0`: 点灯谜题，将所有灯熄灭。
- `TextArena/Minesweeper-v0`: 经典扫雷游戏。
- `TextArena/PegJump-v0`: 孔明棋（跳棋），通过跳跃消除棋子。
- `TextArena/RushHour-v0`: 塞车时间，移动车辆为目标车开路。
- `TextArena/Sokoban-v0`: 经典推箱子游戏。
- `TextArena/Sudoku-v0`: 文本渲染风格数独。
- `TextArena/TowerOfHanoi-v0`: 文本渲染风格汉诺塔。
- `TextArena/Wordle-v0`: 猜单词游戏，根据反馈线索猜五位单词。
- `TextArena/WordSearch-v0`: 单词搜索谜题，在字母阵列中寻找单词。

### GameRL (经典游戏 RL 版)
- `GameRL/Snake-v0`: 经典贪吃蛇。
- `GameRL/Pacman-v0`: 经典吃豆人。
- `GameRL/Tetris-v0`: 俄罗斯方块。
- `GameRL/SpaceInvaders-v0`: 太空侵略者射击游戏。
- `GameRL/Maze-v0`: 经典迷宫寻路。
- `GameRL/Lifegame-v0`: 康威生命游戏模拟。
- `GameRL/LangtonAnt-v0`: 兰顿蚂蚁移动行为模拟。
- `GameRL/Minesweeper-v0`: 扫雷逻辑推理。
- `GameRL/Sudoku-v0`: 数独求解博弈。

### GameRL QA (视觉问答/单轮)
- `GameRL/Snake-QA-v0` / `Pacman-QA-v0` 等共计 30 个 QA 环境，侧重于对当前游戏画面的视觉理解与单步决策问答（包含：魔方、空当接龙、三消、我的世界、祖玛、象棋等）。

### GUI
- `Webshop`
- `miniwob`

### Offline (离线数据)
- `Offline/SingleTurn-v0`: 通用单轮任务环境，支持加载 JSONL 离线多模态数据。

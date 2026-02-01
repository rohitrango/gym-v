# Difficulty 统一规范

本规范定义 gym-v 的 difficulty 设计与落地原则，目标是提供统一、可控、可比较的难度刻度，同时不破坏现有环境的传统用法。

## Goals

- 统一接口：所有环境统一接收 `difficulty: int | None`。
- 可比较：同一 difficulty 水平在同域环境中具有可比复杂度。
- 可控与可扩展：难度调参逻辑集中、可测试、可校准。
- 兼容传统方式：用户显式传入参数时，difficulty 不应覆盖。

## Non-goals

- 不追求跨域绝对可比（例如 GameRL vs RLVE 绝对难度排序）。
- 不要求所有环境都必须支持 difficulty 调参。

## API 约定

- `difficulty: int | None`
  - `None`：传统模式，保持原有随机/默认逻辑。
  - `int >= 0`：统一难度刻度，0 为最简单。
- 参数优先级：
  1) 显式参数（如 `max_n_m`, `steps`, `grid_size`）
  2) difficulty
  3) 环境默认逻辑
- `set_difficulty()` 不应覆盖显式参数。
- 对于未实现动态难度的环境，difficulty 仅在创建时生效（make-time 参数映射）。

## Controller 设计

难度逻辑集中在 `ParameterController` 家族：
- `LinearController`: 线性增长（规模、步数、密度等）。
- `StageController`: 分段档位（Easy/Medium/Hard）。
- `CompositeController`: 组合多个控制器。
- DefaultController: 无可控参数时，仅记录 difficulty。

环境应通过 `_apply_difficulty_parameters()` 应用参数；若无法应用，返回默认行为。
对于未实现 `_apply_difficulty_parameters()` 的环境，统一通过 `make()` 侧参数映射应用难度。

## Mapping 指南

建议将难度映射到“可观测复杂度”：
- 规模类：`N/M`、节点数、网格尺寸、对象数量。
- 密度类：障碍密度、边密度、遮挡比例、噪声强度。
- 过程类：步数上限、扰动次数、操作深度。
- 推理类：规则数量、约束数量、干扰项数量、上下文长度。

原则：
- 难度单调递增（difficulty 越大，复杂度不下降）。
- 保持环境内部的一致性（相同 difficulty 不要产生巨大波动）。
- 可复现（使用固定随机种子时应稳定）。

## 标定流程（无法直接计算的环境）

对于缺乏明确参数的环境，采用“可观测复杂度 → 标定映射 → 统一刻度”的流程：

1) 选择可观测指标
   - 成功率、平均解题步数、平均耗时、搜索节点数等。
2) 运行基线 solver 或启发式
   - 采样 N 个实例，记录指标分布。
3) 分位数映射为 difficulty
   - 例如 difficulty=0..10 对应 0%..100% 分位数。
4) 固化参数组合
   - 将 difficulty 对应的参数组合写入 controller。

无法标定时，保持 DefaultController（difficulty 记录但不改变参数）。

## 领域映射模板

以下模板用于快速为不同环境域补齐 difficulty 映射策略。优先选择“规模→密度→步数”的顺序，保持单调递增。

### RLVE

**典型参数类型**
- Grid/Matrix：`max_n_m`, `max_n`, `max_r_c`
- Sparsity/Density：`sparsity`, `edge_density`
- Steps/Depth：`steps`
- Range：`min_n`, `max_n`

**推荐映射**
- 网格类：先增尺寸，再增稀疏度（或障碍比例）
- 图类：先增节点数，再增边密度
- 过程类：按 `steps` 线性增长
- 区间类：扩展上界（`max_n`），下界保持

**示例配置**
- `Grid+Sparsity`: size from 2→8, sparsity from 0.3→0.7
- `Graph`: nodes 4→12, density 0.2→0.6
- `Puzzle steps`: 3→25, step_increment=2

### GameRL

**典型参数类型**
- 档位类：`Easy/Medium/Hard`（StageController）
- 尺寸类：`grid_size`, `maze_size`, `cols/rows`
- 深度类：`scramble_depth`, `level`

**推荐映射**
- 档位优先：difficulty 映射到 Easy/Medium/Hard
- 尺寸类：尺寸线性上升，必要时上限封顶
- 组合类：优先提升尺寸，再提升深度（或反之，保持单调）

**示例配置**
- Minesweeper: 0..2→Easy, 3..5→Medium, 6+→Hard
- Maze: size 5→15
- Rubik: 2x2/3x3 + scramble depth staged

### ReasoningGym

**典型参数类型**
- `dataset_kwargs` 内部参数
  - `grid_size_x/y`, `min_grid_size/max_grid_size`
  - `min_empty/max_empty`
  - `n`, `min_remove/max_remove`
  - `min_disks/max_disks`

**推荐映射**
- 直接驱动 dataset_kwargs
- 规模类参数同步（min=max=目标尺寸）
- 难度类参数（空格数/删除数）线性增长

**示例配置**
- Sudoku: `min_empty/max_empty` 随 difficulty 上升
- Maze: `min_grid_size=max_grid_size` 递增
- N-Queens: `n` 递增，`max_remove=n-1`

### Perception / Sphinx / TextArena（建议）

**可观测复杂度方向**
- Perception：噪声强度、遮挡比例、对象数量、图形复杂度
- Sphinx：样式复杂度、颜色数量、形状数量、干扰项
- TextArena：棋盘规模、障碍/干扰物数量、行动空间大小

**标定建议**
- 若缺少明确参数，使用基线 solver/启发式采样标定 difficulty。
 - 对未实现动态难度的环境，采用 make-time 参数映射注入难度。

## 测试要求

- controller 单元测试（单调性/上限/分段逻辑）。
- 关键环境在 difficulty 下的参数是否生效。
- 显式参数优先级测试。

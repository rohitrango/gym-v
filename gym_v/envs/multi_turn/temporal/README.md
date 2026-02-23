# Temporal 环境 (Retro Games)

基于 [stable-retro](https://github.com/Farama-Foundation/stable-retro) 封装的经典 Sega Genesis 游戏环境，共 13 个。

## 安装依赖

```bash
pip install stable-retro
```

## ROM 下载与导入

stable-retro 不自带商业游戏的 ROM 文件，需要自行下载并导入。

### 1. 商业游戏 ROM（12 个）

从 Archive.org 下载 Mega Drive Mini ROM 包：

```
https://archive.org/download/megadrivemini/Mega_Drive_Mini_Full_Set.zip
```

文件约 84MB，包含 Mega Drive Mini 收录的全部游戏。

下载后解压并导入：

```bash
# 解压外层 zip
unzip Mega_Drive_Mini_Full_Set.zip -d genesis_roms/

# 内层每个游戏也是 zip，再解压到一个平铺目录
mkdir genesis_roms_flat
for f in genesis_roms/Mega_Drive_Mini_ROMs/*.zip; do
    unzip -o "$f" -d genesis_roms_flat/
done

# 导入到 stable-retro（自动匹配 SHA1 哈希）
python -m stable_retro.import genesis_roms_flat/
```

成功导入后会显示：

```
Importing CastleOfIllusion-Genesis-v0
Importing ThunderForceIII-Genesis-v0
...
Imported 12 games
```

ROM 会被复制到 stable-retro 的 **Python 包安装目录**（如 `site-packages/stable_retro/data/stable/<game>/rom.md`）。

> **注意**：导入是把 ROM 复制到 stable-retro 的 site-packages 目录，而非项目目录。
> 如果切换了 Python 环境（新建 conda env、重装 stable-retro 等），需要重新执行导入。
> 本目录下的 `roms/Mega_Drive_Mini_Full_Set.zip` 是下载存档，可用于重新导入。

### 2. Airstriker ROM（1 个，免费 homebrew）

Airstriker 是免费的同人游戏，ROM 可从 OpenAI retro 仓库下载：

```bash
curl -sL "https://github.com/openai/retro/raw/master/retro/data/stable/Airstriker-Genesis/rom.md" \
    -o /path/to/stable_retro/data/stable/Airstriker-Genesis-v0/rom.md
```

其中 `/path/to/stable_retro/` 是 stable-retro 的安装目录，可通过以下命令查看：

```python
import stable_retro
print(stable_retro.data.path())
# 例如: /opt/conda/lib/python3.10/site-packages/stable_retro/data
```

## 验证 ROM 导入

```python
import stable_retro

games = [
    "Airstriker-Genesis-v0",
    "AlteredBeast-Genesis-v0",
    "CastleOfIllusion-Genesis-v0",
    "CastlevaniaBloodlines-Genesis-v0",
    "Columns-Genesis-v0",
    "DynamiteHeaddy-Genesis-v0",
    "GoldenAxe-Genesis-v0",
    "KidChameleon-Genesis-v0",
    "MortalKombatII-Genesis-v0",
    "SpaceHarrierII-Genesis-v0",
    "StreetsOfRage2-Genesis-v0",
    "Strider-Genesis-v0",
    "ThunderForceIII-Genesis-v0",
]

for game in games:
    try:
        path = stable_retro.data.get_romfile_path(game)
        print(f"OK  {game}")
    except FileNotFoundError:
        print(f"ERR {game} (ROM 缺失)")
```

## ROM 加载原理

`gym_v.make()` 时并不会自动下载或导入 ROM，整个链路如下：

```
gym_v.make("Temporal/GoldenAxe-v0")
  → RetroGymVEnv(game="GoldenAxe-Genesis-v0")        # 由 __init__.py 注册的 kwargs
    → stable_retro.make(game="GoldenAxe-Genesis-v0")  # 底层调用
      → 从 stable-retro 安装目录读取 ROM:
        site-packages/stable_retro/data/stable/GoldenAxe-Genesis-v0/rom.md
```

也就是说：
- `python -m stable_retro.import` 是一次性操作，把 ROM 复制到 stable-retro 的安装目录
- 之后每次 `make()` 都从那个目录读取，和项目里的 `roms/` 目录无关
- 如果 ROM 没导入，`make()` 会抛出 `FileNotFoundError: No romfiles found for game: ...`

查看 stable-retro 的数据目录：

```python
import stable_retro
print(stable_retro.data.path())
# 例如: /opt/conda/lib/python3.10/site-packages/stable_retro/data
```

## 在 gym-v 中使用

```python
import gym_v

# 创建环境
env = gym_v.make("Temporal/GoldenAxe-v0")

# 重置
obs, info = env.reset(seed=42)
# obs = {"agent_0": Observation(image=PIL.Image, text="Game: GoldenAxe-Genesis-v0 | ...")}

# 执行动作
obs, reward, terminated, truncated, info = env.step({"agent_0": "A"})        # 单键
obs, reward, terminated, truncated, info = env.step({"agent_0": "A+RIGHT"})  # 组合键
obs, reward, terminated, truncated, info = env.step({"agent_0": "NOOP"})     # 空操作

# 关闭
env.close()
```

## 环境列表

| gym-v 环境 ID | stable-retro game 名 | ROM 来源 |
|---|---|---|
| Temporal/Airstriker-v0 | Airstriker-Genesis-v0 | OpenAI retro 仓库（免费） |
| Temporal/AlteredBeast-v0 | AlteredBeast-Genesis-v0 | Mega Drive Mini ROM 包 |
| Temporal/CastleOfIllusion-v0 | CastleOfIllusion-Genesis-v0 | Mega Drive Mini ROM 包 |
| Temporal/CastlevaniaBloodlines-v0 | CastlevaniaBloodlines-Genesis-v0 | Mega Drive Mini ROM 包 |
| Temporal/Columns-v0 | Columns-Genesis-v0 | Mega Drive Mini ROM 包 |
| Temporal/DynamiteHeaddy-v0 | DynamiteHeaddy-Genesis-v0 | Mega Drive Mini ROM 包 |
| Temporal/GoldenAxe-v0 | GoldenAxe-Genesis-v0 | Mega Drive Mini ROM 包 |
| Temporal/KidChameleon-v0 | KidChameleon-Genesis-v0 | Mega Drive Mini ROM 包 |
| Temporal/MortalKombatII-v0 | MortalKombatII-Genesis-v0 | Mega Drive Mini ROM 包 |
| Temporal/SpaceHarrierII-v0 | SpaceHarrierII-Genesis-v0 | Mega Drive Mini ROM 包 |
| Temporal/StreetsOfRage2-v0 | StreetsOfRage2-Genesis-v0 | Mega Drive Mini ROM 包 |
| Temporal/Strider-v0 | Strider-Genesis-v0 | Mega Drive Mini ROM 包 |
| Temporal/ThunderForceIII-v0 | ThunderForceIII-Genesis-v0 | Mega Drive Mini ROM 包 |

## 按键说明

Genesis 控制器按键：`UP`, `DOWN`, `LEFT`, `RIGHT`, `A`, `B`, `C`, `X`, `Y`, `Z`, `START`, `MODE`

用 `+` 组合多个按键，如 `A+UP`、`B+LEFT+DOWN`。空操作用 `NOOP`。

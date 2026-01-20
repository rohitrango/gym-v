# Stable-Retro 环境

基于 [stable-retro](https://github.com/Farama-Foundation/stable-retro) 的经典游戏环境封装。

## 安装

stable-retro 是可选依赖，需要单独安装：

```bash
cd /home/mengfanqing/visiongym/gym-v
uv sync --extra stable-retro
```

## ROM 下载

stable-retro 支持 1000+ 经典游戏，但需要自行下载 ROM 文件。

### 方法一：从 Archive.org 下载（推荐）

1. 下载 No-Intro ROM 集合：
   - Genesis/Mega Drive: https://archive.org/download/megadrivemini/
   - 其他平台: https://archive.org/details/no-intro-collection

2. 导入 ROM：
```bash
python -m stable_retro.import /path/to/roms/
```

导入脚本会自动匹配 ROM 的 SHA-1 哈希值。

### 方法二：Steam Sega Genesis Classics

1. 在 Steam 购买 [Sega Genesis Classics](https://store.steampowered.com/app/34270)
2. 运行导入脚本：
```bash
python -m stable_retro.scripts.import_sega_classics
```

### 方法三：自有 ROM

从自己的游戏卡带提取 ROM，然后导入：
```bash
python -m stable_retro.import /path/to/your/roms/
```

## 使用方法

```python
import gym_v

# 创建环境
env = gym_v.make("Retro/GoldenAxe-v0")

# 重置环境
obs, info = env.reset()
# obs = {"agent_0": Observation(image=PIL.Image, text="Game: ...")}

# 执行动作（支持组合键）
obs, reward, terminated, truncated, info = env.step({"agent_0": "A"})       # 单键
obs, reward, terminated, truncated, info = env.step({"agent_0": "A+UP"})    # 组合键
obs, reward, terminated, truncated, info = env.step({"agent_0": "B+LEFT"})  # 组合键

# 渲染
image = env.render()  # 返回 PIL.Image

# 关闭
env.close()
```

## 示例游戏

### 免费自带游戏

| 环境 ID | 游戏 | 说明 |
|---------|------|------|
| Retro/Airstriker-v0 | Airstriker | stable-retro 自带，无需下载 |

### Mega Drive Mini ROM 包游戏

以下 12 个游戏来自同一个 ROM 包，下载一次即可全部使用：

**下载链接**: https://archive.org/download/megadrivemini/Mega_Drive_Mini_Full_Set.zip (约 84MB)

| 环境 ID | 游戏 |
|---------|------|
| Retro/GoldenAxe-v0 | Golden Axe |
| Retro/StreetsOfRage2-v0 | Streets of Rage 2 |
| Retro/MortalKombatII-v0 | Mortal Kombat II |
| Retro/Strider-v0 | Strider |
| Retro/CastleOfIllusion-v0 | Castle of Illusion |
| Retro/CastlevaniaBloodlines-v0 | Castlevania: Bloodlines |
| Retro/Columns-v0 | Columns |
| Retro/ThunderForceIII-v0 | Thunder Force III |
| Retro/SpaceHarrierII-v0 | Space Harrier II |
| Retro/AlteredBeast-v0 | Altered Beast |
| Retro/DynamiteHeaddy-v0 | Dynamite Headdy |
| Retro/KidChameleon-v0 | Kid Chameleon |

### 快速下载和导入

```bash
# 下载 Mega Drive Mini ROM 包
wget https://archive.org/download/megadrivemini/Mega_Drive_Mini_Full_Set.zip -O genesis_mini.zip

# 解压
unzip genesis_mini.zip -d genesis_roms/

# 导入到 stable-retro
python -m stable_retro.import genesis_roms/
```

### 更多 ROM 资源

| 平台 | 下载链接 | 说明 |
|------|----------|------|
| Genesis/Mega Drive | https://archive.org/download/megadrivemini/ | Mega Drive Mini 官方收录 |
| Genesis (完整) | https://archive.org/details/no-intro_romset | No-Intro 完整收录 |
| NES | https://archive.org/details/no-intro_romset | No-Intro 完整收录 |
| SNES | https://archive.org/details/no-intro_romset | No-Intro 完整收录 |
| Atari 2600 | https://archive.org/details/atari-2600-romset | Atari 2600 收录 |
| Game Boy | https://archive.org/details/no-intro_romset | No-Intro 完整收录 |

## 按键映射

Genesis/Mega Drive 控制器按键：
- 方向键: `UP`, `DOWN`, `LEFT`, `RIGHT`
- 动作键: `A`, `B`, `C`
- 扩展键: `X`, `Y`, `Z`
- 系统键: `START`, `MODE`

组合多个按键用 `+` 连接，例如 `A+UP`, `B+LEFT+DOWN`。

## 查看所有支持的游戏

```python
import stable_retro
print(stable_retro.data.list_games())
```

## 查看已导入的游戏

```python
import stable_retro

for game in sorted(stable_retro.data.list_games()):
    try:
        path = stable_retro.data.get_romfile_path(game)
        if path:
            print(f"✓ {game}")
    except:
        pass
```

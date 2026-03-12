# Temporal Environments (Retro Games)

Classic Sega Genesis game environments powered by [stable-retro](https://github.com/Farama-Foundation/stable-retro). 13 environments total.

## Install Dependencies

```bash
pip install stable-retro
```

## ROM Download & Import

stable-retro does not ship commercial game ROMs. You need to download and import them yourself.

### 1. Commercial Game ROMs (12 games)

Download the Mega Drive Mini ROM pack from Archive.org:

```
https://archive.org/download/megadrivemini/Mega_Drive_Mini_Full_Set.zip
```

The file is ~84 MB and contains all games from the Mega Drive Mini collection.

After downloading, extract and import:

```bash
# Extract the outer zip
unzip Mega_Drive_Mini_Full_Set.zip -d genesis_roms/

# Each game inside is also zipped — flatten them into one directory
mkdir genesis_roms_flat
for f in genesis_roms/Mega_Drive_Mini_ROMs/*.zip; do
    unzip -o "$f" -d genesis_roms_flat/
done

# Import into stable-retro (auto-matches by SHA1 hash)
python -m stable_retro.import genesis_roms_flat/
```

On success you will see:

```
Importing CastleOfIllusion-Genesis-v0
Importing ThunderForceIII-Genesis-v0
...
Imported 12 games
```

ROMs are copied into stable-retro's **site-packages directory** (e.g. `site-packages/stable_retro/data/stable/<game>/rom.md`).

> **Note:** Import copies ROMs into stable-retro's site-packages, not the project directory.
> If you switch Python environments (new conda env, reinstall stable-retro, etc.), you must re-run the import.

### 2. Airstriker ROM (1 game, free homebrew)

Airstriker is a free homebrew game. Download its ROM from the OpenAI retro repository:

```bash
curl -sL "https://github.com/openai/retro/raw/master/retro/data/stable/Airstriker-Genesis/rom.md" \
    -o /path/to/stable_retro/data/stable/Airstriker-Genesis-v0/rom.md
```

Find your stable-retro data directory with:

```python
import stable_retro
print(stable_retro.data.path())
# e.g.: /opt/conda/lib/python3.10/site-packages/stable_retro/data
```

## Verify ROM Import

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
        print(f"ERR {game} (ROM missing)")
```

## How ROM Loading Works

`gym_v.make()` does **not** download or import ROMs automatically. The call chain is:

```
gym_v.make("Temporal/GoldenAxe-v0")
  → RetroGymVEnv(game="GoldenAxe-Genesis-v0")        # registered in __init__.py
    → stable_retro.make(game="GoldenAxe-Genesis-v0")  # underlying call
      → reads ROM from stable-retro install directory:
        site-packages/stable_retro/data/stable/GoldenAxe-Genesis-v0/rom.md
```

In short:
- `python -m stable_retro.import` is a one-time operation that copies ROMs into stable-retro's install directory
- Every subsequent `make()` reads from that directory — the project's `roms/` folder is not involved
- If the ROM has not been imported, `make()` raises `FileNotFoundError: No romfiles found for game: ...`

## Usage

```python
import gym_v

# Create environment
env = gym_v.make("Temporal/GoldenAxe-v0")

# Reset
obs, info = env.reset(seed=42)
# obs = {"agent_0": Observation(image=PIL.Image, text="Game: GoldenAxe-Genesis-v0 | ...")}

# Step with actions
obs, reward, terminated, truncated, info = env.step({"agent_0": "A"})        # single key
obs, reward, terminated, truncated, info = env.step({"agent_0": "A+RIGHT"})  # combo
obs, reward, terminated, truncated, info = env.step({"agent_0": "NOOP"})     # no-op

# Close
env.close()
```

## Environment List

| Gym-V Environment ID | stable-retro Game Name | ROM Source |
|---|---|---|
| Temporal/Airstriker-v0 | Airstriker-Genesis-v0 | OpenAI retro repo (free) |
| Temporal/AlteredBeast-v0 | AlteredBeast-Genesis-v0 | Mega Drive Mini ROM pack |
| Temporal/CastleOfIllusion-v0 | CastleOfIllusion-Genesis-v0 | Mega Drive Mini ROM pack |
| Temporal/CastlevaniaBloodlines-v0 | CastlevaniaBloodlines-Genesis-v0 | Mega Drive Mini ROM pack |
| Temporal/Columns-v0 | Columns-Genesis-v0 | Mega Drive Mini ROM pack |
| Temporal/DynamiteHeaddy-v0 | DynamiteHeaddy-Genesis-v0 | Mega Drive Mini ROM pack |
| Temporal/GoldenAxe-v0 | GoldenAxe-Genesis-v0 | Mega Drive Mini ROM pack |
| Temporal/KidChameleon-v0 | KidChameleon-Genesis-v0 | Mega Drive Mini ROM pack |
| Temporal/MortalKombatII-v0 | MortalKombatII-Genesis-v0 | Mega Drive Mini ROM pack |
| Temporal/SpaceHarrierII-v0 | SpaceHarrierII-Genesis-v0 | Mega Drive Mini ROM pack |
| Temporal/StreetsOfRage2-v0 | StreetsOfRage2-Genesis-v0 | Mega Drive Mini ROM pack |
| Temporal/Strider-v0 | Strider-Genesis-v0 | Mega Drive Mini ROM pack |
| Temporal/ThunderForceIII-v0 | ThunderForceIII-Genesis-v0 | Mega Drive Mini ROM pack |

## Controls

Genesis controller buttons: `UP`, `DOWN`, `LEFT`, `RIGHT`, `A`, `B`, `C`, `X`, `Y`, `Z`, `START`, `MODE`

Combine multiple buttons with `+`, e.g. `A+UP`, `B+LEFT+DOWN`. Use `NOOP` for no action.

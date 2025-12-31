"""Minecraft QA environment based on GameRL."""

from __future__ import annotations

from importlib import resources
import random
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class GameRLMinecraftQAEnv(Env):
    """Minecraft QA environment.

    A Minecraft-like world where players build and explore with cubes.
    Players can place blocks adjacent to existing ones and in fluids.

    Question Types:
    - Scenery Recognition: Identify sceneries in the scene
    - Cube Count: Count total cubes in a cuboid structure
    - Cross Fluid: Minimum blocks needed to cross a river
    - Climb: Minimum blocks needed to reach a high block
    - Cross River Climb: Combined river crossing and climbing

    Args:
        space_ub: Upper bound for 3D space (x, y, z)
        question_type: Type of question to ask
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    QUESTION_TYPES = [
        {
            "id": "scenery",
            "name": "Scenery Recognition",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "cube_count",
            "name": "Cube Count",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "Target Perception",
        },
        {
            "id": "cross_fluid",
            "name": "Cross Fluid",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "climb",
            "name": "Climb to Block",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "cross_river_climb",
            "name": "Cross River and Climb",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
    ]

    ALL_SCENERY_TYPES = {
        "stone": {"name": "Stone", "color": (128, 128, 128)},
        "brick": {"name": "Brick", "color": (178, 34, 34)},
        "gold_ore": {
            "name": "Gold Ore",
            "color": (255, 215, 0),
        },
        "diamond_ore": {
            "name": "Diamond Ore",
            "color": (64, 224, 208),
        },
        "tnt": {"name": "TNT", "color": (220, 20, 60)},
        "pumpkin": {"name": "Pumpkin", "color": (255, 165, 0)},
        "ladder": {"name": "Ladder", "color": (139, 69, 19)},
        "river": {"name": "River", "color": (135, 206, 250)},
        "lava": {"name": "Lava", "color": (255, 140, 0)},
    }

    CANDIDATE_SCENERY_TYPES = {
        k: v for k, v in ALL_SCENERY_TYPES.items() if k not in ["stone"]
    }

    MINECRAFT_RULE = dedent("""
        You are provided with a game interface that mimics "Minecraft", where all objects are composed of equal-sized cubes. Players build and explore in this world. They have numerous blocks and can place them following the basic placement rules of "Minecraft." In simple terms, players can place blocks around their current position, and new blocks must be adjacent to existing ones (i.e., sharing a common face). Placing blocks in fluids (such as river water) is allowed, replacing the fluid at that position directly with the block. Players can also sometimes remove blocks.
    """).strip()

    def __init__(
        self,
        space_ub: tuple[int, int, int] = (5, 5, 5),
        question_type: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._space_ub = space_ub
        self._question_type = question_type

        # Game state
        self._blocks: list[dict[str, Any]] = []
        self._sceneries: dict[str, list[tuple[int, int, int]]] = {}
        self._current_question: dict[str, Any] = {}
        self._camera_angle = (20, 20)  # For isometric view

    @property
    def description(self) -> str:
        base_desc = dedent(f"""
            This is a Minecraft QA environment.

            {self.MINECRAFT_RULE}

            Question Types:
            - Scenery Recognition: Identify the sceneries in the scene
            - Cube Count: Count total cubes in a structure
            - Cross Fluid: Calculate minimum blocks to cross a river
            - Climb to Block: Calculate minimum blocks to reach a high block
            - Cross River and Climb: Combined challenge

            The system will present you with a scene and ask a specific question.
        """).strip()

        # Add question and answer format if question has been generated
        if hasattr(self, "_current_question") and self._current_question:
            desc = base_desc + "\n\n" + self._current_question["question"]
            desc += """

**Answer Format:**
Reply with only the answer (number or option number).

Examples:
- For multiple choice: 1, 2, 3, etc.
- For numbers: 42, 100, etc.

Do not include any explanation or extra text.
"""
            return desc.strip()

        return base_desc

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        # Select question type
        if self._question_type is None:
            question_type = random.choice(self.QUESTION_TYPES)["id"]
        else:
            question_type = self._question_type

        # Generate question
        if question_type == "scenery":
            self._current_question = self._generate_scenery_question()
        elif question_type == "cube_count":
            self._current_question = self._generate_cube_count_question()
        elif question_type == "cross_fluid":
            self._current_question = self._generate_cross_fluid_question()
        elif question_type == "climb":
            self._current_question = self._generate_climb_question()
        elif question_type == "cross_river_climb":
            self._current_question = self._generate_cross_river_climb_question()
        else:
            raise ValueError(f"Unknown question type: {question_type}")

        logger.info(f"Reset Minecraft QA (question: {question_type}).")

        obs = Observation(image=self.render(), text=self._current_question["question"])

        info = {
            "oracle_answer": self._current_question["answer"],
            "question_type": question_type,
        }

        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        reward = 0.0
        terminated = True
        truncated = False

        # Check answer
        correct = self._check_answer(action.strip())

        if correct:
            reward = 1.0
            response = "Correct!"
        else:
            reward = 0.0
            response = (
                f"Incorrect. The correct answer is: {self._current_question['answer']}"
            )

        info = {
            "correct": correct,
            "user_answer": action.strip(),
            "oracle_answer": self._current_question["answer"],
        }

        obs = Observation(image=self.render(), text=response)
        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image:
        """Render the Minecraft scene in isometric view."""
        # Image dimensions
        img_width, img_height = 800, 600
        img = Image.new("RGB", (img_width, img_height), (200, 200, 255))  # Sky blue
        draw = ImageDraw.Draw(img)

        try:
            label_font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 12)
        except Exception:
            label_font = ImageFont.load_default()

        # Draw blocks in isometric view
        # Sort blocks by depth for proper occlusion
        sorted_blocks = sorted(
            self._blocks, key=lambda b: (b["pos"][2], -b["pos"][1], b["pos"][0])
        )

        for block in sorted_blocks:
            x, y, z = block["pos"]
            color = block["color"]
            block_type = block.get("type", "cube")

            # Convert 3D coordinates to 2D isometric
            iso_x, iso_y = self._to_isometric(x, y, z)

            # Center and scale
            screen_x = img_width // 2 + iso_x * 20
            screen_y = img_height // 2 - iso_y * 20 + 100

            if block_type == "river" or block_type == "lava":
                # Draw as flat square
                self._draw_flat_block(
                    draw, screen_x, screen_y, color, block.get("alpha", 128)
                )
            elif block_type == "ladder":
                # Draw ladder as thin vertical lines
                self._draw_ladder(draw, screen_x, screen_y, (139, 69, 19))
            elif block_type == "player":
                # Draw player marker
                draw.ellipse(
                    [screen_x - 10, screen_y - 10, screen_x + 10, screen_y + 10],
                    fill=color,
                    outline=(0, 0, 0),
                    width=2,
                )
            else:
                # Draw as 3D cube
                self._draw_cube(draw, screen_x, screen_y, color)

        # Draw labels if needed
        if hasattr(self, "_labels"):
            for label_pos, label_text in self._labels:
                x, y, z = label_pos
                iso_x, iso_y = self._to_isometric(x, y, z)
                screen_x = img_width // 2 + iso_x * 20
                screen_y = img_height // 2 - iso_y * 20 + 100

                # Draw label background
                bbox = draw.textbbox(
                    (screen_x, screen_y - 20), label_text, font=label_font
                )
                draw.rectangle(bbox, fill=(255, 255, 200, 200), outline=(0, 0, 0))
                draw.text(
                    (screen_x, screen_y - 20),
                    label_text,
                    fill=(0, 0, 0),
                    font=label_font,
                )

        return img

    def _to_isometric(self, x: float, y: float, z: float) -> tuple[float, float]:
        """Convert 3D coordinates to 2D isometric projection."""
        iso_x = (x - z) * np.cos(np.radians(30))
        iso_y = (x + z) * np.sin(np.radians(30)) - y
        return iso_x, iso_y

    def _draw_cube(self, draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple):
        """Draw an isometric cube."""
        size = 18

        # Calculate cube vertices
        # Top face
        top_points = [
            (x, y - size),
            (x + size, y - size // 2),
            (x, y),
            (x - size, y - size // 2),
        ]

        # Left face
        left_points = [
            (x - size, y - size // 2),
            (x, y),
            (x, y + size),
            (x - size, y + size // 2),
        ]

        # Right face
        right_points = [
            (x, y),
            (x + size, y - size // 2),
            (x + size, y + size // 2),
            (x, y + size),
        ]

        # Darken colors for shading
        color_top = color
        color_left = tuple(max(0, int(c * 0.7)) for c in color)
        color_right = tuple(max(0, int(c * 0.85)) for c in color)

        # Draw faces
        draw.polygon(left_points, fill=color_left, outline=(0, 0, 0))
        draw.polygon(right_points, fill=color_right, outline=(0, 0, 0))
        draw.polygon(top_points, fill=color_top, outline=(0, 0, 0))

    def _draw_flat_block(
        self,
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        color: tuple,
        alpha: int = 128,
    ):
        """Draw a flat block (for water/lava)."""
        size = 18
        points = [
            (x, y),
            (x + size, y - size // 2),
            (x, y - size),
            (x - size, y - size // 2),
        ]
        # Use semi-transparent color
        transparent_color = color + (alpha,) if len(color) == 3 else color
        draw.polygon(points, fill=transparent_color, outline=(0, 0, 0))

    def _draw_ladder(self, draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple):
        """Draw a ladder."""
        size = 18
        # Draw vertical bars
        draw.line([(x - 5, y - size), (x - 5, y + size)], fill=color, width=2)
        draw.line([(x + 5, y - size), (x + 5, y + size)], fill=color, width=2)
        # Draw horizontal rungs
        for i in range(-1, 2):
            draw.line([(x - 5, y + i * 8), (x + 5, y + i * 8)], fill=color, width=2)

    def _clear_scene(self):
        """Clear the scene."""
        self._blocks = []
        self._sceneries = {k: [] for k in self.ALL_SCENERY_TYPES.keys()}
        if hasattr(self, "_labels"):
            delattr(self, "_labels")

    def _add_block(
        self, pos: tuple[int, int, int], color: tuple, block_type: str = "cube"
    ):
        """Add a block to the scene."""
        self._blocks.append({"pos": pos, "color": color, "type": block_type})

    def _generate_scenery_question(self) -> dict[str, Any]:
        """Generate scenery recognition question."""
        self._clear_scene()

        # Generate ground
        ground_color = (34, 139, 34) if random.random() < 0.5 else (139, 69, 19)
        for x in range(-10, 11):
            for z in range(-10, 11):
                self._add_block((x, -1, z), ground_color, "ground")

        # Select 3-5 sceneries
        num_scenery = random.randint(3, 5)
        selected_scenery = random.sample(
            list(self.CANDIDATE_SCENERY_TYPES.keys()), num_scenery
        )

        # Place sceneries
        occupied = set()
        for scenery in selected_scenery:
            if scenery == "river":
                self._place_river(occupied, is_x=random.choice([True, False]))
            elif scenery == "lava":
                self._place_lava(occupied)
            else:
                self._place_scenery(scenery, occupied)

        # Generate options
        idx, options = self._generate_scenery_options(selected_scenery)
        selected_names = [
            self.CANDIDATE_SCENERY_TYPES[s]["name"] for s in selected_scenery
        ]
        selected_names = [n.lower() if n != "TNT" else n for n in selected_names]

        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

        question = f"""{self.MINECRAFT_RULE}

The scene contains several sceneries. Sceneries can be:
1. Bricks
2. Gold Ore (embedded with gold-colored stone)
3. Diamond Ore (embedded with blue-green stone)
4. TNT (like dynamite, a red block labeled "TNT")
5. Pumpkin (a yellow block resembling a pumpkin)
6. Ladder (not a block, but a wooden ladder attached to blocks)
7. River (beneath ground level, blue, spanning the screen)
8. Lava (beneath ground level, consisting of orange and yellow)

Please select the option that correctly describes the sceneries contained in the image.

Options:
{options_text}"""

        return {"question": question, "answer": str(idx + 1), "options": options}

    def _generate_cube_count_question(self) -> dict[str, Any]:
        """Generate cube counting question."""
        self._clear_scene()

        # Generate ground
        ground_color = random.choice([(34, 139, 34), (139, 69, 19), (200, 200, 200)])
        for x in range(-10, 11):
            for z in range(-10, 11):
                self._add_block((x, -1, z), ground_color, "ground")

        # Create cuboid structure
        x_dim = random.randint(2, 6)
        y_dim = random.randint(1, 4)
        z_dim = random.randint(2, 6)

        # Random position
        base_x = random.randint(-3, 3)
        base_y = 0
        base_z = random.randint(-3, 3)

        # Random cube color
        cube_color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255),
        )

        # Place cubes
        for dx in range(x_dim):
            for dy in range(y_dim):
                for dz in range(z_dim):
                    self._add_block(
                        (base_x + dx, base_y + dy, base_z + dz), cube_color, "cube"
                    )

        answer = x_dim * y_dim * z_dim

        question = f"""{self.MINECRAFT_RULE}

How many cubes are there in total in the scene?"""

        return {"question": question, "answer": str(answer)}

    def _generate_cross_fluid_question(self) -> dict[str, Any]:
        """Generate cross fluid question."""
        self._clear_scene()

        # Generate ground (land)
        for x in range(-10, 11):
            for z in range(-10, 11):
                self._add_block((x, -1, z), (34, 139, 34), "ground")

        # Create river
        is_x = random.choice([True, False])
        fluid_width = random.randint(1, 4)

        if is_x:
            z_pos = random.randint(-3, 3)
            for x in range(-10, 11):
                for w in range(fluid_width):
                    z = z_pos - fluid_width // 2 + w
                    self._add_block((x, 0, z), (135, 206, 250), "river")

            # Player position
            dis = random.randint(1, 3)
            player_pos = (0, -1, z_pos - fluid_width // 2 - dis)
        else:
            x_pos = random.randint(-3, 3)
            for z in range(-10, 11):
                for w in range(fluid_width):
                    x = x_pos - fluid_width // 2 + w
                    self._add_block((x, 0, z), (135, 206, 250), "river")

            # Player position
            dis = random.randint(1, 3)
            player_pos = (x_pos - fluid_width // 2 - dis, -1, 0)

        self._add_block(player_pos, (255, 0, 0), "player")

        question = f"""{self.MINECRAFT_RULE}

There is a river in the scene, and the player's position is marked in red. The player needs to cross the river. Note that the river is presented in clearly bounded grid forms to allow players to discern the river's width. Suppose the player cannot wade directly through the water nor can they make a horizontal jump to cross the river directly.

Question: If the player wants to cross the river, what is the minimum number of blocks they need to consume?"""

        return {"question": question, "answer": str(fluid_width)}

    def _generate_climb_question(self) -> dict[str, Any]:
        """Generate climbing question."""
        self._clear_scene()

        # Generate ground
        for x in range(-10, 11):
            for z in range(-10, 11):
                self._add_block((x, -1, z), (34, 139, 34), "ground")

        # Random position for pillar
        x = random.randint(-3, 3)
        z = random.randint(-3, 3)
        num_stones = random.randint(0, 4)
        with_ladder = random.choices([True, False], weights=[0.3, 0.7])[0]

        # Random stone color
        stone_color = (
            random.randint(100, 200),
            random.randint(100, 200),
            random.randint(100, 200),
        )

        # Build pillar
        for y in range(num_stones):
            self._add_block((x, y, z), stone_color, "cube")
            if with_ladder:
                self._add_block((x, y, z), (139, 69, 19), "ladder")

        # Place target block
        block_name = random.choice(["pumpkin", "gold ore", "diamond ore"])
        block_color = {
            "pumpkin": (255, 165, 0),
            "gold ore": (255, 215, 0),
            "diamond ore": (64, 224, 208),
        }[block_name]
        self._add_block((x, num_stones, z), block_color, "cube")

        # Calculate answer
        required_blocks = max(0, (num_stones + 1) - 2) if not with_ladder else 0

        question = f"""{self.MINECRAFT_RULE}

There is a {block_name} block in the scene. If the {block_name} is simply on the ground (i.e., at the height of 1 blocks), the player can directly mine it. Otherwise, n (a positive integer) block(s) is/are under the {block_name} so that the {block_name} is a height of n+1. Under this circumstance:
- If there are ladders attached to the block(s) under the {block_name}, the player can climb up (if needed) and mine the {block_name} directly without consuming any blocks.
- If there are no ladders attached to the block(s) under the {block_name}. We should know that to mine the {block_name}, the player's upper body (assuming the player is 2 blocks tall) must not be below the {block_name}'s height. For example, if the {block_name} is located at a height of 2 blocks, the player can directly mine the {block_name}. However, if the {block_name} is at a height of 3 blocks, the player cannot mine it directly. The player can instead build blocks under his feet to raise his height. For instance, if the player builds 2 blocks under his feet, he will reach a height of 4 blocks.

Question: To obtain the {block_name}, what is the minimum number of blocks the player needs to consume?"""

        return {"question": question, "answer": str(required_blocks)}

    def _generate_cross_river_climb_question(self) -> dict[str, Any]:
        """Generate combined cross river and climb question."""
        self._clear_scene()

        # Generate ground
        for x in range(-10, 11):
            for z in range(-10, 11):
                self._add_block((x, -1, z), (34, 139, 34), "ground")

        # Create river (always along x-axis)
        river_width = random.randint(1, 4)
        z_river = 0
        for x in range(-10, 11):
            for w in range(river_width):
                z = z_river - river_width // 2 + w
                self._add_block((x, 0, z), (135, 206, 250), "river")

        # Player position
        same_side = random.choices([True, False], weights=[0.3, 0.7])[0]
        dis = random.randint(1, 2)
        player_pos = (0, -1, -river_width // 2 - dis)
        self._add_block(player_pos, (255, 0, 0), "player")

        # Target position
        if same_side:
            target_x = random.randint(-3, 3)
            target_z = random.randint(-5, -(river_width // 2) - 1)
        else:
            target_x = random.randint(-3, 3)
            target_z = random.randint((river_width // 2) + 1, 5)

        # Build pillar at target
        stone_height = random.randint(0, 4)
        with_ladder = random.choices([True, False], weights=[0.3, 0.7])[0]
        stone_color = (
            random.randint(100, 200),
            random.randint(100, 200),
            random.randint(100, 200),
        )

        for y in range(stone_height):
            self._add_block((target_x, y, target_z), stone_color, "cube")
            if with_ladder:
                self._add_block((target_x, y, target_z), (139, 69, 19), "ladder")

        # Place target block
        block_name = random.choice(["pumpkin", "gold ore", "diamond ore"])
        block_color = {
            "pumpkin": (255, 165, 0),
            "gold ore": (255, 215, 0),
            "diamond ore": (64, 224, 208),
        }[block_name]
        self._add_block((target_x, stone_height, target_z), block_color, "cube")

        # Calculate answer
        blocks_for_river = river_width if not same_side else 0
        blocks_for_climbing = max(0, (stone_height + 1) - 2) if not with_ladder else 0
        answer = blocks_for_river + blocks_for_climbing

        question = f"""{self.MINECRAFT_RULE}

There is a river and a {block_name} block in the scene, and the player's position is marked in red. Note that the river is presented in clearly bounded grid forms to allow players to discern the river's width. Suppose the player cannot wade directly through the water nor can they make a horizontal jump to cross the river directly.
Additionally, if the {block_name} is simply on the ground (i.e., at the height of 1 blocks), the player can directly mine it. Otherwise, n (a positive integer) block(s) is/are under the {block_name} so that the {block_name} is a height of n+1. Under this circumstance:
- If there are ladders attached to the block(s) under the {block_name}, the player can climb up (if needed) and mine the {block_name} directly without consuming any blocks.
- If there are no ladders attached to the block(s) under the {block_name}. We should know that to mine the {block_name}, the player's upper body (assuming the player is 2 blocks tall) must not be below the {block_name}'s height. For example, if the {block_name} is located at a height of 2 blocks, the player can directly mine the {block_name}. However, if the {block_name} is at a height of 3 blocks, the player cannot mine it directly. The player can instead build blocks under his feet to raise his height. For instance, if the player builds 2 blocks under his feet, he will reach a height of 4 blocks.
On rare occasions, the red sign indicating the player's position may be fully blocked by the blocks, thus making it invisible. In this case, we consider that the player and the {block_name} are on the same side of the river.

Question: To obtain the {block_name}, what is the minimum number of blocks the player needs to consume?"""

        return {"question": question, "answer": str(answer)}

    def _place_scenery(self, scenery: str, occupied: set):
        """Place a scenery block."""
        # Find valid position
        for _ in range(100):
            x = random.randint(-5, 5)
            z = random.randint(-5, 5)
            if (x, 0, z) not in occupied:
                occupied.add((x, 0, z))
                color = self.ALL_SCENERY_TYPES[scenery]["color"]
                self._add_block((x, 0, z), color, scenery)
                self._sceneries[scenery].append((x, 0, z))
                break

    def _place_river(self, occupied: set, is_x: bool):
        """Place a river."""
        width = random.randint(1, 3)
        if is_x:
            z = random.randint(-3, 3)
            for x in range(-10, 11):
                for w in range(width):
                    pos = (x, 0, z - width // 2 + w)
                    self._add_block(pos, (135, 206, 250), "river")
                    occupied.add(pos)
                    self._sceneries["river"].append(pos)
        else:
            x = random.randint(-3, 3)
            for z in range(-10, 11):
                for w in range(width):
                    pos = (x - width // 2 + w, 0, z)
                    self._add_block(pos, (135, 206, 250), "river")
                    occupied.add(pos)
                    self._sceneries["river"].append(pos)

    def _place_lava(self, occupied: set):
        """Place lava blocks."""
        for _ in range(100):
            x = random.randint(-5, 5)
            z = random.randint(-5, 5)
            if (x, -1, z) not in occupied:
                occupied.add((x, -1, z))
                self._add_block((x, -1, z), (255, 140, 0), "lava")
                self._sceneries["lava"].append((x, -1, z))
                # Maybe add second block
                if random.random() < 0.5:
                    dx, dz = random.choice([(1, 0), (0, 1)])
                    if (x + dx, -1, z + dz) not in occupied:
                        occupied.add((x + dx, -1, z + dz))
                        self._add_block((x + dx, -1, z + dz), (255, 140, 0), "lava")
                        self._sceneries["lava"].append((x + dx, -1, z + dz))
                break

    def _generate_scenery_options(
        self, correct_scenery: list[str]
    ) -> tuple[int, list[str]]:
        """Generate multiple choice options for scenery question."""
        options = []
        correct_names = [
            self.CANDIDATE_SCENERY_TYPES[s]["name"] for s in correct_scenery
        ]
        correct_option = ", ".join(sorted(correct_names))
        options.append(correct_option)

        all_scenery = list(self.ALL_SCENERY_TYPES.keys())
        if "stone" in all_scenery:
            all_scenery.remove("stone")

        # Generate 7 wrong options
        while len(options) < 8:
            wrong_scenery = random.sample(all_scenery, len(correct_scenery))
            wrong_names = [self.ALL_SCENERY_TYPES[s]["name"] for s in wrong_scenery]
            wrong_option = ", ".join(sorted(wrong_names))
            if wrong_option not in options:
                options.append(wrong_option)

        random.shuffle(options)
        idx = options.index(correct_option)

        # Maybe shuffle order within options
        if random.random() < 0.5:
            for i in range(len(options)):
                parts = options[i].split(", ")
                random.shuffle(parts)
                options[i] = ", ".join(parts)

        return idx, options

    def _check_answer(self, action: str) -> bool:
        """Check if answer is correct."""
        correct_answer = self._current_question["answer"]
        return action.strip().lower() == correct_answer.strip().lower()

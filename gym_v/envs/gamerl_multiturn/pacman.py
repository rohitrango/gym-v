"""Pacman game based on GameRL."""

from __future__ import annotations

from collections import deque
from importlib import resources
import random
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class Ghost:
    """Class representing a ghost in the Pac-Man game."""

    def __init__(self, name: str, position: tuple[int, int], game: GameRLPacmanEnv):
        """Initialize a Ghost instance.

        Args:
            name: Name of the ghost (e.g., 'Pinky', 'Blinky').
            position: Initial position (row, col).
            game: Reference to the GameRLPacmanEnv instance.
        """
        self.name = name
        self.game = game
        self.position = position
        self.path: list[tuple[int, int]] = []

    def update_direction(self) -> None:
        """Update the ghost's path based on its target using BFS."""
        if self.name == "Pinky":
            target = self.game._get_pinky_target()
        else:  # Blinky
            target = self.game._pacman_position

        if target:
            self.path = self.game._bfs(self.position, target)

    def move(self) -> None:
        """Move the ghost along the path towards its target."""
        if not self.path or len(self.path) < 2:
            return

        self.position = self.path[1]
        self.path.pop(0)


class GameRLPacmanEnv(Env):
    """Pacman game environment.

    Control Pac-Man to eat beans while avoiding ghosts.

    Args:
        grid_size: Size of the grid (default 16)
        wall_ratio: Ratio of internal walls (default 0.1)
        cell_size: Size of each cell in pixels for rendering (default 25)
    """

    gamerl_assets_dir = resources.files("gym_v.envs.gamerl") / "assets"
    assets_dir = resources.files("gym_v.envs") / "assets"

    # Direction mappings
    DIRECTIONS = {
        "up": (-1, 0),
        "w": (-1, 0),
        "down": (1, 0),
        "s": (1, 0),
        "left": (0, -1),
        "a": (0, -1),
        "right": (0, 1),
        "d": (0, 1),
    }

    # Direction names for internal use
    DIR_NAMES = {
        (-1, 0): "UP",
        (1, 0): "DOWN",
        (0, -1): "LEFT",
        (0, 1): "RIGHT",
    }

    # Colors
    COLORS = {
        "background": (0, 0, 0),  # Black
        "wall": (0, 0, 139),  # Deep blue
        "bean": (255, 255, 0),  # Yellow
        "text": (255, 255, 255),  # White
        "score": (255, 255, 0),  # Yellow
        "game_over": (255, 0, 0),  # Red
    }

    def __init__(
        self,
        grid_size: int = 16,
        wall_ratio: float = 0.1,
        cell_size: int = 25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._grid_size = grid_size
        self._wall_ratio = wall_ratio
        self._cell_size = cell_size
        self._margin = 30  # Margin for coordinate labels
        self._score_height = 20  # Height for score display

        # Game state (initialized in reset)
        self._walls: set[tuple[int, int]] = set()
        self._beans: set[tuple[int, int]] = set()
        self._pacman_position: tuple[int, int] = (1, 1)
        self._direction: str = "RIGHT"
        self._ghosts: list[Ghost] = []
        self._score: int = 0
        self._game_over: bool = False

        # Load character images
        self._pacman_image = self._load_image("pacman.png")
        self._ghost_images = {
            "Pinky": self._load_image("Pinky.png"),
            "Blinky": self._load_image("Blinky.png"),
        }

    def _load_image(self, filename: str) -> Image.Image | None:
        """Load and scale an image from assets."""
        image_path = self.gamerl_assets_dir / filename
        try:
            if image_path.is_file():
                img = Image.open(str(image_path))
                return img.resize(
                    (self._cell_size, self._cell_size), Image.Resampling.LANCZOS
                )
        except Exception as e:
            logger.warning(f"Error loading image {filename}: {e}")
        return None

    @property
    def description(self) -> str:
        return dedent("""
            # Game Overview
            Pac-Man is a maze arcade game where the player controls Pac-Man to eat as many beans as possible while avoiding ghosts. If a ghost catches Pac-Man, the game ends.

            # Basic Elements
            - **Pac-Man**: The yellow circular character that the player controls
            - **Beans**: Yellow dots that Pac-Man can eat to score points
            - **Walls**: Blue barriers that restrict movement
            - **Ghosts**: Two ghosts (Pinky and Blinky) that chase Pac-Man

            # Game Rules
            - Pac-Man must eat beans while avoiding ghosts
            - Each bean eaten adds 1 point to the score
            - The game ends if a ghost catches Pac-Man
            - Movement is restricted by walls

            # Movement and Direction
            - Pac-Man's mouth opening indicates its current direction
            - The direction can be UP, DOWN, LEFT, or RIGHT
            - Neither Pac-Man nor ghosts can move through walls

            # **Ghost Behavior**
            - **Pinky** (Pink Ghost): Targets up to 4 spaces ahead of Pac-Man's current position and direction (stops at walls)
            - **Blinky** (Red Ghost): Directly targets Pac-Man's current position
            - Both ghosts follow a movement priority system based on the direction they are trying to move:
              - When moving in more than one direction is optimal, the priority order for both ghosts is **UP > DOWN > LEFT > RIGHT**.
              - This means if a ghost has multiple possible directions to move in, it will first attempt to move **UP** if possible, then **DOWN**, followed by **LEFT**, and finally **RIGHT** if all other directions are blocked.

            # Board Layout
            - The board is surrounded by walls on all four sides
            - Position (0,0) is located at the top-left corner wall
            - Movement grid uses (row, column) coordinates

            # Scoring
            The score equals the total number of beans eaten by Pac-Man
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)

        self._game_over = False
        self._score = 0
        self._direction = "RIGHT"

        # Create walls
        self._walls = self._create_outer_walls()
        self._add_internal_walls()

        # Initialize beans
        self._initialize_beans()

        # Initialize Pac-Man position
        self._pacman_position = self._get_random_start_position()

        # Remove bean from Pac-Man's starting position
        if self._pacman_position in self._beans:
            self._beans.remove(self._pacman_position)
            self._score += 1

        # Initialize ghosts
        self._ghosts = []
        self._add_ghosts()

        logger.info("Reset Pacman game.")

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, {}

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        reward = 0.0
        terminated = False
        truncated = False

        if self._game_over:
            obs = Observation(image=self.render(), text=self._get_observation_text())
            return obs, reward, True, truncated, info

        # Normalize action
        action_lower = action.lower().strip()

        # Check for invalid action
        if action_lower not in self.DIRECTIONS:
            info["invalid_action"] = True
            obs = Observation(image=self.render(), text=self._get_observation_text())
            return obs, reward, terminated, truncated, info

        info["invalid_action"] = False

        # Get direction
        dr, dc = self.DIRECTIONS[action_lower]
        row, col = self._pacman_position
        new_pos = (row + dr, col + dc)

        # Check for wall collision
        if new_pos not in self._walls:
            self._pacman_position = new_pos
            self._direction = self.DIR_NAMES[(dr, dc)]

            # Check for bean eating
            if self._pacman_position in self._beans:
                self._beans.remove(self._pacman_position)
                self._score += 1
                reward = 1.0
                info["ate_bean"] = True

        # Move ghosts
        for ghost in self._ghosts:
            ghost.update_direction()
            ghost.move()

            # Check for collision with Pac-Man
            if ghost.position == self._pacman_position:
                self._game_over = True
                terminated = True
                info["death_reason"] = f"caught_by_{ghost.name}"

        # Check if all beans eaten
        if not self._beans and not self._game_over:
            self._game_over = True
            terminated = True
            info["win"] = True

        info["score"] = self._score

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image:
        """Render the game state as a PIL Image."""
        img_width = self._margin + self._grid_size * self._cell_size
        img_height = (
            self._score_height + self._margin + self._grid_size * self._cell_size
        )

        img = Image.new("RGB", (img_width, img_height), self.COLORS["background"])
        draw = ImageDraw.Draw(img)

        # Try to load font
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 12)
        else:
            font = ImageFont.load_default()

        # Draw score
        score_text = f"Score: {self._score}"
        draw.text((10, 2), score_text, fill=self.COLORS["score"], font=font)

        # Draw coordinate labels
        for i in range(self._grid_size):
            # Column numbers (top)
            x = self._margin + i * self._cell_size + self._cell_size // 2
            draw.text(
                (x, self._score_height + 2),
                str(i),
                fill=self.COLORS["text"],
                font=font,
                anchor="mt",
            )
            # Row numbers (left)
            y = (
                self._score_height
                + self._margin
                + i * self._cell_size
                + self._cell_size // 2
            )
            draw.text(
                (2, y),
                str(i),
                fill=self.COLORS["text"],
                font=font,
                anchor="lm",
            )

        # Draw walls
        for row, col in self._walls:
            x = self._margin + col * self._cell_size
            y = self._score_height + self._margin + row * self._cell_size
            draw.rectangle(
                [x, y, x + self._cell_size, y + self._cell_size],
                fill=self.COLORS["wall"],
            )

        # Draw beans
        for row, col in self._beans:
            cx = self._margin + col * self._cell_size + self._cell_size // 2
            cy = (
                self._score_height
                + self._margin
                + row * self._cell_size
                + self._cell_size // 2
            )
            radius = self._cell_size // 6
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=self.COLORS["bean"],
            )

        # Draw ghosts
        for ghost in self._ghosts:
            row, col = ghost.position
            x = self._margin + col * self._cell_size
            y = self._score_height + self._margin + row * self._cell_size
            ghost_img = self._ghost_images.get(ghost.name)
            if ghost_img:
                img.paste(ghost_img, (x, y))

        # Draw Pac-Man
        row, col = self._pacman_position
        x = self._margin + col * self._cell_size
        y = self._score_height + self._margin + row * self._cell_size
        if self._pacman_image:
            rotated = self._rotate_image(self._pacman_image, self._direction)
            img.paste(rotated, (x, y))

        # Draw game over text
        if self._game_over:
            large_font_path = self.assets_dir / "DejaVuSans.ttf"
            if large_font_path.exists():
                large_font = ImageFont.truetype(str(large_font_path), 36)
            else:
                large_font = ImageFont.load_default()

            text = "GAME OVER"
            bbox = draw.textbbox((0, 0), text, font=large_font)
            text_width = bbox[2] - bbox[0]
            text_x = (img_width - text_width) // 2
            text_y = img_height // 2 - 20
            draw.text(
                (text_x, text_y), text, fill=self.COLORS["game_over"], font=large_font
            )

        return img

    def _rotate_image(self, image: Image.Image, direction: str) -> Image.Image:
        """Rotate image based on direction."""
        if direction == "LEFT":
            return image.rotate(180)
        elif direction == "UP":
            return image.rotate(90)
        elif direction == "DOWN":
            return image.rotate(-90)
        return image  # RIGHT or default

    def _create_outer_walls(self) -> set[tuple[int, int]]:
        """Create outer walls around the grid."""
        walls = set()
        for row in range(self._grid_size):
            for col in range(self._grid_size):
                if (
                    row == 0
                    or row == self._grid_size - 1
                    or col == 0
                    or col == self._grid_size - 1
                ):
                    walls.add((row, col))
        return walls

    def _add_internal_walls(self) -> None:
        """Add internal walls randomly based on wall ratio."""
        total_cells = self._grid_size * self._grid_size
        num_internal_walls = int(total_cells * self._wall_ratio)

        available_positions = [
            (row, col)
            for row in range(1, self._grid_size - 1)
            for col in range(1, self._grid_size - 1)
            if (row, col) not in self._walls
        ]

        random.shuffle(available_positions)

        internal_walls: set[tuple[int, int]] = set()
        for position in available_positions:
            if len(internal_walls) >= num_internal_walls:
                break

            row, col = position
            neighbors = [
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ]
            existing_wall_neighbors = [n for n in neighbors if n in internal_walls]

            if len(existing_wall_neighbors) < 2:
                internal_walls.add(position)

        self._walls.update(internal_walls)

    def _initialize_beans(self) -> None:
        """Initialize beans in all non-wall cells."""
        self._beans = set(
            (row, col)
            for row in range(self._grid_size)
            for col in range(self._grid_size)
            if (row, col) not in self._walls
        )

    def _get_random_start_position(self) -> tuple[int, int]:
        """Get a random starting position that is not a wall."""
        available = [
            (row, col)
            for row in range(self._grid_size)
            for col in range(self._grid_size)
            if (row, col) not in self._walls
        ]
        return random.choice(available)

    def _add_ghosts(self) -> None:
        """Initialize and add ghosts to the game."""
        for name in ["Pinky", "Blinky"]:
            available = [
                (row, col)
                for row in range(self._grid_size)
                for col in range(self._grid_size)
                if (row, col) not in self._walls
                and (row, col) != self._pacman_position
                and all(g.position != (row, col) for g in self._ghosts)
            ]
            if available:
                pos = random.choice(available)
                self._ghosts.append(Ghost(name, pos, self))

    def _get_pinky_target(self) -> tuple[int, int]:
        """Calculate Pinky's target: 4 cells ahead of Pac-Man."""
        row, col = self._pacman_position
        target = (row, col)

        dr, dc = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
        }.get(self._direction, (0, 1))

        for _ in range(4):
            next_cell = (target[0] + dr, target[1] + dc)
            if (
                0 <= next_cell[0] < self._grid_size
                and 0 <= next_cell[1] < self._grid_size
                and next_cell not in self._walls
            ):
                target = next_cell
            else:
                break

        return target

    def _bfs(
        self, start: tuple[int, int], goal: tuple[int, int], max_depth: int = 1000
    ) -> list[tuple[int, int]]:
        """Perform BFS to find the shortest path from start to goal."""
        queue: deque[list[tuple[int, int]]] = deque()
        queue.append([start])
        visited: set[tuple[int, int]] = {start}
        depth = 0

        while queue and depth < max_depth:
            path = queue.popleft()
            current = path[-1]
            depth += 1

            if current == goal:
                return path

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                if (
                    0 <= neighbor[0] < self._grid_size
                    and 0 <= neighbor[1] < self._grid_size
                    and neighbor not in visited
                    and neighbor not in self._walls
                ):
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return []

    def _get_observation_text(self) -> str:
        """Get text description of current game state."""
        if self._game_over:
            if not self._beans:
                return f"You Win! Final Score: {self._score}"
            return f"Game Over! Final Score: {self._score}"

        ghost_info = ", ".join(f"{g.name}: {g.position}" for g in self._ghosts)
        return (
            f"Pac-Man: {self._pacman_position}, Direction: {self._direction}, "
            f"Score: {self._score}, Ghosts: [{ghost_info}]"
        )

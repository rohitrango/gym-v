"""Pyramid Chess QA environment for gym-v.

Pyramid Chess is a strategic game where two players alternate placing colored balls
on a pyramid structure. The game has complex mechanics involving 2x2 block formation
and take-back mechanisms.

This environment provides 6 question types about game states and optimal strategies.
"""

from __future__ import annotations

import random
from textwrap import dedent
from typing import Any

from PIL import Image

from gym_v import Env, Observation, get_logger
from gym_v.utils.gamerl_utils import build_description, score_exact

logger = get_logger()

# Constants
PLAYER_0 = 0
PLAYER_1 = 1
MAX_TURN_NUM = {"Easy": 13, "Medium": 29, "Hard": 54}
PLOT_LEVEL = {"Easy": 3, "Medium": 4, "Hard": 5}
NUM_BALLS = {"Easy": 7, "Medium": 15, "Hard": 27}


# ============================================================================
# Core Game Classes
# ============================================================================


class Grid:
    """Represents a single position in the pyramid."""

    def __init__(self, level: int, id: int, legal: bool, board_level: int):
        self.Level = level  # Level of grid in pyramid
        self.Position = [
            int(id / (board_level - level)),
            int(id % (board_level - level)),
        ]

        self.Available = True  # If this place has no ball
        self.Legal = legal  # If a ball can be placed here
        self.Can_be_taken = False  # If the ball here can be taken

        self.Base = []  # Balls under this ball
        self.Color = None  # Color of ball (None, PLAYER_0, or PLAYER_1)
        self.Upper = 0  # How many balls press this ball

        self.counted = False

    def put(self, color: int):
        """Place a ball of given color at this position."""
        if self.Legal:
            self.Available = False
            self.Legal = False
            self.Can_be_taken = True

            self.Color = color
            for item in self.Base:
                item.Can_be_taken = False
                item.Upper += 1
            return True
        else:
            return False

    def take(self):
        """Remove the ball from this position."""
        if not self.Available:
            self.Available = True
            self.Legal = True
            self.Can_be_taken = False

            self.Color = None
            for item in self.Base:
                item.Upper -= 1
                if item.Upper == 0:
                    item.Can_be_taken = True
            return True
        else:
            return False


class Board:
    """Represents the pyramid chess board."""

    def __init__(self, level: int = 4):
        self.Level = level
        self.Board = []
        for i in range(level):
            num_plot = (level - i) ** 2
            if i == 0:
                sub_board = [Grid(0, id, True, level) for id in range(num_plot)]
            else:
                sub_board = [Grid(i, id, False, level) for id in range(num_plot)]

            self.Board.append(sub_board)
        self.construct_layer(self.Board)
        self.counter = 0

    def board_dict(self):
        """Convert board to dictionary representation."""
        board_dict = dict()
        for level in range(self.Level):
            count = 0
            whole_list = []
            row_list = []
            for item in self.Board[level]:
                if item.Color is None:
                    row_list.append("--")
                elif item.Color == PLAYER_0:
                    row_list.append("P0")
                else:
                    row_list.append("P1")
                count += 1
                if count == self.Level - level:
                    count = 0
                    whole_list.append(row_list)
                    row_list = []
            board_dict[level] = whole_list
        return board_dict

    def __getitem__(self, index):
        i, j, k = index
        i, j, k = int(i), int(j), int(k)
        level = self.Level
        if i >= 0 and i < level:
            num = j * (level - i) + k
            if num < (level - i) * (level - i):
                return self.Board[i][j * (level - i) + k]
        raise ValueError("Index out of bound")

    def construct_layer(self, Board):
        """Construct the base relationships between levels."""
        for level in range(self.Level):
            if level == 0:
                continue
            else:
                for item in Board[level]:
                    position = item.Position
                    base_level = level - 1
                    item.Base.append(
                        Board[base_level][
                            position[0] * (self.Level - base_level) + position[1]
                        ]
                    )
                    item.Base.append(
                        Board[base_level][
                            (position[0] + 1) * (self.Level - base_level) + position[1]
                        ]
                    )
                    item.Base.append(
                        Board[base_level][
                            position[0] * (self.Level - base_level) + (position[1] + 1)
                        ]
                    )
                    item.Base.append(
                        Board[base_level][
                            (position[0] + 1) * (self.Level - base_level)
                            + (position[1] + 1)
                        ]
                    )

    def take_put_check(self, stop_mode=False):
        """Check which balls can be taken after a placement."""
        take_pos = []

        for level in range(self.Level):
            if level == 0:
                continue
            else:
                for item in self.Board[level]:
                    take_flag = 1  # All four base balls same color
                    put_flag = 1  # Can put ball on upper level
                    color = item.Base[0].Color
                    for base in item.Base:
                        if base.Available:
                            item.Legal = False
                            take_flag = 0
                            put_flag = 0
                            break
                        if base.Color != color:
                            take_flag = 0
                    if take_flag:
                        for base in item.Base:
                            if stop_mode:
                                take_pos.append(base)
                            else:
                                if base.Can_be_taken:
                                    if base not in take_pos:
                                        take_pos.append(base)
                        continue

                    if put_flag:
                        if item.Available:
                            item.Legal = True

        return take_pos

    def find_all_legal(self):
        """Find all legal positions to place a ball."""
        legal_pos = []
        for level in range(self.Level):
            for item in self.Board[level]:
                if item.Legal:
                    legal_pos.append(item)
        return legal_pos

    def all_pos(self):
        """Get all positions on the board."""
        pos = []
        for level in range(self.Level):
            for item in self.Board[level]:
                pos.append(item)
        return pos

    def all_avl_pos(self):
        """Get all available (empty) positions."""
        pos = []
        for level in range(self.Level):
            for item in self.Board[level]:
                if item.Available:
                    pos.append(item)
        return pos

    def count_balls(self):
        """Count balls at each level."""
        balls = []
        for _ in range(self.Level):
            balls.append(0)
        counter = 0
        for level in range(self.Level):
            for item in self.Board[level]:
                if not item.Available:
                    balls[level] += 1
                    counter += 1
        return balls, counter


class PyramidChessRandomGenerate:
    """Generate random pyramid chess game states."""

    def __init__(
        self,
        rand_turn_num: bool,
        plot_level: str = "Medium",
        num_turns: int | None = None,
        max_turn: int | None = None,
    ):
        if plot_level in PLOT_LEVEL:
            self.Level = PLOT_LEVEL[plot_level]
        else:
            raise ValueError("plot level not fit")
        self.Chess_Board = Board(level=self.Level)
        self.Turn = PLAYER_0
        self.Num_balls = NUM_BALLS[plot_level]
        self.Balls = [self.Num_balls, self.Num_balls]
        if plot_level == "Hard":
            self.Balls[self.Turn] += 1
        if max_turn is None:
            self.Max_turn = MAX_TURN_NUM[plot_level]
        else:
            self.Max_turn = max_turn
        self.Rand = rand_turn_num
        self.Win_status = -1

        if rand_turn_num:
            self.Num_turns = random.randint(0, self.Max_turn - 1)
        else:
            self.Num_turns = num_turns if num_turns is not None else 0

    def change_turn(self):
        """Switch to the other player."""
        if self.Turn == PLAYER_0:
            self.Turn = PLAYER_1
        else:
            self.Turn = PLAYER_0

    def take_random(self, take_pos):
        """Randomly take balls from take_pos."""
        if len(take_pos) <= 2:
            for item in take_pos:
                item.take()
            return len(take_pos), take_pos
        else:
            num_take = len(take_pos)
            index = random.sample(range(0, num_take), 2)
            input1 = index[0]
            input2 = index[1]
            info = [take_pos[input1], take_pos[input2]]
            take_pos[input1].take()
            take_pos[input2].take()
            return 2, info

    def one_turn_random(self):
        """Execute one random turn."""
        put_pos = self.Chess_Board.find_all_legal()
        if not put_pos:
            return None, [], 0

        index = random.randint(0, len(put_pos) - 1)
        put_pos[index].put(self.Turn)
        put_info = put_pos[index]
        self.Balls[self.Turn] -= 1

        num_take = 0
        take_info = []
        take_pos = self.Chess_Board.take_put_check()
        if take_pos:
            num_take, take_info = self.take_random(take_pos)
            self.Balls[self.Turn] += num_take

        return put_info, take_info, num_take

    def random_game(self):
        """Generate a random game state."""
        for _ in range(self.Num_turns):
            put_info, take_info, num_take = self.one_turn_random()
            if put_info is None:
                break
            self.change_turn()
        return self.Chess_Board


def count_base(grid: Grid, info: list, dict_info: dict):
    """Count how many balls need to be placed before this grid can have a ball."""
    if grid.Level == 0:
        return 0

    total_steps = 0
    for base in grid.Base:
        if base.Available:
            # This base position needs a ball
            if not base.counted:
                base.counted = True
                info[base.Level] += 1
                dict_info[base.Level].append(base.Position)
                total_steps += 1
                # Recursively count bases of this base
                total_steps += count_base(base, info, dict_info)

    return total_steps


def count_ball(board_dict: dict):
    """Count all balls on the board."""
    balls_list = []
    total_count = 0

    for _, grid in board_dict.items():
        level_details = []
        level_count = 0
        for row_idx, row in enumerate(grid):
            row_details = []
            for col_idx, cell in enumerate(row):
                if cell != "--":
                    color = "blue" if cell == "P0" else "red"
                    coord = [row_idx, col_idx]
                    row_details.append((color, coord))
                    level_count += 1
                    total_count += 1
            if row_details:
                level_details.append(row_details)
        balls_list.append((level_count, level_details))

    return balls_list, total_count


# ============================================================================
# Rendering Functions
# ============================================================================


def draw_pyramid_2d(layers: dict, plot_level: str) -> Image.Image:
    """Draw 2D representation of pyramid layers using matplotlib."""
    import io

    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_LEVEL_MAP = {"Easy": 3, "Medium": 4, "Hard": 5}
    num_levels = PLOT_LEVEL_MAP[plot_level]

    fig, axs = plt.subplots(1, num_levels, figsize=(25, 6))
    try:
        if num_levels == 1:
            axs = [axs]

        # Define color mapping
        color_map = {"P0": "blue", "P1": "red", "--": "none"}

        # Draw each layer
        for i, (layer_id, layer_data) in enumerate(layers.items()):
            if i >= num_levels:
                break
            ax = axs[i]
            ax.set_title(f"Level {layer_id}", fontsize=40)

            # Set axis range
            ax.set_xlim(-0.5, len(layer_data[0]) - 0.5)
            ax.set_ylim(-0.5, len(layer_data) - 0.5)

            # Set axis ticks
            ax.set_xticks(np.arange(len(layer_data[0])))
            ax.set_yticks(np.arange(len(layer_data)))
            ax.set_xticklabels([str(x) for x in range(len(layer_data[0]))], fontsize=40)
            ax.set_yticklabels([str(y) for y in range(len(layer_data))], fontsize=40)

            ax.invert_xaxis()
            ax.set_xlabel("x-axis", fontsize=30)
            ax.set_ylabel("y-axis", fontsize=30)
            ax.yaxis.set_ticks_position("right")
            ax.yaxis.set_label_position("right")

            # Draw circles
            for x, row in enumerate(layer_data):
                for y, cell in enumerate(row):
                    if cell != "--":
                        ax.scatter(
                            x, y, s=2000, c=color_map[cell], marker="o", edgecolor="black"
                        )

        # Adjust spacing
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2, wspace=0.3)

        buf = io.BytesIO()
        plt.savefig(buf, format="PNG")
    finally:
        plt.close(fig)
    buf.seek(0)

    return Image.open(buf)


def draw_pyramid_3d(layers: dict, plot_level: str) -> Image.Image:
    """Draw 3D representation of pyramid using matplotlib."""
    import io

    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_LEVEL_MAP = {"Easy": 3, "Medium": 4, "Hard": 5}
    num_levels = PLOT_LEVEL_MAP[plot_level]

    # Define colors for the balls
    color_map = {"P0": "blue", "P1": "red", "--": None}

    fig = plt.figure(figsize=(12, 12))
    try:
        ax = fig.add_subplot(111, projection="3d")

        ball_radius = 0.6  # Radius of the balls

        # Draw each layer in 3D
        for level, grid in layers.items():
            if level >= num_levels:
                break
            for i, row in enumerate(grid):
                for j, cell in enumerate(row):
                    if cell != "--":
                        # Calculate positions to align higher layers' balls on the center of lower layers
                        x = i + level * 0.5
                        y = j + level * 0.5
                        z = level

                        # Draw a sphere to represent the ball
                        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
                        sphere_x = ball_radius * np.cos(u) * np.sin(v) + x
                        sphere_y = ball_radius * np.sin(u) * np.sin(v) + y
                        sphere_z = ball_radius * np.cos(v) + z
                        ax.plot_surface(
                            sphere_x,
                            sphere_y,
                            sphere_z,
                            color=color_map[cell],
                            edgecolor="k",
                            linewidth=0.5,
                        )

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=15,
                label="PLAYER_0",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=15,
                label="PLAYER_1",
            ),
        ]
        ax.legend(
            handles=legend_elements, loc="upper left", prop={"size": 20}, markerscale=2.0
        )

        # Set axis limits (matching original: X-axis inverted, Y-axis normal)
        ax.set_xlim(num_levels, -1)  # X-axis: from large to small (inverted)
        ax.set_ylim(-1, num_levels)  # Y-axis: from small to large (normal)
        ax.set_zlim(-0.5, num_levels)
        ax.set_xlabel("X-axis", fontsize=15)
        ax.set_ylabel("Y-axis", fontsize=15)
        ax.set_zlabel("Z-axis", fontsize=15)
        ax.set_xticks(range(0, num_levels))
        ax.set_yticks(range(0, num_levels))
        ax.set_zticks(range(0, num_levels))
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.tick_params(axis="z", labelsize=15)

        buf = io.BytesIO()
        plt.savefig(buf, format="PNG")
    finally:
        plt.close(fig)
    buf.seek(0)

    return Image.open(buf)


def draw_pyramid_combined(layers: dict, plot_level: str) -> Image.Image:
    """Draw combined 2D and 3D representation of pyramid using matplotlib."""
    # Draw both representations
    image_3d = draw_pyramid_3d(layers, plot_level)
    image_2d = draw_pyramid_2d(layers, plot_level)

    # Resize 2D image to be shorter (like original implementation)
    image_2d = image_2d.resize((1200, 300))

    # Crop 3D image to remove excess whitespace (like original)
    crop_box = (0, 155, 1200, 1020)
    image_3d = image_3d.crop(crop_box)

    # Combine images vertically (2D on top, 3D on bottom)
    new_image = Image.new("RGB", (image_3d.width, image_3d.height + image_2d.height))
    new_image.paste(image_2d, (0, 0))
    new_image.paste(image_3d, (0, image_2d.height))

    # Resize to final size
    new_image = new_image.resize((550, 550))

    return new_image


# ============================================================================
# Pyramid Chess QA Environment
# ============================================================================


GAME_RULES = dedent("""
    Pyramid Chess Rules:
0. Game Board:
The game board is square. On an nxn board, there are n levels (0 to n-1). At each level k, the x and y coordinates range from 0 to n-1-k, resulting in (n-k)**2 slots per level. The slots in the lower levels act as the base for the slots in the upper levels. Slots at level 0 have no base, while slots at level j (j!=0) with coordinates (m,n) are supported by four base slots (m,n),(m+1,n),(m,n+1),(m+1,n+1) from level j-1. The top-left slot on each level is (0,0).

1. Players and Initial Setup:
The game is played between two players, designated as PLAYER_0 and PLAYER_1, each using balls of a distinct color from their color pool, blue balls for PLAYER_0 and red balls for PLAYER_1. Players take turns placing their balls on a square game board. The number of balls available to each player depends on the size of the board: on a 3x3 board, each player has 7 balls; on a 4x4 board, each has 15 balls; and on a 5x5 board, PLAYER_0 (the first player to place a ball) has 28 balls, while PLAYER_1 has 27 balls.

2. Placing Balls and Creating New Slots:
At the start of the game, the lowest level of the board (Level 0) is completely open and balls can be placed in any available slot on this level(since there is no base for slots in level 0, slots in level 0 have full base). After a ball is placed in a slot, that slot is no longer available for placing another ball. A ball can only be placed on the upper level if it is supported by a fully completed 2x2 block of balls on the level directly beneath, which means all the base of the slot is full(there is a ball in each of these slots).

3. Take-back mechanism:
If a player places a ball that completes a 2x2 block of the same color (all four balls belonging to that player), they may return up to two balls from the block to their color pool. A ball can only be removed if it does not have another ball directly above it, as removing a "base" ball would collapse the pyramid. Returning a ball reopens the slot it occupied, allowing it to be used for future placements, but the rule requiring a full 2x2 block as a base for placing balls on upper levels still applies.

4. Winning the Game:
The game ends when one player successfully places the last ball on top of the pyramid. The player who place the ball on the top of the pyramid wins.
""").strip()


class PyramidChessQAEnv(Env):
    # Meta: source=GameRL, category=puzzles, turn=single
    # Overrides: interaction_mode=single_turn, action_format=open_ended
    """Pyramid Chess QA environment.

    This environment provides 6 question types:
    0. What is the status of the ball at coordinate? (Easy, MCQ)
    1. Can a ball be placed and what outcome? (Medium, MCQ)
    2. How many steps to place ball at coordinate? (Hard, Fill)
    3. What is the best position to place a ball? (Hard, Fill)
    4. How many balls are on the board? (Easy, Fill)
    5. Higher level status of a coordinate (Medium, MCQ)
    """

    QUESTION_TYPES = [
        {
            "id": "status_at_coordinate",
            "name": "Status at Coordinate",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
            "question_id": 0,
        },
        {
            "id": "can_place_outcome",
            "name": "Can Place & Outcome",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
            "question_id": 1,
        },
        {
            "id": "steps_to_place",
            "name": "Steps to Place",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
            "question_id": 2,
        },
        {
            "id": "best_position",
            "name": "Best Position",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "Strategy Optimization",
            "question_id": 3,
        },
        {
            "id": "count_balls",
            "name": "Count Balls",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "Target Perception",
            "question_id": 4,
        },
        {
            "id": "coordinate_details",
            "name": "Coordinate Status Details",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
            "question_id": 5,
        },
    ]

    def __init__(
        self,
        plot_level: str = "Easy",
        question_type: int | None = None,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.plot_level = plot_level
        self._question_type_param = question_type
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}
        self._board = None
        self._game_gen = None

        # Standard QA variables
        self._question_type_idx: int = 0
        self._question: str = ""
        self._options: list[str] | None = None
        self._oracle_answer: str = ""

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        return build_description(
            game_name="Pyramid Chess",
            rules=GAME_RULES,
            question=self._question,
            options=self._options,
            oracle_answer=self._oracle_answer,
        )

    def _get_state_text(self) -> str:
        """Generate text description of current pyramid chess state."""
        board_dict = self._board.board_dict()
        balls_list, total_count = count_ball(board_dict)

        text = f"Pyramid Chess - {self.plot_level}\n"
        text += f"Board Level: {self._board.Level}\n"
        text += f"Total Balls: {total_count}\n"
        text += f"Current Turn: PLAYER_{self._game_gen.Turn}\n\n"

        text += "Board State by Level:\n"
        for level, (num_balls, details) in enumerate(balls_list):
            text += f"  Level {level}: {num_balls} ball(s)\n"
            for _row_idx, row in enumerate(details):
                for color, coord in row:
                    text += f"    {color.capitalize()} ball at position {coord}\n"

        return text.strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        # Generate random game state
        self._game_gen = PyramidChessRandomGenerate(
            rand_turn_num=True, plot_level=self.plot_level
        )
        self._board = self._game_gen.random_game()

        # Select question type
        if self._question_type_param is not None:
            self._question_type_idx = self._question_type_param
        else:
            self._question_type_idx = int(self.np_random.integers(0, 6))
        q_type = self.QUESTION_TYPES[self._question_type_idx]

        # Generate question - sets _question, _options, _oracle_answer
        if self._question_type_idx == 0:
            result = self._generate_status_question()
        elif self._question_type_idx == 1:
            result = self._generate_can_place_question()
        elif self._question_type_idx == 2:
            result = self._generate_steps_question()
        elif self._question_type_idx == 3:
            result = self._generate_best_position_question()
        elif self._question_type_idx == 4:
            result = self._generate_count_balls_question()
        elif self._question_type_idx == 5:
            result = self._generate_coordinate_details_question()

        # Extract to instance variables
        self._question = result["question"]
        self._options = result.get("options")
        self._oracle_answer = result["answer"]

        # Render board
        layers = self._board.board_dict()
        combined_image = draw_pyramid_combined(layers, self.plot_level)

        # Generate text state
        text_state = self._get_state_text()

        obs = Observation(
            image=combined_image,
            text=None,
            metadata={
                "state_text": text_state,
                "text_prompt": f"{self.description}",
                "question": self._question,
                "options": self._options,
                "question_type": q_type["name"],
                "level": q_type["level"],
            },
        )

        info = {
            "seed": seed,
            "oracle_answer": self._oracle_answer,
            "question_type": q_type["id"],
        }

        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

    def _score_answer(self, answer: str) -> float:
        """Score the user's answer.

        Args:
            answer: User's answer string

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        return score_exact(answer, str(self._oracle_answer))

    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]

        # Check if correct
        reward = self._score_answer(action_str)
        correct = reward == 1.0

        # Generate response
        if correct:
            response = "Correct!"
        else:
            response = f"Incorrect. The correct answer is: {self._oracle_answer}"

        # Re-render board
        layers = self._board.board_dict()
        combined_image = draw_pyramid_combined(layers, self.plot_level)

        obs = Observation(image=combined_image, text=response)

        terminated = True
        truncated = False
        info = {
            "oracle_answer": self._oracle_answer,
            "user_answer": action_str,
            "correct": correct,
        }

        return (
            {agent_id: obs for agent_id in self._agent_ids},
            {agent_id: reward for agent_id in self._agent_ids},
            {
                **{agent_id: terminated for agent_id in self._agent_ids},
                "__all__": terminated,
            },
            {
                **{agent_id: truncated for agent_id in self._agent_ids},
                "__all__": truncated,
            },
            {agent_id: info for agent_id in self._agent_ids},
        )

    def _generate_status_question(self) -> dict:
        """Type 0: What is the status of the ball at coordinate?"""
        position_table = self._board.all_pos()
        if not position_table:
            raise ValueError("Empty position table")

        answer_item = self.py_random.choice(position_table)
        level, position = answer_item.Level, answer_item.Position

        question = f"What is the status of the ball on Level {level}, which has coordinate {position}?"
        options = ["PLAYER_0", "PLAYER_1", "Empty", "Index out of bound"]

        length = len(self._board.Board)
        analysis = f"From the image provided, we can recognize that the board is a {length}x{length} board. "

        ball_color = answer_item.Color
        if ball_color is None:
            answer = 3
            analysis += f"The coordinate {position} in level {level} does not contain any balls (it is empty). Therefore, the status of the ball at coordinate {position} in level {level} is 3. Empty."
        elif ball_color == 0:
            answer = 1
            analysis += f"We can observe the layout of the pyramid across its levels. Based on level {level}'s grid (specifically at coordinate {position}), the ball is blue, which corresponds to PLAYER_0."
        elif ball_color == 1:
            answer = 2
            analysis += f"We can observe the layout of the pyramid across its levels. Based on level {level}'s grid (specifically at coordinate {position}), the ball is red, which corresponds to PLAYER_1."

        return {
            "question": question,
            "answer": str(answer),
            "options": options,
            "analysis": analysis,
        }

    def _generate_can_place_question(self) -> dict:
        """Type 1: Can a ball be placed and what outcome?"""
        position_table = self._board.all_pos()
        if not position_table:
            raise ValueError("Empty position table")

        answer_item = self.py_random.choice(position_table)
        level, position = answer_item.Level, answer_item.Position

        COLOR = ["blue", "red"]
        color_ind = self.py_random.choice([0, 1])
        color = COLOR[color_ind]

        question = f"Can a ball be placed at coordinate {position} on Level {level}? If a {color} ball is placed there, what would be the outcome?"
        options = [
            "Can place and no balls taken",
            "Can place and then balls can be taken",
            "Cannot place, position already occupied",
            "Cannot place, ball not ready below",
        ]

        length = len(self._board.Board)
        analysis = f"From the image provided, we can recognize that the board is a {length}x{length} board. "

        if answer_item.Legal:
            answer_item.put(color_ind)
            take_check_result = self._board.take_put_check(stop_mode=True)

            if not take_check_result:
                answer = 1
                analysis += f"Coordinate {position} on level {level} is empty and a ball can be placed there. Placing a {color} ball at coordinate {position} on level {level} does not form 2x2 block of the same color, and therefore no balls would be taken. Therefore, the status is: Can place and no balls taken."
            else:
                answer = 2
                take_level = take_check_result[0].Level
                take_position = [item.Position for item in take_check_result]
                analysis += f"Coordinate {position} on level {level} is empty and a ball can be placed there. Placing a {color} ball at coordinate {position} on level {level} forms a 2x2 block of the same color {color} at Level:{take_level}, Position:{take_position} and triggers a take-back mechanism. Therefore, the status is: Can place and 2x2 block formed, balls can be taken."
        else:
            if answer_item.Available is False:
                answer = 3
                current_color = COLOR[answer_item.Color]
                analysis += f"The coordinate {position} on level {level} is already occupied by a {current_color} ball, so it is not possible to place a ball there. Therefore, the status is: Cannot place, position already occupied."
            else:
                answer = 4
                bases = [base.Position for base in answer_item.Base]
                not_ready = [
                    base.Position for base in answer_item.Base if base.Available
                ]
                analysis += f"The coordinate {position} on level {level} cannot have a ball placed there. Because there are no platform which four balls below it form a 2x2 block to support it. To put a ball at coordinate {position} on level {level}, the bases of the position which are {bases} on level {level - 1} must be full. But there is no ball at {not_ready} on level {level - 1}. If a ball is placed at the position it will fall down. Therefore, the status is: Cannot place, ball not ready below."

        return {
            "question": question,
            "answer": str(answer),
            "options": options,
            "analysis": analysis,
        }

    def _generate_steps_question(self) -> dict:
        """Type 2: How many steps to place ball at coordinate?"""
        position_table = self._board.all_avl_pos()
        if not position_table:
            raise ValueError("Empty position table")

        answer_item = self.py_random.choice(position_table)
        level, position = answer_item.Level, answer_item.Position

        # Calculate steps needed
        info = [0, 0, 0, 0, 0]
        dict_info = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        steps_needed = count_base(answer_item, info, dict_info)
        steps_needed += 1

        question = f"How many steps (turns) are required for a ball to be placed at coordinate {position} on Level {level}? (including the turn placing the ball)"

        length = len(self._board.Board)
        analysis = f"From the image provided, we can recognize that the board is a {length}x{length} board. To place a ball at coordinate {position} on Level {level}, we need to ensure all the balls in its sub-pyramid, which are the balls supporting the position, are placed. This is determined by checking each level below the target position, from the highest level below it to the base level, and counting how many balls that support the position are missing in each layer. The total number of missing balls represents the steps needed.\n"

        full_flag = 1
        for i in range(level - 1, -1, -1):
            if info[i] != 0:
                full_flag = 0
                analysis += f"Level {i}: {info[i]} more ball(s) need to be placed at {dict_info[i]}.\n"

        if level == 0:
            analysis += "Since the ball is on level 0 the ground of the board, there is no ball need to be placed to support the ball, the ball at the target position can be placed immediately.\nTherefore, it needs 1 step in total."
        elif full_flag == 0:
            analysis += f"Once all the required balls in the sub-pyramid are placed, the ball at the target position can be placed.\nTherefore, it needs {steps_needed} steps in total."
        else:
            analysis += "All the required balls in the sub-pyramid has already been placed, the ball at the target position can be placed.\nTherefore, it needs 1 step in total."

        return {"question": question, "answer": str(steps_needed), "analysis": analysis}

    def _generate_best_position_question(self) -> dict:
        """Type 3: What is the best position to place a ball?"""
        # This requires finding a position that would form a 2x2 block
        # For simplicity, we'll find any legal position
        legal_positions = self._board.find_all_legal()
        if not legal_positions:
            # Regenerate
            self._game_gen = PyramidChessRandomGenerate(
                rand_turn_num=True, plot_level=self.plot_level
            )
            self._board = self._game_gen.random_game()
            return self._generate_best_position_question()

        # Try to find a position that would create a 2x2 block
        best_pos = None
        for pos in legal_positions:
            test_color = self._game_gen.Turn
            pos.put(test_color)
            take_result = self._board.take_put_check(stop_mode=True)
            pos.take()  # Undo

            if take_result:
                best_pos = pos
                break

        if best_pos is None:
            best_pos = self.py_random.choice(legal_positions)

        turn = self._game_gen.Turn
        PLAYER = ["PLAYER_0", "PLAYER_1"]
        COLOR = ["blue", "red"]

        question = f'It is {PLAYER[turn]}\'s turn (which uses the {COLOR[turn]} ball). What is the best coordinate to put a ball in order to maximize the opportunity of winning? Please answer in the form of "[x,y] at level z".'

        answer = f"{best_pos.Position} at level {best_pos.Level}"

        length = len(self._board.Board)
        analysis = f"From the image provided, we can recognize that the board is a {length}x{length} board. To maximize the winning chance, one must try to form a 2x2 block of their color for the take-back mechanism. So the answer is putting a ball at {best_pos.Position} at level {best_pos.Level}."

        return {"question": question, "answer": answer, "analysis": analysis}

    def _generate_count_balls_question(self) -> dict:
        """Type 4: How many balls are on the board?"""
        question = "How many balls are there on the board in the image?"

        board_dict = self._board.board_dict()
        balls_list, count = count_ball(board_dict)

        length = len(self._board.Board)
        analysis = f"From the image provided, we can recognize that the board is a {length}x{length} board. To count the total number of balls on the board, we start from the downmost level and proceed upward. For each level, we use the 2D representation of that level to count the balls row by row and column by column. Here is the detailed count:\n"

        for level, (num_balls, details) in enumerate(balls_list):
            analysis += f"Level {level} contains {num_balls} ball(s):\n"
            for row in details:
                for color, coord in row:
                    analysis += f"A {color} ball at {coord}.\n"

        analysis += f"\nFrom the image provided, the total number of balls on the board is {count}.\n"

        return {"question": question, "answer": str(count), "analysis": analysis}

    def _generate_coordinate_details_question(self) -> dict:
        """Type 5: Higher level status of a coordinate."""
        position_table = self._board.all_pos()
        if not position_table:
            raise ValueError("Empty position table")

        answer_item = self.py_random.choice(position_table)
        level, position = answer_item.Level, answer_item.Position

        # Determine status
        if answer_item.Legal:
            answer = 1
            status_text = "Legal to place ball"
        elif not answer_item.Available:
            answer = 2
            status_text = "Contains a ball"
        elif answer_item.Can_be_taken:
            answer = 3
            status_text = "Ball can be taken"
        else:
            answer = 4
            status_text = "Not ready (base incomplete)"

        question = f"What is the higher level status of coordinate {position} at Level {level}?"
        options = [
            "Legal to place ball",
            "Contains a ball",
            "Ball can be taken",
            "Not ready",
        ]

        length = len(self._board.Board)
        analysis = f"From the image provided, we can recognize that the board is a {length}x{length} board. Based on the coordinate {position} at level {level}, the status is: {status_text}."

        return {
            "question": question,
            "answer": str(answer),
            "options": options,
            "analysis": analysis,
        }

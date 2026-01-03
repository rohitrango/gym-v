"""Jewel2 QA Environment

A match-3 puzzle game QA environment with 7 question types.
"""

from __future__ import annotations

import copy
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()

# Game constants
COMMON_ELEMENTS = ["A", "B", "C", "D", "E"]
SPECIAL_ELEMENTS = ["a", "b", "c", "d", "e", "+", "|"]
DIRECTIONS = ["up", "down", "left", "right"]

# Element probabilities (from randomizer.py)
ELEMENT_PROBABILITIES = {
    "A": 0.15,
    "B": 0.15,
    "C": 0.15,
    "D": 0.15,
    "E": 0.15,
    "a": 0.04,
    "b": 0.04,
    "c": 0.04,
    "d": 0.04,
    "e": 0.04,
    "+": 0.03,
    "|": 0.02,
}


# ============================================================================
# Chessboard Game Logic
# ============================================================================


class Chessboard:
    """Jewel2 game board with match-3 mechanics"""

    def __init__(self, size=5):
        self.size = size
        self.normal_elements = COMMON_ELEMENTS
        self.special_elements = SPECIAL_ELEMENTS
        self.chessboard = self._generate_random_board()
        self.signboard = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.score = 0

    def _generate_random_board(self):
        """Generate random board based on probabilities"""
        board = []
        elements = list(ELEMENT_PROBABILITIES.keys())
        weights = list(ELEMENT_PROBABILITIES.values())

        for _ in range(self.size):
            row = []
            for _ in range(self.size):
                element = random.choices(elements, weights=weights, k=1)[0]
                row.append(element)
            board.append(row)
        return board

    def reset_signboard(self):
        """Reset the signboard marking"""
        for i in range(self.size):
            for j in range(self.size):
                self.signboard[i][j] = 0

    def clear_chess(self, x: int, y: int) -> int:
        """Clear element at (x,y) if valid, return number cleared"""
        self.reset_signboard()
        if not self._check_chess(x, y):
            return 0
        else:
            cleared = self._delete_chess()
            self._fill_chess()
            return cleared

    def _delete_chess(self) -> int:
        """Delete marked elements and return count"""
        cleared = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.signboard[i][j] == 1:
                    if (
                        self.chessboard[i][j] in self.normal_elements
                        or self.chessboard[i][j] in self.special_elements
                    ):
                        cleared += 1
                    self.chessboard[i][j] = " "
        self.score += cleared
        return cleared

    def _check_special_character(self, target: str, x: int, y: int):
        """Handle special element abilities"""
        if target in ["a", "b", "c", "d", "e"]:
            # Clear all corresponding uppercase letters
            uppercase_target = target.upper()
            for i in range(self.size):
                for j in range(self.size):
                    if self.chessboard[i][j] == uppercase_target:
                        self.signboard[i][j] = 1
            self.signboard[x][y] = 1
        elif target == "|":
            # Clear entire column
            for i in range(self.size):
                self.signboard[i][y] = 1
        elif target == "+":
            # Clear surrounding 3x3 area
            for i in range(x - 1, x + 2):
                for j in range(y - 1, y + 2):
                    if 0 <= i < self.size and 0 <= j < self.size:
                        self.signboard[i][j] = 1

    def _mark_elements_for_line(self, x: int, y: int) -> bool:
        """Check and mark elements forming 3+ in a line"""
        marked = False

        # Check horizontal
        horizontal = self._get_horizontal_line(x, y)
        if len(horizontal) >= 3:
            for i, j in horizontal:
                self.signboard[i][j] = 1
            marked = True

        # Check vertical
        vertical = self._get_vertical_line(x, y)
        if len(vertical) >= 3:
            for i, j in vertical:
                self.signboard[i][j] = 1
            marked = True

        return marked

    def _check_chess(self, x: int, y: int) -> bool:
        """Check if element at (x,y) can be cleared"""
        target = self.chessboard[x][y]
        if target == " ":
            return False

        if target in self.special_elements:
            self._check_special_character(target, x, y)
            return any(
                self.signboard[i][j] == 1
                for i in range(self.size)
                for j in range(self.size)
            )
        elif target in self.normal_elements:
            return self._mark_elements_for_line(x, y)
        else:
            return False

    def _get_horizontal_line(self, x: int, y: int) -> list[tuple[int, int]]:
        """Get horizontal line of matching elements"""
        target = self.chessboard[x][y]
        line = [(x, y)]

        # Check left
        j = y - 1
        while j >= 0 and self.chessboard[x][j] == target:
            line.append((x, j))
            j -= 1

        # Check right
        j = y + 1
        while j < self.size and self.chessboard[x][j] == target:
            line.append((x, j))
            j += 1

        return line

    def _get_vertical_line(self, x: int, y: int) -> list[tuple[int, int]]:
        """Get vertical line of matching elements"""
        target = self.chessboard[x][y]
        line = [(x, y)]

        # Check up
        i = x - 1
        while i >= 0 and self.chessboard[i][y] == target:
            line.append((i, y))
            i -= 1

        # Check down
        i = x + 1
        while i < self.size and self.chessboard[i][y] == target:
            line.append((i, y))
            i += 1

        return line

    def _fill_chess(self):
        """Fill empty spaces with new elements (gravity + new generation)"""
        for j in range(self.size):
            for i in range(self.size - 1, -1, -1):
                if self.chessboard[i][j] == " ":
                    # Find first non-empty cell above
                    k = i - 1
                    while k >= 0 and self.chessboard[k][j] == " ":
                        k -= 1
                    if k >= 0:
                        self.chessboard[i][j] = self.chessboard[k][j]
                        self.chessboard[k][j] = " "
                    else:
                        # Generate new element
                        elements = list(ELEMENT_PROBABILITIES.keys())
                        weights = list(ELEMENT_PROBABILITIES.values())
                        self.chessboard[i][j] = random.choices(
                            elements, weights=weights, k=1
                        )[0]

    def swap_chess(self, x: int, y: int, pos: str) -> bool:
        """Swap element at (x,y) with adjacent element in direction pos"""
        direction_map = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        if pos not in direction_map:
            return False

        dx, dy = direction_map[pos]
        nx, ny = x + dx, y + dy

        if not (0 <= nx < self.size and 0 <= ny < self.size):
            return False

        # Check for special elements
        elem1 = self.chessboard[x][y]
        elem2 = self.chessboard[nx][ny]
        if elem1 in self.special_elements or elem2 in self.special_elements:
            return False

        # Swap elements
        self.chessboard[x][y], self.chessboard[nx][ny] = (
            self.chessboard[nx][ny],
            self.chessboard[x][y],
        )

        # Check if elimination conditions are formed
        self.reset_signboard()
        check1 = self._check_chess(x, y)
        check2 = self._check_chess(nx, ny)

        if check1 or check2:
            self._delete_chess()
            self._fill_chess()
            self.reset_signboard()
            return True
        else:
            # Swap back
            self.chessboard[x][y], self.chessboard[nx][ny] = (
                self.chessboard[nx][ny],
                self.chessboard[x][y],
            )
            return False


# ============================================================================
# Jewel2 QA Environment
# ============================================================================

JEWEL2_RULES = """
# **Game Overview**
Jewel2 is a strategic puzzle game played on a grid. Your primary objective is to eliminate elements by forming horizontal or vertical lines of three or more identical items. Successfully eliminating elements increases your score and clears space on the board for new elements to appear.

# **Elements**
## **Basic Elements**
- **A, B, C, D, E**
  - **Description**: These are the standard elements in the game.
  - **Shape**: Diamond-shaped gems in various colors (Red, Green, Blue, Yellow, Purple).
    - A: Red
    - B: Green
    - C: Blue
    - D: Yellow
    - E: Purple
  - **Interactions**:
    - **Clearing**: When three or more identical basic elements align horizontally or vertically, they are eliminated from the board.
    - **Swapping**: Basic elements can be swapped with adjacent basic elements to form eliminations.

## **Special Elements**
- **a, b, c, d, e, +, |**
  - **Description**: These elements possess unique abilities that trigger specific elimination patterns when activated.
  - **Shape**:
    - **a, b, c, d, e**: Round gems in various colors (Red, Green, Blue, Yellow, Purple).
        - a: Red
        - b: Green
        - c: Blue
        - d: Yellow
        - e: Purple
    - **+**: A round black gem with low transparency.
    - **|**: A tall, rectangular cyan gem.
  - **Effects of Special Elements**:
    - **a, b, c, d, e**:
      - **Function**: Clearing one of these removes all corresponding uppercase basic elements from the board.
        - *Example*: Clearing element 'a' will eliminate all 'A's on the board.
    - **| (Vertical Clear)**:
      - **Function**: Activating this element clears all elements in its vertical column.
    - **+ (Surrounding Clear)**:
      - **Function**: Activating this element clears all elements within a distance of 1 from its position, including diagonals.

  - **Notes**:
    - Special elements do **not** trigger further eliminations if they remove other special elements.
    - Swapping involving special elements is **not allowed** and will be rejected by the game.

# **Commands**
## **Available Operations**
1. **Clear Operation**
   - **Syntax**: clear x y
   - **Description**: Attempts to clear the element located at coordinates (x, y).
   - **Conditions**:
     - The targeted element must form a valid elimination (i.e., be part of a horizontal or vertical line of three or more identical elements).
     - If the element is special, its unique ability is activated upon clearing.
   - **State Changes**:
     - **Basic Element**: If the clearance is valid, the element(s) are removed, the score (Total Cleared) increases accordingly, and new elements fall into place to fill the gaps.
     - **Special Element**: Activating a special element triggers its specific clearance effect as described above.

2. **Swap Operation**
   - **Syntax**: swap x y pos
   - **Parameters**:
     - (x, y): Coordinates of the element to be swapped.
     - pos: Direction to swap the element (up, down, left, right).
   - **Description**: Swaps the element at (x, y) with the adjacent element in the specified direction.
     - **pos** can be one of four directions:
       - **up**: Swap with the element directly above (in the same column but one row above).
       - **down**: Swap with the element directly below (in the same column but one row below).
       - **left**: Swap with the element directly to the left (in the same row but one column left).
       - **right**: Swap with the element directly to the right (in the same row but one column right).
   - **Conditions**:
     - Both elements involved in the swap must be basic elements. Swaps involving special elements are rejected.
     - The swap must result in a valid elimination; otherwise, the swap is undone.
   - **State Changes**:
     - **Successful Swap**: Elements are exchanged, any resulting eliminations are performed, and the score (Total Cleared) is updated accordingly.
     - **Unsuccessful Swap**: Elements revert to their original positions, and no changes are made to the score.

# **Coordinate System**
- The board uses **0-based coordinates**.
- **Top-left cell**: (0, 0)
- **Bottom-right cell**: (size-1, size-1)

### **Coordinate Explanation**:
  - **x (Row)**: Represents the **row number** of the element. Rows are numbered from **top to bottom**, starting from 0.
  - **y (Column)**: Represents the **column number** of the element. Columns are numbered from **left to right**, starting from 0.

# **Gameplay Mechanics**
## **Score Tracking**
- **Total Cleared**: Represents the cumulative number of elements that have been eliminated throughout the game.
  - **Incremented By**: The number of elements cleared in each successful operation (clear or swap).

# **Objective**
Maximize your **Total Cleared** count by strategically performing clear and swap operations to eliminate as many elements as possible. Effective use of special elements can significantly enhance your score by triggering large-scale eliminations.
"""


class GameRLJewel2QAEnv(Env):
    """Jewel2 QA Environment with 7 question types"""

    QUESTION_TYPES = [
        {
            "id": "count_element",
            "name": "Count Specific Element",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "StateInfo",
        },
        {
            "id": "identify_position",
            "name": "Identify Element Position",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "StateInfo",
        },
        {
            "id": "count_special",
            "name": "Count Special Elements",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "StateInfo",
        },
        {
            "id": "simulate_clear",
            "name": "Simulate Clear Operation",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "ActionOutcome",
        },
        {
            "id": "simulate_swap",
            "name": "Simulate Swap Operation",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "ActionOutcome",
        },
        {
            "id": "command_sequence",
            "name": "Simulate Command Sequence",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "ActionOutcome",
        },
        {
            "id": "optimal_move",
            "name": "Find Optimal Move",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "StrategyOptimization",
        },
    ]

    def __init__(self, size=5, question_type: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self._question_type = question_type
        self._chessboard = None
        self._total_cleared = 0
        self._current_question = None
        self._element_images = None

    @property
    def description(self) -> str:
        return f"Jewel2 QA (size={self.size})\n\n" + JEWEL2_RULES

    def _initialize_game(self):
        """Initialize game state"""
        self._chessboard = Chessboard(self.size)
        self._total_cleared = random.randint(0, 100)

    def _load_element_images(self):
        """Load element images"""
        if self._element_images is not None:
            return

        self._element_images = {}
        images_dir = Path(
            "/mnt/petrelfs/gujiawei/jiawei/env-v/Game-RL/src/jewel2/images"
        )

        element_map = {
            "A": "A.png",
            "B": "B.png",
            "C": "C.png",
            "D": "D.png",
            "E": "E.png",
            "a": "a_s.png",
            "b": "b_s.png",
            "c": "c_s.png",
            "d": "d_s.png",
            "e": "e_s.png",
            "+": "cross.png",
            "|": "bar.png",
            " ": "empty.png",
        }

        for elem, filename in element_map.items():
            img_path = images_dir / filename
            if img_path.exists():
                self._element_images[elem] = Image.open(img_path).convert("RGBA")

    def render(self) -> Image.Image:
        """Render the game board"""
        image_width = 480
        image_height = 640

        # Margins
        left_margin = 60
        top_margin = 150
        right_margin = 20
        bottom_margin = 50

        # Calculate cell size
        available_width = image_width - left_margin - right_margin
        available_height = image_height - top_margin - bottom_margin
        cell_size = min(available_width // self.size, available_height // self.size)
        board_x = left_margin
        board_y = top_margin

        # Create image
        img = Image.new("RGB", (image_width, image_height), "white")
        draw = ImageDraw.Draw(img)

        # Load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 55)
            number_font = ImageFont.truetype("arial.ttf", 20)
            cleared_font = ImageFont.truetype("arial.ttf", 30)
        except OSError:
            title_font = ImageFont.load_default()
            number_font = ImageFont.load_default()
            cleared_font = ImageFont.load_default()

        # Draw title
        title = "Jewel2 Game"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (image_width - title_width) // 2
        draw.text((title_x, 10), title, fill="black", font=title_font)

        # Draw column numbers
        for c in range(self.size):
            col_num = str(c)
            col_bbox = draw.textbbox((0, 0), col_num, font=number_font)
            col_width = col_bbox[2] - col_bbox[0]
            col_x = board_x + c * cell_size + cell_size // 2 - col_width // 2
            col_y = board_y - 30
            draw.text((col_x, col_y), col_num, fill="black", font=number_font)

            # Tick mark
            tick_x = board_x + c * cell_size + cell_size // 2
            draw.line([(tick_x, board_y - 5), (tick_x, board_y)], fill="black", width=1)

        # Draw row numbers
        for r in range(self.size):
            row_num = str(r)
            row_bbox = draw.textbbox((0, 0), row_num, font=number_font)
            row_width = row_bbox[2] - row_bbox[0]
            row_x = board_x - 30 - row_width // 2
            row_y = board_y + r * cell_size + cell_size // 2 - 10
            draw.text((row_x, row_y), row_num, fill="black", font=number_font)

            # Tick mark
            tick_y = board_y + r * cell_size + cell_size // 2
            draw.line([(board_x - 5, tick_y), (board_x, tick_y)], fill="black", width=1)

        # Draw chessboard elements
        for r in range(self.size):
            for c in range(self.size):
                x0 = board_x + c * cell_size
                y0 = board_y + r * cell_size
                element = self._chessboard.chessboard[r][c]

                if element in self._element_images:
                    element_img = self._element_images[element].resize(
                        (cell_size, cell_size), Image.Resampling.LANCZOS
                    )
                    img.paste(element_img, (x0, y0), element_img)

        # Draw total cleared
        cleared_text = f"Total Cleared: {self._total_cleared}"
        cleared_bbox = draw.textbbox((0, 0), cleared_text, font=cleared_font)
        cleared_width = cleared_bbox[2] - cleared_bbox[0]
        cleared_x = (image_width - cleared_width) // 2
        cleared_y = image_height - 100
        draw.text((cleared_x, cleared_y), cleared_text, fill="black", font=cleared_font)

        return img

    # ========================================================================
    # Question Generation
    # ========================================================================

    def _find_valid_clear_position(self):
        """Find a position where clear can be executed"""
        for x in range(self.size):
            for y in range(self.size):
                temp_board = copy.deepcopy(self._chessboard)
                if temp_board.clear_chess(x, y) > 0:
                    return (x, y)
        return None

    def _find_valid_swap_position(self):
        """Find a position where swap can be executed"""
        for x in range(self.size):
            for y in range(self.size):
                for pos in DIRECTIONS:
                    temp_board = copy.deepcopy(self._chessboard)
                    if temp_board.swap_chess(x, y, pos):
                        return (x, y, pos)
        return None

    def _generate_question_type_0(self) -> dict:
        """Count specific element"""
        element = random.choice(COMMON_ELEMENTS)
        positions = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self._chessboard.chessboard[r][c] == element
        ]
        count = len(positions)

        question = f"{JEWEL2_RULES}\n\n**Question:** How many '{element}' elements are currently on the board?"
        answer = str(count)
        analysis = (
            f"By iterating through each row and column of the chessboard, we identified and counted all occurrences of the '{element}' element. "
            f"The '{element}' elements are located at the following positions: "
            f"{', '.join([f'({r},{c})' for r, c in positions]) if positions else 'No positions found'}. "
            f"So there are **{count}** '{element}' in total."
        )

        return {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "options": None,
        }

    def _generate_question_type_1(self) -> dict:
        """Identify element position (MCQ)"""
        while True:
            element = random.choice(COMMON_ELEMENTS)
            positions = [
                (r, c)
                for r in range(self.size)
                for c in range(self.size)
                if self._chessboard.chessboard[r][c] == element
            ]
            if positions:
                break

        all_positions = [(r, c) for r in range(self.size) for c in range(self.size)]
        incorrect_candidates = [pos for pos in all_positions if pos not in positions]

        correct_position = random.choice(positions)
        options_list = [f"Position {correct_position}"] + [
            f"Position {pos}"
            for pos in random.sample(
                incorrect_candidates, min(7, len(incorrect_candidates))
            )
        ]
        random.shuffle(options_list)

        options = [f"{chr(65 + idx)}. {opt}" for idx, opt in enumerate(options_list)]
        correct_answer_letter = next(
            (opt[0] for opt in options if f"Position {correct_position}" in opt), "A"
        )
        answer = correct_answer_letter

        question = f"{JEWEL2_RULES}\n\n**Question:** Which of the following positions does element '{element}' reside in?"
        question += "\n\n**Options:**\n" + "\n".join(options)

        analysis = (
            f"The '{element}' element is located at the following positions: {', '.join([f'({r},{c})' for r, c in positions])}. "
            f"Option {correct_answer_letter} refers to position {correct_position}, where the '{element}' element resides."
        )

        return {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "options": options,
        }

    def _generate_question_type_2(self) -> dict:
        """Count special elements"""
        special_positions = {elem: [] for elem in SPECIAL_ELEMENTS}
        for r in range(self.size):
            for c in range(self.size):
                if self._chessboard.chessboard[r][c] in SPECIAL_ELEMENTS:
                    special_positions[self._chessboard.chessboard[r][c]].append((r, c))

        count = sum(len(positions) for positions in special_positions.values())

        question = f"{JEWEL2_RULES}\n\n**Question:** How many special elements (a, b, c, d, e, +, |) are there on the board?"
        answer = str(count)
        analysis = (
            "By iterating through the chessboard, we counted all special elements (a, b, c, d, e, +, |).\n\n"
            "Positions of special elements:\n"
            + "\n".join(
                [
                    f"- Element '{elem}': "
                    + (
                        ", ".join([f"({r},{c})" for r, c in positions])
                        if positions
                        else "None found"
                    )
                    for elem, positions in special_positions.items()
                ]
            )
            + f"\n\nSo there are **{count}** special elements in total."
        )

        return {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "options": None,
        }

    def _generate_question_type_3(self) -> dict:
        """Simulate clear operation (MCQ)"""
        if random.random() < 0.75:
            valid_pos = self._find_valid_clear_position()
            if valid_pos:
                x, y = valid_pos
            else:
                x, y = (
                    random.randint(0, self.size - 1),
                    random.randint(0, self.size - 1),
                )
        else:
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)

        target = self._chessboard.chessboard[x][y]
        temp_board = copy.deepcopy(self._chessboard)
        current_cleared = self._total_cleared
        cleared = temp_board.clear_chess(x, y)
        new_total_cleared = current_cleared + cleared

        question = f"{JEWEL2_RULES}\n\n**Question:** What will happen if you execute clear {x} {y}?"

        if cleared == 0:
            options = [
                "A. Nothing will happen because the clear does not meet elimination conditions."
            ] + [
                f"{chr(66+i)}. Perform elimination, eliminate {random.randint(1, 5)} elements, total cleared becomes {current_cleared + random.randint(1, 5)}."
                for i in range(7)
            ]
            answer = "A"
            analysis = f"Attempting to clear the element at position ({x},{y}) did not meet the elimination conditions, so no elements were cleared."
        else:
            if target in SPECIAL_ELEMENTS:
                options = [
                    "A. Nothing will happen because the clear does not meet elimination conditions.",
                    f"B. Trigger a special element, total cleared becomes {new_total_cleared}.",
                ] + [
                    f"{chr(67+i)}. Perform elimination, eliminate {random.randint(1, 5)} elements, total cleared becomes {current_cleared + random.randint(1, 5)}."
                    for i in range(6)
                ]
                answer = "B"
                analysis = f"Clearing the special element '{target}' at position ({x},{y}) triggered its ability, resulting in the elimination of additional elements. The total cleared count increased to {new_total_cleared}."
            else:
                correct_option = random.choice(["C", "D", "E", "F", "G", "H"])
                options = [
                    "A. Nothing will happen because the clear does not meet elimination conditions.",
                    f"B. Trigger a special element, total cleared becomes {new_total_cleared}.",
                ]

                for option in ["C", "D", "E", "F", "G", "H"]:
                    random_value = random.randint(1, 9)
                    while random_value == cleared:
                        random_value = random.randint(1, 9)
                    options.append(
                        f"{option}. Perform elimination, eliminate {random_value} elements, total cleared becomes {current_cleared + random_value}."
                    )

                correct_idx = ord(correct_option) - ord("C") + 2
                options[correct_idx] = (
                    f"{correct_option}. Perform elimination, eliminate {cleared} elements, total cleared becomes {new_total_cleared}."
                )
                answer = correct_option
                analysis = f"Clearing the element '{target}' at position ({x},{y}) successfully eliminated {cleared} elements vertically/horizontally, increasing the total cleared count to {new_total_cleared}."

        question += "\n\n**Options:**\n" + "\n".join(options)
        return {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "options": options,
        }

    def _generate_question_type_4(self) -> dict:
        """Simulate swap operation (MCQ)"""
        if random.random() < 0.75:
            valid_move = self._find_valid_swap_position()
            if valid_move:
                x, y, pos = valid_move
            else:
                x, y = (
                    random.randint(0, self.size - 1),
                    random.randint(0, self.size - 1),
                )
                pos = random.choice(DIRECTIONS)
        else:
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            pos = random.choice(DIRECTIONS)

        question = f"{JEWEL2_RULES}\n\n**Question:** What will happen if you execute swap {x} {y} {pos}?"

        direction_map = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        dx, dy = direction_map[pos]
        nx, ny = x + dx, y + dy

        elem1 = self._chessboard.chessboard[x][y]
        elem2 = (
            self._chessboard.chessboard[nx][ny]
            if (0 <= nx < self.size and 0 <= ny < self.size)
            else None
        )

        if not (0 <= nx < self.size and 0 <= ny < self.size):
            options = (
                [
                    "A. Nothing will happen because the swap does not meet elimination conditions."
                ]
                + ["B. Cannot perform swap because one of the elements is special."]
                + [
                    f"{chr(67+i)}. After swap, elimination occurs, clearing {random.randint(1, 5)} elements, total cleared becomes {self._total_cleared + random.randint(1, 5)}."
                    for i in range(6)
                ]
            )
            answer = "A"
            analysis = f"Attempting to swap the element at position ({x},{y}) in the '{pos}' direction goes out of the board's boundaries. Therefore, the swap action cannot be performed."
        elif elem1 in SPECIAL_ELEMENTS or elem2 in SPECIAL_ELEMENTS:
            options = [
                "A. Nothing will happen because the swap does not meet elimination conditions.",
                "B. Cannot perform swap because one of the elements is special.",
            ] + [
                f"{chr(67+i)}. After swap, elimination occurs, clearing {random.randint(1, 5)} elements, total cleared becomes {self._total_cleared + random.randint(1, 5)}."
                for i in range(6)
            ]
            answer = "B"
            analysis = f"Swapping involves special elements '{elem1}' or '{elem2}', which is not allowed. The swap action is rejected."
        else:
            temp_board = copy.deepcopy(self._chessboard)
            success = temp_board.swap_chess(x, y, pos)
            cleared_after_swap = temp_board.score - self._chessboard.score

            if success:
                correct_option = random.choice(["C", "D", "E", "F", "G", "H"])
                options = [
                    "A. Nothing will happen because the swap does not meet elimination conditions.",
                    "B. Cannot perform swap because one of the elements is special.",
                ]

                for option in ["C", "D", "E", "F", "G", "H"]:
                    random_value = random.randint(1, 9)
                    while random_value == cleared_after_swap:
                        random_value = random.randint(1, 9)
                    options.append(
                        f"{option}. After swap, elimination occurs, clearing {random_value} elements, total cleared becomes {self._total_cleared + random_value}."
                    )

                correct_idx = ord(correct_option) - ord("C") + 2
                options[correct_idx] = (
                    f"{correct_option}. After swap, elimination occurs, clearing {cleared_after_swap} elements, total cleared becomes {self._total_cleared + cleared_after_swap}."
                )
                answer = correct_option
                analysis = f"Successfully swapped the elements at position ({x},{y}) with ({nx},{ny}) in the '{pos}' direction, resulting in the vertical/horizontal elimination of {cleared_after_swap} elements. The total cleared count increased to {self._total_cleared + cleared_after_swap}."
            else:
                options = (
                    [
                        "A. Nothing will happen because the swap does not meet elimination conditions."
                    ]
                    + ["B. Cannot perform swap because one of the elements is special."]
                    + [
                        f"{chr(67+i)}. After swap, elimination occurs, clearing {random.randint(1, 5)} elements, total cleared becomes {self._total_cleared + random.randint(1, 5)}."
                        for i in range(6)
                    ]
                )
                answer = "A"
                analysis = f"Swapping the elements at position ({x},{y}) with ({nx},{ny}) in the '{pos}' direction did not create any valid elimination conditions. No elements were cleared."

        question += "\n\n**Options:**\n" + "\n".join(options)
        return {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "options": options,
        }

    def _generate_question_type_5(self) -> dict:
        """Simulate command sequence"""
        possible_commands = []
        for r in range(self.size):
            for c in range(self.size):
                possible_commands.append(f"clear {r} {c}")
                for d in DIRECTIONS:
                    possible_commands.append(f"swap {r} {c} {d}")

        command1 = random.choice(possible_commands)
        command2 = random.choice(possible_commands)
        while command2 == command1:
            command2 = random.choice(possible_commands)

        temp_board = copy.deepcopy(self._chessboard)
        total_cleared = 0
        cleared1, cleared2 = 0, 0

        # Execute first command
        if command1.startswith("clear"):
            _, x1, y1 = command1.split()
            cleared1 = temp_board.clear_chess(int(x1), int(y1))
            total_cleared += cleared1
        elif command1.startswith("swap"):
            _, x1, y1, pos1 = command1.split()
            score_before = temp_board.score
            if temp_board.swap_chess(int(x1), int(y1), pos1):
                cleared1 = temp_board.score - score_before
                total_cleared += cleared1

        # Execute second command
        if command2.startswith("clear"):
            _, x2, y2 = command2.split()
            cleared2 = temp_board.clear_chess(int(x2), int(y2))
            total_cleared += cleared2
        elif command2.startswith("swap"):
            _, x2, y2, pos2 = command2.split()
            score_before = temp_board.score
            if temp_board.swap_chess(int(x2), int(y2), pos2):
                cleared2 = temp_board.score - score_before
                total_cleared += cleared2

        question = f"{JEWEL2_RULES}\n\n**Question:** How many elements will be eliminated at least after performing {command1} followed by {command2}?"
        answer = str(total_cleared)
        analysis = f"Executing `{command1}` resulted in vertically/horizontally clearing {cleared1} elements, and executing `{command2}` resulted in vertically/horizontally clearing {cleared2} elements. Overall, a total of {total_cleared} elements were cleared."

        return {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "options": None,
        }

    def _generate_question_type_6(self) -> dict:
        """Find optimal move"""
        max_cleared = 0
        best_command = "No command can clear any elements."
        valid_moves = []

        # Evaluate all swap commands
        for r in range(self.size):
            for c in range(self.size):
                for d in DIRECTIONS:
                    temp_board = copy.deepcopy(self._chessboard)
                    score_before = temp_board.score
                    if temp_board.swap_chess(r, c, d):
                        cleared = temp_board.score - score_before
                        valid_moves.append(
                            {"command": f"swap {r} {c} {d}", "cleared": cleared}
                        )
                        if cleared > max_cleared:
                            max_cleared = cleared
                            best_command = f"swap {r} {c} {d}"

        # Evaluate all clear commands
        for r in range(self.size):
            for c in range(self.size):
                temp_board = copy.deepcopy(self._chessboard)
                cleared = temp_board.clear_chess(r, c)
                if cleared > 0:
                    valid_moves.append(
                        {"command": f"clear {r} {c}", "cleared": cleared}
                    )
                    if cleared > max_cleared:
                        max_cleared = cleared
                        best_command = f"clear {r} {c}"

        valid_moves.sort(key=lambda x: x["cleared"], reverse=True)

        question = f"{JEWEL2_RULES}\n\n**Question:** What command will result in the maximum number of elements being cleared in a single move?"
        answer = best_command

        if max_cleared > 0:
            analysis = (
                "Analysis of all possible clearing moves:\n\n"
                + "\n".join(
                    [
                        f"- Command `{move['command']}` will vertically/horizontally clear {move['cleared']} elements"
                        for move in valid_moves[:10]
                    ]
                )  # Show top 10
                + "\n\nBest strategy analysis:\n"
                + f"The optimal move is `{best_command}`, which will vertically/horizontally clear the maximum number of {max_cleared} elements."
            )
        else:
            analysis = "In the current board state, no command can clear any elements."

        return {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "options": None,
        }

    def _generate_question(self, question_type: int) -> dict:
        """Generate question of specified type"""
        if question_type == 0:
            return self._generate_question_type_0()
        elif question_type == 1:
            return self._generate_question_type_1()
        elif question_type == 2:
            return self._generate_question_type_2()
        elif question_type == 3:
            return self._generate_question_type_3()
        elif question_type == 4:
            return self._generate_question_type_4()
        elif question_type == 5:
            return self._generate_question_type_5()
        elif question_type == 6:
            return self._generate_question_type_6()
        else:
            raise ValueError(f"Invalid question type: {question_type}")

    # ========================================================================
    # Gym-v Interface
    # ========================================================================

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment"""
        super().reset(seed=seed)

        logger.info(f"Reset Jewel2 QA (size={self.size})")

        self._initialize_game()
        self._load_element_images()

        # Generate question
        q_type = (
            self._question_type
            if self._question_type is not None
            else random.randint(0, 6)
        )
        self._current_question = self._generate_question(q_type)

        obs = Observation(image=self.render(), text=self._current_question["question"])
        return obs, {}

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Execute an action (answer the question)"""
        correct = (
            action.strip().lower() == self._current_question["answer"].strip().lower()
        )
        reward = 1.0 if correct else 0.0

        if correct:
            response = "Correct!"
        else:
            response = f"Incorrect. The correct answer is: {self._current_question['answer']}\n\n{self._current_question['analysis']}"

        obs = Observation(image=self.render(), text=response)
        return obs, reward, True, False, {}

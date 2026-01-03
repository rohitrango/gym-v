"""FreeCell QA environment based on GameRL."""

from __future__ import annotations

from enum import Enum
from importlib import resources
import random
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class Suit(Enum):
    """Card suits."""

    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"


class Color(Enum):
    """Card colors."""

    RED = "red"
    BLACK = "black"


class Card:
    """Represents a playing card."""

    def __init__(self, suit: Suit, value: int):
        self.suit = suit
        self.value = value

    @property
    def color(self) -> Color:
        return Color.RED if self.suit in [Suit.HEARTS, Suit.DIAMONDS] else Color.BLACK

    def __str__(self) -> str:
        value_map = {1: "A", 11: "J", 12: "Q", 13: "K"}
        value_str = value_map.get(self.value, str(self.value))
        return f"{value_str} {self.suit.value}"


class GameRLFreecellQAEnv(Env):
    """FreeCell QA environment.

    A solitaire card game with cascade piles, free cells, and foundation piles.

    Question Types:
    - Specified Card: Identify card at a specific position in a cascade pile
    - Valid Move: Identify which move is valid
    - Card After Move: Identify top card after a move

    Args:
        cascade_number: Number of cascade piles (4, 6, or 8)
        question_type: Type of question to ask
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    QUESTION_TYPES = [
        {
            "id": "specified_card",
            "name": "Specified Card",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "valid_move",
            "name": "Valid Move",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "card_after_move",
            "name": "Card After Move",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
    ]

    SUIT_TO_STRING = {"♥": "Heart", "♦": "Diamond", "♣": "Club", "♠": "Spade"}

    STRING_TO_SUIT = {v: k for k, v in SUIT_TO_STRING.items()}

    VALUE_MAP = {
        1: "A",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "10",
        11: "J",
        12: "Q",
        13: "K",
    }

    FREECELL_RULES = dedent("""
        In FreeCell, cards can be moved according to specific rules:
        - Cards in cascade piles must be stacked in descending order with alternating colors
        - Foundation piles must be built up by suit from Ace to King
        - Free cells can hold only one card each
        - A card can be moved to a free cell if available, stacked in descending order alternating colors in cascade piles, or placed in foundation piles starting from Ace
    """).strip()

    def __init__(
        self,
        cascade_number: int | None = None,
        question_type: str | int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if cascade_number is None:
            cascade_number = random.choice([4, 6, 8])

        self._cascade_number = cascade_number
        self._question_type = question_type

        # Game state
        self._cascade_piles: list[list[Card]] = [[] for _ in range(cascade_number)]
        self._free_cells: list[Card | None] = [None] * 4
        self._foundation_piles: dict[Suit, list[Card]] = {suit: [] for suit in Suit}
        self._current_question: dict[str, Any] = {}

    @property
    def description(self) -> str:
        base_desc = dedent(f"""
            This is a FreeCell QA environment.

            {self.FREECELL_RULES}

            Question Types:
            - Specified Card: Identify which card is at a specific position in a cascade pile
            - Valid Move: Determine which of the given moves is valid
            - Card After Move: Identify the top card of a pile after a move

            The system will present you with a game state and ask a specific question.
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

        # Initialize game
        self._initialize_game()

        # Select question type
        if self._question_type is None:
            question_type = random.choice(self.QUESTION_TYPES)["id"]
        elif isinstance(self._question_type, int):
            # Support integer indexing (0, 1, 2, ...)
            question_type = self.QUESTION_TYPES[self._question_type]["id"]
        else:
            question_type = self._question_type

        # Generate question
        if question_type == "specified_card":
            self._current_question = self._generate_specified_card_question()
        elif question_type == "valid_move":
            self._current_question = self._generate_valid_move_question()
        elif question_type == "card_after_move":
            self._current_question = self._generate_card_after_move_question()
        else:
            raise ValueError(f"Unknown question type: {question_type}")

        logger.info(
            f"Reset FreeCell QA (cascade_number={self._cascade_number}, question: {question_type})."
        )

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
        """Render the FreeCell game state."""
        # Image dimensions
        card_width, card_height = 60, 90
        spacing = 10
        margin = 20

        # Calculate image size
        img_width = margin * 2 + self._cascade_number * (card_width + spacing)
        img_height = margin * 2 + card_height * 2 + 400  # Space for cascade piles

        img = Image.new("RGB", (img_width, img_height), (47, 79, 79))  # Dark slate gray
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 16)
            small_font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 12)
        except Exception:
            small_font = ImageFont.load_default()

        # Draw free cells (top left)
        y_top = margin
        for i in range(4):
            x = margin + i * (card_width + spacing)
            self._draw_card_or_empty(
                draw,
                x,
                y_top,
                self._free_cells[i],
                card_width,
                card_height,
                font,
                small_font,
                f"F{i}",
            )

        # Draw foundation piles (top right)
        suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
        for i, suit in enumerate(suits):
            x = margin + (self._cascade_number - 4 + i) * (card_width + spacing)
            pile = self._foundation_piles[suit]
            top_card = pile[-1] if pile else None
            self._draw_card_or_empty(
                draw,
                x,
                y_top,
                top_card,
                card_width,
                card_height,
                font,
                small_font,
                suit.value,
            )

        # Draw cascade piles
        y_cascade = margin + card_height + 30
        for i, pile in enumerate(self._cascade_piles):
            x = margin + i * (card_width + spacing)

            # Draw pile label
            draw.text(
                (x + card_width // 2, y_cascade - 20),
                f"Pile {i}",
                fill=(255, 255, 255),
                font=small_font,
                anchor="mm",
            )

            if not pile:
                # Empty cascade pile
                draw.rectangle(
                    [x, y_cascade, x + card_width, y_cascade + card_height],
                    outline=(255, 255, 255),
                    width=2,
                )
            else:
                # Draw cards with overlap
                overlap = 25
                for j, card in enumerate(pile):
                    y = y_cascade + j * overlap
                    self._draw_card(
                        draw, x, y, card, card_width, card_height, font, True
                    )

        return img

    def _draw_card_or_empty(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        card: Card | None,
        width: int,
        height: int,
        font: ImageFont.ImageFont,
        small_font: ImageFont.ImageFont,
        label: str,
    ):
        """Draw a card or empty cell."""
        if card:
            self._draw_card(draw, x, y, card, width, height, font, False)
        else:
            # Empty cell
            draw.rectangle(
                [x, y, x + width, y + height],
                fill=(28, 53, 45),
                outline=(255, 255, 255),
                width=2,
            )
            draw.text(
                (x + width // 2, y + height // 2),
                label,
                fill=(100, 100, 100),
                font=small_font,
                anchor="mm",
            )

    def _draw_card(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        card: Card,
        width: int,
        height: int,
        font: ImageFont.ImageFont,
        draw_bottom: bool,
    ):
        """Draw a single card."""
        # Card background
        draw.rectangle(
            [x, y, x + width, y + height],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=2,
        )

        # Card color
        color = (255, 0, 0) if card.color == Color.RED else (0, 0, 0)

        # Draw value and suit
        value_str = self.VALUE_MAP[card.value]
        text = f"{value_str}\n{card.suit.value}"
        draw.text(
            (x + width // 2, y + height // 2),
            text,
            fill=color,
            font=font,
            anchor="mm",
            align="center",
        )

    def _initialize_game(self):
        """Initialize a new FreeCell game."""
        # Create deck
        cards = [Card(suit, value) for suit in Suit for value in range(1, 14)]
        random.shuffle(cards)

        # Deal cards to cascade piles
        self._cascade_piles = [[] for _ in range(self._cascade_number)]
        for i, card in enumerate(cards):
            pile_num = i % self._cascade_number
            self._cascade_piles[pile_num].append(card)

        # Reset free cells and foundation piles
        self._free_cells = [None] * 4
        self._foundation_piles = {suit: [] for suit in Suit}

    def _generate_specified_card_question(self) -> dict[str, Any]:
        """Generate question about a specific card in a cascade pile."""
        # Select non-empty pile
        non_empty_piles = [
            (i, pile) for i, pile in enumerate(self._cascade_piles) if pile
        ]
        if not non_empty_piles:
            # Fallback - shouldn't happen
            return {"question": "Error", "answer": "1", "options": ["Error"]}

        cascade_index, selected_pile = random.choice(non_empty_piles)
        n = random.randint(1, len(selected_pile))
        selected_card = selected_pile[-n]

        # Generate options
        correct_answer = f"({self.SUIT_TO_STRING[selected_card.suit.value]}, {self.VALUE_MAP[selected_card.value]})"
        options = [correct_answer]

        # Add distractor options
        all_cards = [card for pile in self._cascade_piles for card in pile]
        while len(options) < 8 and len(all_cards) > 0:
            card = random.choice(all_cards)
            option = f"({self.SUIT_TO_STRING[card.suit.value]}, {self.VALUE_MAP[card.value]})"
            if option not in options:
                options.append(option)

        random.shuffle(options)
        answer_index = options.index(correct_answer) + 1
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

        question = f"""{self.FREECELL_RULES}

We have {self._cascade_number} cascade piles, and their indexes are {list(range(self._cascade_number))}.
We have 4 freecells on the left top, and their indexes are 0,1,2,3.
We have 4 foundation piles on the right top, and their indexes are 0,1,2,3.

Question: Find the {n}-th card from the top of cascade pile {cascade_index}.

Options:
{options_text}"""

        return {"question": question, "answer": str(answer_index), "options": options}

    def _generate_valid_move_question(self) -> dict[str, Any]:
        """Generate question about valid moves."""
        valid_moves = self._get_valid_moves()

        if not valid_moves:
            # No valid moves, fallback to specified card question
            return self._generate_specified_card_question()

        correct_move = random.choice(valid_moves)

        # Format correct option
        from_str = correct_move["from"]
        to_str = correct_move["to"]
        card = correct_move["card"]

        correct_option = f"Move ({self.SUIT_TO_STRING[card.suit.value]},{self.VALUE_MAP[card.value]}) from {from_str} to {to_str}"
        options = [correct_option]

        # Generate invalid moves as distractors
        while len(options) < 4:
            # Random source and dest
            source_type = random.choice(["Cascade", "FreeCell"])
            if source_type == "Cascade":
                source_idx = random.randint(0, self._cascade_number - 1)
                source = f"Cascade {source_idx}"
            else:
                source_idx = random.randint(0, 3)
                source = f"FreeCell {source_idx}"

            dest_type = random.choice(["Cascade", "FreeCell", "Foundation"])
            if dest_type == "Cascade":
                dest_idx = random.randint(0, self._cascade_number - 1)
                dest = f"Cascade {dest_idx}"
            elif dest_type == "FreeCell":
                dest_idx = random.randint(0, 3)
                dest = f"FreeCell {dest_idx}"
            else:
                suit = random.choice(list(Suit))
                dest = f"Foundation {self.SUIT_TO_STRING[suit.value]}"

            if all_cards := [card for pile in self._cascade_piles for card in pile]:
                rand_card = random.choice(all_cards)
                option = f"Move ({self.SUIT_TO_STRING[rand_card.suit.value]},{self.VALUE_MAP[rand_card.value]}) from {source} to {dest}"
                if option not in options and option != correct_option:
                    options.append(option)

        random.shuffle(options)
        answer_index = options.index(correct_option) + 1
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

        question = f"""{self.FREECELL_RULES}

We have {self._cascade_number} cascade piles, and their indexes are {list(range(self._cascade_number))}.
We have 4 freecells on the left top, and their indexes are 0,1,2,3.
We have 4 foundation piles on the right top, and their indexes are 0,1,2,3.

Question: Which of the following moves is valid in the current game state?

Options:
{options_text}"""

        return {"question": question, "answer": str(answer_index), "options": options}

    def _generate_card_after_move_question(self) -> dict[str, Any]:
        """Generate question about card state after a move."""
        valid_moves = self._get_valid_moves()

        # Filter for moves from cascade piles only
        cascade_moves = [m for m in valid_moves if m["from"].startswith("Cascade")]

        if not cascade_moves:
            return self._generate_specified_card_question()

        selected_move = random.choice(cascade_moves)
        cascade_index = int(selected_move["from"].split()[-1])
        selected_pile = self._cascade_piles[cascade_index]

        if len(selected_pile) < 2:
            return self._generate_specified_card_question()

        # The card that will be revealed after the move
        revealed_card = selected_pile[-2]
        answer_text = f"({self.SUIT_TO_STRING[revealed_card.suit.value]}, {self.VALUE_MAP[revealed_card.value]})"
        options = [answer_text]

        # Add distractor options
        all_cards = [card for pile in self._cascade_piles for card in pile]
        while len(options) < 8 and len(all_cards) > 0:
            card = random.choice(all_cards)
            option = f"({self.SUIT_TO_STRING[card.suit.value]}, {self.VALUE_MAP[card.value]})"
            if option not in options:
                options.append(option)

        random.shuffle(options)
        answer_index = options.index(answer_text) + 1
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

        move_card = selected_move["card"]
        from_str = selected_move["from"]
        to_str = selected_move["to"]

        question = f"""{self.FREECELL_RULES}

We have {self._cascade_number} cascade piles, and their indexes are {list(range(self._cascade_number))}.
We have 4 freecells on the left top, and their indexes are 0,1,2,3.
We have 4 foundation piles on the right top, and their indexes are 0,1,2,3.

Question: Find the top card from cascade pile {cascade_index} after moving the card ({self.SUIT_TO_STRING[move_card.suit.value]},{self.VALUE_MAP[move_card.value]}) from {from_str} to {to_str}.

Options:
{options_text}"""

        return {"question": question, "answer": str(answer_index), "options": options}

    def _get_valid_moves(self) -> list[dict[str, Any]]:
        """Get all valid moves in the current state."""
        valid_moves = []

        # Moves from cascade piles
        for i, pile in enumerate(self._cascade_piles):
            if not pile:
                continue

            top_card = pile[-1]

            # To other cascade piles
            for j, dest_pile in enumerate(self._cascade_piles):
                if i == j:
                    continue
                if not dest_pile:
                    # Can move to empty cascade
                    valid_moves.append(
                        {"card": top_card, "from": f"Cascade {i}", "to": f"Cascade {j}"}
                    )
                elif (
                    top_card.color != dest_pile[-1].color
                    and top_card.value == dest_pile[-1].value - 1
                ):
                    # Can stack with alternating colors in descending order
                    valid_moves.append(
                        {"card": top_card, "from": f"Cascade {i}", "to": f"Cascade {j}"}
                    )

            # To free cells
            for j, cell in enumerate(self._free_cells):
                if cell is None:
                    valid_moves.append(
                        {
                            "card": top_card,
                            "from": f"Cascade {i}",
                            "to": f"FreeCell {j}",
                        }
                    )

            # To foundation piles
            foundation_pile = self._foundation_piles[top_card.suit]
            if not foundation_pile and top_card.value == 1:
                # Can place Ace
                valid_moves.append(
                    {
                        "card": top_card,
                        "from": f"Cascade {i}",
                        "to": f"Foundation {self.SUIT_TO_STRING[top_card.suit.value]}",
                    }
                )
            elif foundation_pile and top_card.value == foundation_pile[-1].value + 1:
                # Can place next card in sequence
                valid_moves.append(
                    {
                        "card": top_card,
                        "from": f"Cascade {i}",
                        "to": f"Foundation {self.SUIT_TO_STRING[top_card.suit.value]}",
                    }
                )

        # Moves from free cells
        for i, card in enumerate(self._free_cells):
            if card is None:
                continue

            # To cascade piles
            for j, dest_pile in enumerate(self._cascade_piles):
                if not dest_pile:
                    valid_moves.append(
                        {"card": card, "from": f"FreeCell {i}", "to": f"Cascade {j}"}
                    )
                elif (
                    card.color != dest_pile[-1].color
                    and card.value == dest_pile[-1].value - 1
                ):
                    valid_moves.append(
                        {"card": card, "from": f"FreeCell {i}", "to": f"Cascade {j}"}
                    )

            # To foundation piles
            foundation_pile = self._foundation_piles[card.suit]
            if not foundation_pile and card.value == 1:
                valid_moves.append(
                    {
                        "card": card,
                        "from": f"FreeCell {i}",
                        "to": f"Foundation {self.SUIT_TO_STRING[card.suit.value]}",
                    }
                )
            elif foundation_pile and card.value == foundation_pile[-1].value + 1:
                valid_moves.append(
                    {
                        "card": card,
                        "from": f"FreeCell {i}",
                        "to": f"Foundation {self.SUIT_TO_STRING[card.suit.value]}",
                    }
                )

        return valid_moves

    def _check_answer(self, action: str) -> bool:
        """Check if answer is correct."""
        correct_answer = self._current_question["answer"]
        return action.strip().lower() == correct_answer.strip().lower()

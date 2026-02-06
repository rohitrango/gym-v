"""FreeCell QA environment based on GameRL."""

from __future__ import annotations

from enum import Enum
from importlib import resources
import random
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger
from gym_v.utils.gamerl_utils import build_description, score_exact

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


class FreecellQAEnv(Env):
    # Meta: source=GameRL, category=puzzles, turn=single
    # Overrides: interaction_mode=single_turn, action_format=open_ended
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

    GAME_RULES = dedent("""
        In FreeCell, cards can be moved according to specific rules:
        - Cards in cascade piles must be stacked in descending order with alternating colors
        - Foundation piles must be built up by suit from Ace to King
        - Free cells can hold only one card each
        - A card can be moved to a free cell if available, stacked in descending order alternating colors in cascade piles, or placed in foundation piles starting from Ace
    """).strip()

    def __init__(
        self,
        cascade_number: int | None = None,
        question_type: int | None = None,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if cascade_number is None:
            cascade_number = random.choice([4, 6, 8])

        self._cascade_number = cascade_number
        self._question_type_param = question_type
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Game state
        self._cascade_piles: list[list[Card]] = [[] for _ in range(cascade_number)]
        self._free_cells: list[Card | None] = [None] * 4
        self._foundation_piles: dict[Suit, list[Card]] = {suit: [] for suit in Suit}

        # Standard QA variables
        self._question_type_idx: int = 0
        self._question: str = ""
        self._options: list[str] | None = None
        self._oracle_answer: str = ""

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        return build_description(
            game_name="FreeCell Solitaire",
            rules=self.GAME_RULES,
            question=self._question,
            options=self._options,
            oracle_answer=self._oracle_answer,
        )

    def _get_state_text(self) -> str:
        """Generate text description of current FreeCell game state.

        Returns a text representation matching the rendered image.
        """
        lines = []

        # Free cells
        freecells_str = []
        for i, cell in enumerate(self._free_cells):
            if cell is None:
                freecells_str.append(f"Cell {i}: empty")
            else:
                freecells_str.append(
                    f"Cell {i}: {self.VALUE_MAP[cell.value]}{self.SUIT_TO_STRING[cell.suit.value][0]}"
                )
        lines.append("FreeCells: " + ", ".join(freecells_str))

        # Foundation piles
        foundation_strs = []
        for suit in Suit:
            pile = self._foundation_piles[suit]
            if pile:
                top = pile[-1]
                foundation_strs.append(
                    f"{self.SUIT_TO_STRING[suit.value]}: {self.VALUE_MAP[top.value]} ({len(pile)} cards)"
                )
            else:
                foundation_strs.append(f"{self.SUIT_TO_STRING[suit.value]}: empty")
        lines.append("Foundations: " + ", ".join(foundation_strs))

        # Cascade piles
        for i, pile in enumerate(self._cascade_piles):
            if pile:
                cards_str = ", ".join(
                    [
                        f"{self.VALUE_MAP[c.value]}{self.SUIT_TO_STRING[c.suit.value][0]}"
                        for c in pile
                    ]
                )
                lines.append(f"Cascade {i}: {cards_str}")
            else:
                lines.append(f"Cascade {i}: empty")

        return "\n".join(lines)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        # Initialize game
        self._initialize_game()

        # Select question type
        if self._question_type_param is None:
            self._question_type_idx = random.randint(0, len(self.QUESTION_TYPES) - 1)
        else:
            if not (0 <= self._question_type_param < len(self.QUESTION_TYPES)):
                raise ValueError(
                    f"Invalid question type index: {self._question_type_param}"
                )
            self._question_type_idx = self._question_type_param

        q_type = self.QUESTION_TYPES[self._question_type_idx]

        # Generate question - sets _question, _options, _oracle_answer
        if q_type["id"] == "specified_card":
            result = self._generate_specified_card_question()
        elif q_type["id"] == "valid_move":
            result = self._generate_valid_move_question()
        elif q_type["id"] == "card_after_move":
            result = self._generate_card_after_move_question()
        else:
            raise ValueError(f"Unknown question type: {q_type['id']}")

        # Extract to instance variables
        self._question = result["question"]
        self._options = result.get("options")
        self._oracle_answer = result["answer"]

        logger.info(
            f"Reset FreeCell QA (cascade_number={self._cascade_number}, question: {q_type['id']})."
        )

        text_state = self._get_state_text()
        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "state_text": text_state,
                "text_prompt": f"{text_state}\n\n{self.description}",
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
        return score_exact(answer, self._oracle_answer)

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

        info: dict[str, Any] = {}
        reward = 0.0
        terminated = True
        truncated = False

        # Check answer
        reward = self._score_answer(action_str)
        correct = reward == 1.0

        if correct:
            response = "Correct!"
        else:
            response = f"Incorrect. The correct answer is: {self._oracle_answer}"

        info = {
            "correct": correct,
            "user_answer": action_str.strip(),
            "oracle_answer": self._oracle_answer,
        }

        obs = Observation(image=self.render(), text=response)

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

    def render(self) -> Image.Image | list[Image.Image] | None:
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
                        draw,
                        x,
                        y,
                        card,
                        card_width,
                        card_height,
                        font,
                        j == len(pile) - 1,
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
            self._draw_card(draw, x, y, card, width, height, font, True)
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
        """Draw a single card with playing card pattern."""
        # Card background
        draw.rectangle(
            [x, y, x + width, y + height],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=2,
        )

        # Card color
        color = (255, 0, 0) if card.color == Color.RED else (0, 0, 0)

        value_str = self.VALUE_MAP[card.value]
        suit_symbol = card.suit.value

        # Load fonts for card display
        try:
            corner_font = ImageFont.truetype(
                str(self.assets_dir / "DejaVuSans.ttf"), 10
            )
        except Exception:
            corner_font = ImageFont.load_default()

        # Use a single corner marker for all cards (top visible or covered).
        draw.text(
            (x + 5, y + 5),
            value_str,
            fill=color,
            font=corner_font,
            anchor="lt",
        )
        draw.text(
            (x + 5, y + 16),
            suit_symbol,
            fill=color,
            font=corner_font,
            anchor="lt",
        )

    def _draw_card_symbols(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        width: int,
        height: int,
        value: int,
        symbol: str,
        color: tuple,
        font: ImageFont.ImageFont,
    ):
        """Draw suit symbols in the center of the card based on value."""
        cx = x + width // 2
        cy = y + height // 2

        # Define symbol positions for different card values
        # Positions are relative offsets from center
        if value == 1:  # Ace
            positions = [(0, 0)]
        elif value == 2:
            positions = [(0, -height // 4), (0, height // 4)]
        elif value == 3:
            positions = [(0, -height // 4), (0, 0), (0, height // 4)]
        elif value == 4:
            positions = [
                (-width // 4, -height // 4),
                (width // 4, -height // 4),
                (-width // 4, height // 4),
                (width // 4, height // 4),
            ]
        elif value == 5:
            positions = [
                (-width // 4, -height // 4),
                (width // 4, -height // 4),
                (0, 0),
                (-width // 4, height // 4),
                (width // 4, height // 4),
            ]
        elif value == 6:
            positions = [
                (-width // 4, -height // 4),
                (width // 4, -height // 4),
                (-width // 4, 0),
                (width // 4, 0),
                (-width // 4, height // 4),
                (width // 4, height // 4),
            ]
        elif value == 7:
            positions = [
                (-width // 4, -height // 4),
                (width // 4, -height // 4),
                (0, -height // 8),
                (-width // 4, 0),
                (width // 4, 0),
                (-width // 4, height // 4),
                (width // 4, height // 4),
            ]
        elif value == 8:
            positions = [
                (-width // 4, -height // 4),
                (width // 4, -height // 4),
                (0, -height // 8),
                (-width // 4, 0),
                (width // 4, 0),
                (0, height // 8),
                (-width // 4, height // 4),
                (width // 4, height // 4),
            ]
        elif value == 9:
            positions = [
                (-width // 4, -height // 4),
                (width // 4, -height // 4),
                (-width // 4, -height // 8),
                (width // 4, -height // 8),
                (0, 0),
                (-width // 4, height // 8),
                (width // 4, height // 8),
                (-width // 4, height // 4),
                (width // 4, height // 4),
            ]
        elif value == 10:
            positions = [
                (-width // 4, -height // 4),
                (width // 4, -height // 4),
                (0, -height // 3),
                (-width // 4, -height // 8),
                (width // 4, -height // 8),
                (-width // 4, height // 8),
                (width // 4, height // 8),
                (0, height // 3),
                (-width // 4, height // 4),
                (width // 4, height // 4),
            ]
        else:  # J, Q, K (11, 12, 13)
            # For face cards, just draw one large symbol in center
            positions = [(0, 0)]

        # Draw symbols at calculated positions
        for dx, dy in positions:
            draw.text(
                (cx + dx, cy + dy),
                symbol,
                fill=color,
                font=font,
                anchor="mm",
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
        options_text = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])

        question = f"""{self.GAME_RULES}

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
        options_text = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])

        question = f"""{self.GAME_RULES}

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
        options_text = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])

        move_card = selected_move["card"]
        from_str = selected_move["from"]
        to_str = selected_move["to"]

        question = f"""{self.GAME_RULES}

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
        return action.strip().lower() == self._oracle_answer.strip().lower()

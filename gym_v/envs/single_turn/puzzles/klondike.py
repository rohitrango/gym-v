"""Klondike Solitaire QA environment for gym-v.

Klondike is the classic solitaire card game with:
- Stock pile (draw pile) - 24 cards initially
- Waste pile - cards drawn from stock
- 4 Foundation piles - build up from Ace to King by suit
- 7 Tableau piles - build down in alternating colors

This environment provides 5 question types about game moves and strategy.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
import random
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger
from gym_v.utils.gamerl_utils import build_description, score_exact

logger = get_logger()

# Game Rules
GAME_RULES = dedent("""
    The given image represents the interface of the game Klondike Solitaire. The user interface consists of a board with 52 playing cards divided into four distinct areas:

1. **Stock Pile (Draw Pile):** Initially composed of 24 face-down cards. The player can draw one card at a time to reveal its face.

2. **Waste Pile (Dump Pile):** This pile holds the cards drawn from the Stock Pile that have not been moved to other areas. Only the topmost card in the Waste Pile is available for play.

3. **Foundation Piles:** These four piles are designated for each suit (hearts, diamonds, clubs, and spades, but not necessarily following this order). From left to right, they are referred to as foundation 1 through foundation 4. Players must build up the foundation starting with the Ace and then place cards in ascending order (2 through King) of the same suit.

4. **Tableau Piles:** There are seven tableau piles. From left to right, these piles are referred to as Tab 1 through Tab 7, and initially contain an increasing number of cards from 1 to 7. Only the topmost cards in each pile are face-up and built in descending order, alternating colors (red and black suits). Only when the topmost cards are removed to some other place (e.g. another tableau pile or the foundation pile) will the hidden card beneath be revealed. Only a King can be placed on an empty tableau pile unless it starts there at the beginning of the game.

**Objective:**
The goal of Klondike Solitaire is to move all cards to the Foundation Piles, organized by suit in ascending order from Ace to King.
""").strip()


# ============================================================================
# Card and Game State Classes
# ============================================================================


class Card:
    """Represents a playing card."""

    RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    SUITS = ["heart", "spade", "diamond", "club"]
    RANK_VALUES = {
        "A": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
    }

    def __init__(self, rank: str, suit: str, faceup: bool = False):
        self.rank = rank
        self.suit = suit
        self.faceup = faceup
        self.color = "red" if suit in ["heart", "diamond"] else "black"

    def __repr__(self):
        return f"{self.suit} {self.rank}" if self.faceup else "<Card>"

    def rank_value(self) -> int:
        return self.RANK_VALUES[self.rank]


class KlondikeGame:
    """Manages the Klondike Solitaire game state."""

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

        # Create deck
        self.deck = [Card(rank, suit) for rank in Card.RANKS for suit in Card.SUITS]
        random.shuffle(self.deck)

        # Initialize game areas
        self.stock = []  # Draw pile
        self.waste = []  # Waste pile
        self.foundation = [[] for _ in range(4)]  # 4 foundation piles
        self.tableau = [[] for _ in range(7)]  # 7 tableau piles

        # Setup initial board
        self._setup_board()

    def _setup_board(self):
        """Set up initial Klondike board configuration."""
        deck_index = 0

        # Deal to tableau piles (1, 2, 3, ..., 7 cards)
        for i in range(7):
            for j in range(i + 1):
                card = self.deck[deck_index]
                if j == i:  # Top card is face up
                    card.faceup = True
                self.tableau[i].append(card)
                deck_index += 1

        # Remaining cards go to stock
        self.stock = self.deck[deck_index:]
        self.stock.reverse()  # So we can pop from end

    def can_move_to_tableau(self, card: Card, target_pile: list[Card]) -> bool:
        """Check if a card can be moved to a tableau pile."""
        if not card.faceup:
            return False

        if not target_pile:
            # Only King can go on empty tableau
            return card.rank == "K"

        target_card = target_pile[-1]
        if not target_card.faceup:
            return False

        # Must be descending rank and alternating color
        return (
            card.rank_value() == target_card.rank_value() - 1
            and card.color != target_card.color
        )

    def can_move_to_foundation(self, card: Card, foundation_pile: list[Card]) -> bool:
        """Check if a card can be moved to a foundation pile."""
        if not card.faceup:
            return False

        if not foundation_pile:
            # Only Ace can start foundation
            return card.rank == "A"

        top_card = foundation_pile[-1]
        # Must be same suit and ascending rank
        return (
            card.suit == top_card.suit
            and card.rank_value() == top_card.rank_value() + 1
        )

    def get_valid_moves(self) -> list[str]:
        """Get all valid moves in current game state."""
        valid_moves = []

        # Tableau to Tableau
        for from_idx in range(7):
            if self.tableau[from_idx]:
                source_card = self.tableau[from_idx][-1]
                for to_idx in range(7):
                    if from_idx != to_idx:
                        if self.can_move_to_tableau(source_card, self.tableau[to_idx]):
                            valid_moves.append(
                                f"Move from Tab{from_idx + 1} to Tab{to_idx + 1}"
                            )

        # Waste to Tableau
        if self.waste:
            waste_card = self.waste[-1]
            for tab_idx in range(7):
                if self.can_move_to_tableau(waste_card, self.tableau[tab_idx]):
                    valid_moves.append(f"Move from Waste Pile to Tab{tab_idx + 1}")

        # Waste to Foundation
        if self.waste:
            waste_card = self.waste[-1]
            for found_idx in range(4):
                if self.can_move_to_foundation(waste_card, self.foundation[found_idx]):
                    valid_moves.append(
                        f"Move from Waste Pile to Foundation {found_idx + 1}"
                    )

        # Tableau to Foundation
        for tab_idx in range(7):
            if self.tableau[tab_idx]:
                source_card = self.tableau[tab_idx][-1]
                for found_idx in range(4):
                    if self.can_move_to_foundation(
                        source_card, self.foundation[found_idx]
                    ):
                        valid_moves.append(
                            f"Move from Tab{tab_idx + 1} to Foundation {found_idx + 1}"
                        )

        return valid_moves

    def is_deadlock(self) -> bool:
        """Check if game is in a deadlock state."""
        # Has stock cards to draw
        if self.stock:
            return False

        # Has valid moves
        valid_moves = self.get_valid_moves()
        if valid_moves:
            return False

        # Check if any tableau moves would reveal cards
        for tab_idx in range(7):
            if len(self.tableau[tab_idx]) > 1:
                # If there's a face-down card that could be revealed
                for card in self.tableau[tab_idx][:-1]:
                    if not card.faceup:
                        return False

        return True

    def get_board_state(self) -> dict:
        """Get string representation of board state for rendering."""
        state = {
            "stock": len(self.stock),
            "waste": [str(card) for card in self.waste[-3:]] if self.waste else [],
            "foundation": [[str(card) for card in pile] for pile in self.foundation],
            "tableau": [[str(card) for card in pile] for pile in self.tableau],
        }
        return state


# ============================================================================
# PIL Rendering
# ============================================================================


def render_klondike_board(
    game: KlondikeGame, assets_dir: Path | None = None
) -> Image.Image:
    """Render Klondike board using PIL with actual card graphics from assets."""
    if assets_dir is None:
        assets_dir = resources.files("gym_v.envs") / "assets" / "klondike"

    # Load card templates
    try:
        card_front_template = Image.open(assets_dir / "card_front.png").convert("RGBA")
        card_back = Image.open(assets_dir / "card_back.png").convert("RGBA")

        # Load suit images
        big_suits = {
            "heart": Image.open(assets_dir / "big_heart.png").convert("RGBA"),
            "diamond": Image.open(assets_dir / "big_diamond.png").convert("RGBA"),
            "spade": Image.open(assets_dir / "big_spade.png").convert("RGBA"),
            "club": Image.open(assets_dir / "big_club.png").convert("RGBA"),
        }

        small_suits = {
            "heart": Image.open(assets_dir / "small_heart.png").convert("RGBA"),
            "diamond": Image.open(assets_dir / "small_diamond.png").convert("RGBA"),
            "spade": Image.open(assets_dir / "small_spade.png").convert("RGBA"),
            "club": Image.open(assets_dir / "small_club.png").convert("RGBA"),
        }

        # Load face cards
        face_cards = {
            "red": {
                "J": Image.open(assets_dir / "red_J.png").convert("RGBA"),
                "Q": Image.open(assets_dir / "red_Q.png").convert("RGBA"),
                "K": Image.open(assets_dir / "red_K.png").convert("RGBA"),
            },
            "blue": {
                "J": Image.open(assets_dir / "blue_J.png").convert("RGBA"),
                "Q": Image.open(assets_dir / "blue_Q.png").convert("RGBA"),
                "K": Image.open(assets_dir / "blue_K.png").convert("RGBA"),
            },
        }

        # Load number/letter sprites
        letters_img = Image.open(assets_dir / "letters.png").convert("RGBA")
        numbers_img = Image.open(assets_dir / "numbers.png").convert("RGBA")

    except Exception as e:
        logger.warning(f"Failed to load klondike assets: {e}, using fallback rendering")
        return _render_klondike_fallback(game)

    # Card dimensions (from original implementation)
    card_width = card_front_template.width
    card_height = card_front_template.height

    # Create main image with title and decorative elements
    title_height = 60
    # border = 10  # Unused variable
    padding = 15
    width = 800
    height = 650

    # Create image with dark green background (matching Klondike theme)
    img = Image.new("RGB", (width, height), (53, 101, 77))  # Dark green felt
    draw = ImageDraw.Draw(img)

    # Load font for title and labels
    try:
        title_font = ImageFont.truetype(str(assets_dir.parent / "DejaVuSans.ttf"), 36)
        label_font = ImageFont.truetype(str(assets_dir.parent / "DejaVuSans.ttf"), 12)
    except Exception:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    # Draw title
    title_text = "Klondike"
    bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_w = bbox[2] - bbox[0]
    draw.text(
        ((width - title_w) // 2, 15), title_text, fill=(255, 255, 255), font=title_font
    )

    # Helper function to extract character from sprite sheet
    def get_char_image(img, index, char_width=5, char_height=8, is_ten=False):
        width = 8 if is_ten else char_width
        box = (index * char_width, 0, index * char_width + width, char_height)
        return img.crop(box)

    # Helper function to create card image
    def create_card(card_obj):
        """Create a complete card image with suit symbols and rank."""
        card_img = card_front_template.copy()
        # card_draw = ImageDraw.Draw(card_img)  # Unused variable

        rank = card_obj.rank
        suit = card_obj.suit
        color = card_obj.color

        # Get suit images
        big_suit = big_suits[suit]
        small_suit = small_suits[suit]

        # Get rank image
        if rank in ["J", "Q", "K"]:
            # Use face card image
            face_color = "red" if color == "red" else "blue"
            face_img = face_cards[face_color][rank]
            # Paste face image in center
            x = (card_img.width - face_img.width) // 2
            y = (card_img.height - face_img.height) // 2
            card_img.paste(face_img, (x, y), face_img)

            # Get letter for corners
            letter_map = {"A": 0, "K": 1, "Q": 2, "J": 3}
            rank_img = get_char_image(letters_img, letter_map[rank])
        elif rank == "A":
            # Ace - one large suit in center
            x = (card_img.width - big_suit.width) // 2
            y = (card_img.height - big_suit.height) // 2
            card_img.paste(big_suit, (x, y), big_suit)
            rank_img = get_char_image(letters_img, 0)  # 'A'
        else:
            # Number cards - draw suit pattern
            num = int(rank) if rank not in ["J", "Q", "K", "A"] else 1
            _draw_suit_pattern(card_img, big_suit, num, suit)

            # Get number image
            num_map = {2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1, 9: 0, 10: 8}
            rank_img = get_char_image(numbers_img, num_map[num], is_ten=(num == 10))

        # Color the rank image to match suit color
        rank_colored = Image.new("RGBA", rank_img.size)
        suit_color = big_suit.getpixel((big_suit.width // 2, big_suit.height // 2))
        for y in range(rank_img.height):
            for x in range(rank_img.width):
                r, g, b, a = rank_img.getpixel((x, y))
                if a > 0:
                    rank_colored.putpixel(
                        (x, y), (suit_color[0], suit_color[1], suit_color[2], a)
                    )

        # Paste rank in corners
        card_img.paste(rank_colored, (4, 3), rank_colored)
        rank_rotated = rank_colored.rotate(180)
        card_img.paste(
            rank_rotated,
            (
                card_img.width - rank_colored.width - 4,
                card_img.height - rank_colored.height - 3,
            ),
            rank_rotated,
        )

        # Paste small suits in corners
        card_img.paste(small_suit, (6 - small_suit.width // 2, 12), small_suit)
        small_suit_rotated = small_suit.rotate(180)
        card_img.paste(
            small_suit_rotated,
            (
                card_img.width - 6 - small_suit.width // 2,
                card_img.height - 12 - small_suit.height,
            ),
            small_suit_rotated,
        )

        return card_img

    def _draw_suit_pattern(card_img, suit_img, rank, suit):
        """Draw suit symbols in pattern based on rank."""
        cw = card_img.width
        ch = card_img.height
        sw = suit_img.width
        sh = suit_img.height

        x_margin = 13
        y_margin = 9

        positions = []

        # Define positions for each rank
        if rank in [1, 3, 5, 7, 9]:
            # Center position
            positions.append((cw // 2 - sw // 2, ch // 2 - sh // 2))

        if rank in [2, 3]:
            positions.extend(
                [(cw // 2 - sw // 2, y_margin), (cw // 2 - sw // 2, ch - y_margin - sh)]
            )

        if rank >= 4:
            positions.extend(
                [
                    (x_margin, y_margin),
                    (cw - x_margin - sw, y_margin),
                    (x_margin, ch - y_margin - sh),
                    (cw - x_margin - sw, ch - y_margin - sh),
                ]
            )

        if rank in [6, 7, 8]:
            positions.extend(
                [(x_margin, ch // 2 - sh // 2), (cw - x_margin - sw, ch // 2 - sh // 2)]
            )

        if rank == 7:
            positions.append((cw // 2 - sw // 2, ch // 4))

        if rank >= 8:
            positions.extend(
                [
                    (x_margin, ch // 2 - sh - 2),
                    (cw - x_margin - sw, ch // 2 - sh - 2),
                    (x_margin, ch // 2 + 2),
                    (cw - x_margin - sw, ch // 2 + 2),
                ]
            )

        if rank == 10:
            positions.extend(
                [
                    (cw // 2 - sw // 2, ch // 4 - sh // 4),
                    (cw // 2 - sw // 2, ch - ch // 4 + sh // 4),
                ]
            )

        # Paste suits at positions
        for i, (x, y) in enumerate(positions):
            if i >= len(positions) // 2 and rank >= 2:
                # Rotate bottom half
                rotated = suit_img.rotate(180)
                card_img.paste(rotated, (x, y), rotated)
            else:
                card_img.paste(suit_img, (x, y), suit_img)

    # Layout parameters
    top_y = title_height + 20
    stock_x = 20

    # Draw Stock pile
    if game.stock:
        # Show card back
        img.paste(card_back, (stock_x, top_y), card_back)
        # Draw count
        draw.text(
            (stock_x + 5, top_y + card_height + 3),
            f"{len(game.stock)}",
            fill="white",
            font=label_font,
        )
    else:
        # Empty outline
        draw.rectangle(
            [stock_x, top_y, stock_x + card_width, top_y + card_height],
            outline=(150, 150, 150),
            width=2,
        )
    draw.text((stock_x, top_y - 15), "Stock", fill="white", font=label_font)

    # Draw Waste pile
    waste_x = stock_x + card_width + padding
    if game.waste:
        card = game.waste[-1]
        card_img = create_card(card)
        img.paste(card_img, (waste_x, top_y), card_img)
    else:
        draw.rectangle(
            [waste_x, top_y, waste_x + card_width, top_y + card_height],
            outline=(150, 150, 150),
            width=2,
        )
    draw.text((waste_x, top_y - 15), "Waste", fill="white", font=label_font)

    # Draw Foundation piles
    found_x_start = waste_x + card_width + padding * 3
    for i in range(4):
        found_x = found_x_start + i * (card_width + padding)
        if game.foundation[i]:
            card = game.foundation[i][-1]
            card_img = create_card(card)
            img.paste(card_img, (found_x, top_y), card_img)
        else:
            draw.rectangle(
                [found_x, top_y, found_x + card_width, top_y + card_height],
                outline=(150, 150, 150),
                width=2,
            )
            draw.text(
                (found_x + card_width // 2 - 10, top_y + card_height // 2 - 6),
                f"F{i + 1}",
                fill=(180, 180, 180),
                font=label_font,
            )
        draw.text((found_x, top_y - 15), f"F{i + 1}", fill="white", font=label_font)

    # Draw Tableau piles
    tab_y = top_y + card_height + 50
    overlap = 25  # How much cards overlap

    for i in range(7):
        tab_x = 20 + i * (card_width + padding)

        # Draw label
        draw.text((tab_x, tab_y - 15), f"Tab{i + 1}", fill="white", font=label_font)

        if game.tableau[i]:
            y_offset = 0
            for card in game.tableau[i]:
                card_y = tab_y + y_offset

                if card.faceup:
                    card_img = create_card(card)
                    img.paste(card_img, (tab_x, card_y), card_img)
                else:
                    # Show card back
                    img.paste(card_back, (tab_x, card_y), card_back)

                y_offset += overlap
        else:
            # Empty tableau outline
            draw.rectangle(
                [tab_x, tab_y, tab_x + card_width, tab_y + card_height],
                outline=(150, 150, 150),
                width=2,
            )

    return img


def _render_klondike_fallback(game: KlondikeGame) -> Image.Image:
    """Fallback rendering if assets cannot be loaded."""
    width = 800
    height = 600
    img = Image.new("RGB", (width, height), (0, 100, 0))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    draw.text((20, 20), "Klondike (Asset Loading Failed)", fill="white", font=font)
    draw.text((20, 50), f"Stock: {len(game.stock)} cards", fill="white", font=font)
    draw.text((20, 80), f"Waste: {len(game.waste)} cards", fill="white", font=font)

    return img


# ============================================================================
# Klondike QA Environment
# ============================================================================


class KlondikeQAEnv(Env):
    # Meta: source=GameRL, category=puzzles, turn=single
    # Overrides: interaction_mode=single_turn, action_format=open_ended
    """Klondike Solitaire QA environment.

    Question types:
    0. Board State - Which move is valid? (Medium, MCQ)
    1. Deadlock Detection - Is the game in deadlock? (Hard, MCQ)
    2. Move Effectiveness - Which move is most effective? (Hard, MCQ)
    3. Move Validity - Is a specific move valid? (Easy, MCQ)
    4. Foundation Move - Can a card be moved to foundation? (Easy, MCQ)
    """

    QUESTION_TYPES = [
        {
            "id": "board_state",
            "name": "Valid Move Selection",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "deadlock",
            "name": "Deadlock Detection",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "move_effectiveness",
            "name": "Effective Move",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "Strategy Optimization",
        },
        {
            "id": "move_validity",
            "name": "Move Validity",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "Action Outcome",
        },
        {
            "id": "foundation_move",
            "name": "Foundation Move",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "Action Outcome",
        },
    ]

    def __init__(
        self, question_type: int | None = None, num_players: int = 1, **kwargs
    ):
        super().__init__(**kwargs)
        self._question_type_param = question_type
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}
        self._game = None

        # Standard QA variables
        self._question_type_idx: int = 0
        self._question: str = ""
        self._options: list[str] | None = None
        self._oracle_answer: str = ""

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        return build_description(
            game_name="Klondike Solitaire",
            rules=GAME_RULES,
            question=self._question,
            options=self._options,
            oracle_answer=self._oracle_answer,
        )

    def _get_state_text(self) -> str:
        """Generate text description of current Klondike game state.

        Returns a text representation matching the rendered image.
        """
        lines = []

        # Stock pile
        stock_count = len(self._game.stock)
        lines.append(f"Stock: {stock_count} cards")

        # Waste pile
        if self._game.waste:
            top_cards = self._game.waste[-min(3, len(self._game.waste)) :]
            waste_str = ", ".join([f"{c.rank}{c.suit[0].upper()}" for c in top_cards])
            lines.append(f"Waste (top {len(top_cards)}): {waste_str}")
        else:
            lines.append("Waste: empty")

        # Foundation piles
        for i, pile in enumerate(self._game.foundation, 1):
            if pile:
                top = pile[-1]
                lines.append(
                    f"Foundation {i}: {top.rank}{top.suit[0].upper()} ({len(pile)} cards)"
                )
            else:
                lines.append(f"Foundation {i}: empty")

        # Tableau piles
        for i, pile in enumerate(self._game.tableau, 1):
            if pile:
                faceup = [c for c in pile if c.faceup]
                hidden_count = len(pile) - len(faceup)
                if hidden_count > 0:
                    cards_str = f"[{hidden_count} hidden], " + ", ".join(
                        [f"{c.rank}{c.suit[0].upper()}" for c in faceup]
                    )
                else:
                    cards_str = ", ".join(
                        [f"{c.rank}{c.suit[0].upper()}" for c in faceup]
                    )
                lines.append(f"Tableau {i}: {cards_str}")
            else:
                lines.append(f"Tableau {i}: empty")

        return "\n".join(lines)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        # Create game
        self._game = KlondikeGame(seed=seed)

        # Simulate some moves to create interesting state
        num_moves = random.randint(5, 15)
        for _ in range(num_moves):
            # Draw cards
            if self._game.stock and random.random() < 0.3:
                card = self._game.stock.pop()
                card.faceup = True
                self._game.waste.append(card)

            # Make random valid move
            valid_moves = self._game.get_valid_moves()
            if valid_moves and random.random() < 0.4:
                # Simulate a move (simplified for demo)
                pass

        # Select question type
        if self._question_type_param is not None:
            self._question_type_idx = self._question_type_param
        else:
            self._question_type_idx = random.randint(0, 4)
        q_type = self.QUESTION_TYPES[self._question_type_idx]

        # Generate question - sets _question, _options, _oracle_answer
        if self._question_type_idx == 0:
            result = self._generate_board_state_question()
        elif self._question_type_idx == 1:
            result = self._generate_deadlock_question()
        elif self._question_type_idx == 2:
            result = self._generate_effectiveness_question()
        elif self._question_type_idx == 3:
            result = self._generate_validity_question()
        elif self._question_type_idx == 4:
            result = self._generate_foundation_question()

        # Extract to instance variables
        self._question = result["question"]
        self._options = result.get("options")
        self._oracle_answer = result["answer"]

        # Render
        img = render_klondike_board(self._game)
        text_state = self._get_state_text()
        obs = Observation(
            image=img,
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

        # Re-render
        img = render_klondike_board(self._game)
        obs = Observation(image=img, text=response)

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

    def _generate_board_state_question(self) -> dict:
        """Type 0: Which move is valid?"""
        valid_moves = self._game.get_valid_moves()

        # Generate options
        options = [None] * 8

        if valid_moves:
            correct_move = random.choice(valid_moves)
            correct_answer = random.randint(1, 7)
            options[correct_answer - 1] = correct_move
        else:
            correct_move = "No possible moves from the options"
            correct_answer = 8
            options[7] = correct_move

        # Fill with invalid moves
        available_positions = [i for i in range(8) if options[i] is None]
        while available_positions:
            fake_move = random.choice(
                [
                    f"Move from Tab{random.randint(1, 7)} to Tab{random.randint(1, 7)}",
                    f"Move from Waste Pile to Tab{random.randint(1, 7)}",
                    f"Move from Tab{random.randint(1, 7)} to Foundation {random.randint(1, 4)}",
                    f"Move from Waste Pile to Foundation {random.randint(1, 4)}",
                ]
            )

            if fake_move not in valid_moves and fake_move not in options:
                pos = random.choice(available_positions)
                options[pos] = fake_move
                available_positions.remove(pos)

        if correct_answer != 8:
            options[7] = "No possible moves from the options"

        # Build question (without embedded options)
        question = "Which of the following moves is valid?"

        # Analysis
        analysis = "Current board state analysis:\n"
        if valid_moves:
            analysis += f"Valid moves available: {', '.join(valid_moves[:3])}\n"
            analysis += f"The correct answer is: {correct_move}"
        else:
            analysis += "No valid moves available from the options."

        return {
            "question": question,
            "answer": str(correct_answer),
            "options": options,
            "analysis": analysis,
        }

    def _generate_deadlock_question(self) -> dict:
        """Type 1: Is the game in a deadlock?"""
        is_deadlock = self._game.is_deadlock()

        question = "Is the current game state in a deadlock?"
        options = ["Yes", "No"]

        answer = "1" if is_deadlock else "2"

        analysis = "Deadlock analysis:\n"
        if is_deadlock:
            analysis += "The game is in deadlock. No stock cards remain, no valid moves available, and no face-down cards can be revealed."
        else:
            if self._game.stock:
                analysis += "Not in deadlock: stock cards can still be drawn."
            else:
                valid_moves = self._game.get_valid_moves()
                if valid_moves:
                    analysis += (
                        f"Not in deadlock: valid moves available ({valid_moves[0]})."
                    )
                else:
                    analysis += (
                        "Not in deadlock: face-down cards can still be revealed."
                    )

        return {
            "question": question,
            "answer": answer,
            "options": options,
            "analysis": analysis,
        }

    def _generate_effectiveness_question(self) -> dict:
        """Type 2: Which move is most effective?"""
        valid_moves = self._game.get_valid_moves()

        if not valid_moves:
            # Regenerate game
            self._game = KlondikeGame()
            return self._generate_effectiveness_question()

        # Prioritize foundation moves
        foundation_moves = [m for m in valid_moves if "Foundation" in m]

        if foundation_moves:
            best_move = random.choice(foundation_moves)
        else:
            best_move = random.choice(valid_moves)

        # Generate options
        options = [best_move]
        while len(options) < 4:
            other_move = (
                random.choice(valid_moves)
                if len(valid_moves) > 1
                else f"Move from Tab{random.randint(1, 7)} to Tab{random.randint(1, 7)}"
            )
            if other_move not in options:
                options.append(other_move)

        random.shuffle(options)
        answer = str(options.index(best_move) + 1)

        question = "Which move is most effective?"

        analysis = f"The most effective move is: {best_move}\n"
        if "Foundation" in best_move:
            analysis += "Foundation moves are always effective as they progress toward the goal."
        else:
            analysis += "This move helps reveal hidden cards or creates strategic opportunities."

        return {
            "question": question,
            "answer": answer,
            "options": options,
            "analysis": analysis,
        }

    def _generate_validity_question(self) -> dict:
        """Type 3: Is a specific move valid?"""
        # Pick a random move to ask about
        all_possible = [
            f"Move from Tab{random.randint(1, 7)} to Tab{random.randint(1, 7)}",
            f"Move from Waste Pile to Tab{random.randint(1, 7)}",
            f"Move from Tab{random.randint(1, 7)} to Foundation {random.randint(1, 4)}",
        ]

        test_move = random.choice(all_possible)
        valid_moves = self._game.get_valid_moves()
        is_valid = test_move in valid_moves

        question = f"Is the following move valid?\n{test_move}"
        options = ["Yes", "No"]

        answer = "1" if is_valid else "2"

        analysis = f"The move '{test_move}' is {'valid' if is_valid else 'invalid'}.\n"
        if is_valid:
            analysis += "This move follows the game rules."
        else:
            analysis += "This move violates the game rules."

        return {
            "question": question,
            "answer": answer,
            "options": options,
            "analysis": analysis,
        }

    def _generate_foundation_question(self) -> dict:
        """Type 4: Can a card be moved to foundation?"""
        # Check if any card can move to foundation
        can_move = False
        card_desc = None

        # Check waste pile
        if self._game.waste:
            card = self._game.waste[-1]
            for found_idx in range(4):
                if self._game.can_move_to_foundation(
                    card, self._game.foundation[found_idx]
                ):
                    can_move = True
                    card_desc = f"{card.suit} {card.rank} from Waste Pile"
                    break

        # Check tableau
        if not can_move:
            for tab_idx in range(7):
                if self._game.tableau[tab_idx]:
                    card = self._game.tableau[tab_idx][-1]
                    for found_idx in range(4):
                        if self._game.can_move_to_foundation(
                            card, self._game.foundation[found_idx]
                        ):
                            can_move = True
                            card_desc = f"{card.suit} {card.rank} from Tab{tab_idx + 1}"
                            break
                if can_move:
                    break

        question = "Can any card be moved to a foundation pile?"
        options = ["Yes", "No"]

        answer = "1" if can_move else "2"

        analysis = ""
        if can_move:
            analysis = f"Yes, {card_desc} can be moved to a foundation pile."
        else:
            analysis = "No cards can currently be moved to foundation piles."

        return {
            "question": question,
            "answer": answer,
            "options": options,
            "analysis": analysis,
        }

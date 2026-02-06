"""Spider Solitaire QA Environment

This environment implements a single-turn QA environment for Spider Solitaire with 7 question types.
"""

from __future__ import annotations

from importlib import resources
import random
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger
from gym_v.utils.gamerl_utils import build_description, score_exact

logger = get_logger()

# Card constants
ACE = 1
JACK = 11
QUEEN = 12
KING = 13
ALLRANKS = range(1, 14)

RANKNAMES = [
    "",
    "Ace",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "Jack",
    "Queen",
    "King",
]

# Rendering constants
CARDWIDTH = 71
CARDHEIGHT = 96
MARGIN = 2
XSPACING = CARDWIDTH + MARGIN
YSPACING = CARDHEIGHT + MARGIN
OFFSET1 = 12  # Face-down card overlap
OFFSET2 = 25  # Face-up card overlap

MAX_STOCK_DISPLAY = 10
STOCK_DELTA_X = 20

BACKGROUND = (7, 112, 7)
OUTLINE = (6, 96, 6)


# ============================================================================
# Stack Classes (from model.py)
# ============================================================================


class Stack(list):
    """A pile of cards."""

    def __init__(self):
        super().__init__()

    def add(self, card, face_up: bool = True) -> None:
        """Add a card to the stack.

        Args:
            card: The card to add.
            face_up: Whether the card should be face up (default True).
        """
        self.append(card)
        if face_up:
            self[-1].showFace()

    def isEmpty(self) -> bool:
        """Check if the stack is empty.

        Returns:
            True if the stack is empty, False otherwise.
        """
        return not self

    def clear(self) -> None:
        """Clear all cards from the stack."""
        self[:] = []

    def find(self, code: int) -> int:
        """Find a card by its code.

        Args:
            code: The unique code of the card to find.

        Returns:
            The index of the card if found, -1 otherwise.
        """
        for idx, card in enumerate(self):
            if card.code == code:
                return idx
        return -1


class SelectableStack(Stack):
    """
    A stack from which cards can be chosen, if they are face up and in sequence,
    from the top of the stack.
    """

    def __init__(self):
        super().__init__()

    def grab(self, n: int) -> list:
        """Remove the card at index n and all cards above it.

        Args:
            n: The starting index.

        Returns:
            List of cards removed from the stack.
        """
        answer = self[n:]
        del self[n:]
        return answer

    def replace(self, cards: list) -> None:
        """Replace cards on the stack (aborted move).

        Args:
            cards: The cards to put back.
        """
        self.extend(cards)

    def canSelect(self, idx: int) -> bool:
        """Check if a card at the given index can be selected.

        Args:
            idx: The index of the card to check.

        Returns:
            True if the card can be selected, False otherwise.
        """
        if idx >= len(self):
            return False
        if self[idx].faceDown():
            return False
        if not Card.isDescending(self[idx:]):
            return False
        return True


class OneWayStack(Stack):
    """Stack for stock and foundations where cards cannot be selected.

    Cards are either all face up or all face down.

    Args:
        face_up: Whether all cards in this stack should be face up.
    """

    def __init__(self, face_up: bool):
        super().__init__()
        self.face_up = face_up

    def add(self, card) -> None:
        """Add a card to the stack with the stack's face orientation.

        Args:
            card: The card to add.
        """
        super().add(card, self.face_up)


# ============================================================================
# Card Class (from model.py)
# ============================================================================


class Card:
    """A playing card identified by rank, suit, and back color.

    Attributes:
        circular: Class variable for circular mode (King can be placed on Ace).
    """

    circular = False

    def __init__(self, rank: int, suit: str, back: str, code: int):
        self.rank = rank
        self.suit = suit
        self.back = back
        self.up = False  # all cards are initially face down
        self.code = code

    def showFace(self) -> None:
        """Turn the card face up."""
        self.up = True

    def showBack(self) -> None:
        """Turn the card face down."""
        self.up = False

    def faceUp(self) -> bool:
        """Check if the card is face up.

        Returns:
            True if the card is face up, False otherwise.
        """
        return self.up

    def faceDown(self) -> bool:
        """Check if the card is face down.

        Returns:
            True if the card is face down, False otherwise.
        """
        return not self.faceUp()

    def __lt__(self, other):
        if self.suit != other.suit:
            return False
        answer = self.rank == other.rank - 1 or (
            self.circular and self.rank == KING and other.rank == ACE
        )
        return answer

    def __gt__(self, other):
        return other < self

    def __repr__(self):
        return f"{self.suit} {RANKNAMES[self.rank]} {self.back}"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def isDescending(seq: list) -> bool:
        """Check if cards form a descending sequence of the same suit.

        Args:
            seq: Sequence of cards to check.

        Returns:
            True if cards are in descending order of the same suit.
        """
        return all(x > y for x, y in zip(seq, seq[1:], strict=False))


# ============================================================================
# Spider Solitaire QA Environment
# ============================================================================

GAME_RULES = dedent("""
    Spider Solitaire

# OBJECTIVE
Spider is played with eight decks of 13 spade cards each, totaling 104 unique cards. The goal is to arrange all cards in a King-to-Ace sequence in the same suit and move them to the foundation piles. Once all sequences are moved to the foundations, the game is won.

# SETUP
The game features waste piles, a stock pile, and foundation piles. Waste piles are where the action happens, and the stock pile provides new cards when necessary.

**Waste Pile Numbering**: Waste piles are numbered from left to right starting with `0`. The cards within each waste pile are also numbered starting from the bottom card.

# GAME BOARD COMPONENTS

## **Stock Pile**
The **Stock Pile** holds all remaining cards and is used to deal new cards into the waste piles.
Stock Pile is in the top left corner of the board.

- **Staggered Card Stacking**: Cards are stacked in layers, and the number of layers indicates how many more times you can deal cards to the waste piles. Each deal moves one card face-up to each waste pile.

## **Waste Piles**
The **Waste Piles** are where cards are played and organized.
Waste Piles are on the bottom of the chessboard

- **Face-Up vs. Face-Down Cards**: Cards are stacked with face-up cards visible and face-down cards hidden. Only face-up cards can be played. When a face-down card becomes the top card of a pile, it is turned face-up and can be played.

- **Staggered Cards**: Cards in each waste pile are arranged so that face-up cards are on top, and face-down cards are beneath. As you move cards, new face-down cards are revealed.

- **Card Numbering and Screen Position**:
  - **Waste Pile Numbering**: Piles are numbered from left to right starting with `0` for the leftmost pile.
  - The card at the bottom of each waste pile (usually face-down) is numbered **0** and is the **topmost visible card** in the pile.
  - As you move upward in the pile, the next cards are numbered **1**, **2**, **3**, and so on.
  - Visually, the bottom card (number **0**) is the one closest to the top of the screen, and the cards above it are stacked above in the pile, going downwards.

## **Foundation Pile**
Foundation pile stores all the arranged suit. When a suit is arranged in sequence, it may be removed to a foundation pile. If all suits are moved to the foundations, the game is won.
Foundation Pile is in the top right corner of the board.

# MOVING CARDS
- **Movement Conditions**:
  - **Move a single card**: The single card being moved must be placed on a top card that is of the **same suit** and has a **higher rank** (e.g., a Q can be placed on a K).
  - **Move multiple cards**: A complete **descending sequence** of cards (such as K, Q, J, 10, etc.) can be moved from one pile to another. When moving a descending sequence of cards to another pile, the new sequence must be a **same-suit sequence** and follow the **descending order** from K, Q, J, 10, 9, ..., 2, A.
- **Face-Down Cards**: If the sequence you are moving includes face-down cards, they will be flipped face-up once they are moved. After flipping, the newly face-up cards can continue to be moved or interacted with.
- **Example**: If you have a sequence of K-Q-J-10-9-8-7 in the same suit, you can move a card 6 that has the same suit to the top of this pile, resulting in a new sequence K-Q-J-10-9-8-7-6.
- **Empty Pile Rule**: An empty waste pile can accept any card. After placing the card, you can continue adding a descending same-suit sequence to that pile.
- **Reveal Cards**: If a move leaves a face-down card on top, it will be turned face-up.

# DEALING
Click the stock to deal a new row of face-up cards to the waste piles. You may not deal if there is an empty waste pile.

# STRATEGY
- Turn face-down cards face-up.
- Form runs of the same suit in descending order.
- Use empty waste piles strategically.

# VARIANTS
In **circular spider solitaire**, a King can be placed on an Ace, allowing for extended sequences.

# **NOTE: Important Numbering Reminder**
- **Waste Pile Numbering**: Waste piles are numbered from **left to right** starting with `0` for the leftmost pile.
- **Card Numbering within Waste Piles**: The **bottom-most card** of each pile (usually face-down) is numbered **0**, and the cards above it are numbered **1**, **2**, **3**, etc., moving upwards in the pile.
- **Please Pay Attention** to both the waste pile and card numbering methods, as they will help you navigate and make strategic decisions effectively.
""").strip()


class SpiderSolitaireQAEnv(Env):
    # Meta: source=GameRL, category=puzzles, turn=single
    # Overrides: interaction_mode=single_turn, action_format=open_ended
    """Spider Solitaire QA Environment.

    A single-turn QA environment for Spider Solitaire with 7 question types.
    Players answer questions about game state, valid moves, and optimal strategies.
    """

    QUESTION_TYPES = [
        {
            "id": "state_info_stock",
            "name": "State Info: Stock Deals",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "StateInfo",
        },
        {
            "id": "state_info_top",
            "name": "State Info: Top Card",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "StateInfo",
        },
        {
            "id": "state_info_facedown",
            "name": "State Info: Face-Down Count",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "StateInfo",
        },
        {
            "id": "state_info_simulate",
            "name": "State Info: Simulated Clicks",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "StateInfo",
        },
        {
            "id": "action_outcome",
            "name": "Action Outcome: Move Validation",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "ActionOutcome",
        },
        {
            "id": "transition_path",
            "name": "Transition Path: Reveal Card",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "TransitionPath",
        },
        {
            "id": "strategy_optimization",
            "name": "Strategy Optimization",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "StrategyOptimization",
        },
    ]

    def __init__(
        self,
        num_waste=10,
        circular=False,
        open=False,
        question_type: int | None = None,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_waste = num_waste
        self.circular = circular
        self.open = open
        self._question_type_param = question_type
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}
        self._card_images = None

        # Game state
        self.deck = []
        self.stock = None
        self.foundations = []
        self.waste = []

        # Standard QA variables
        self._question_type_idx: int = 0
        self._question: str = ""
        self._options: list[str] | None = None
        self._oracle_answer: str = ""

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        return build_description(
            game_name="Spider Solitaire",
            rules=GAME_RULES,
            question=self._question,
            options=self._options,
            oracle_answer=self._oracle_answer,
        )

    def _initialize_game(self) -> None:
        """Initialize the game state with deck, piles, and initial layout."""
        # Create 104 cards (8 decks of spades)
        self.deck = []
        code = 0
        for _ in range(8):
            for rank in ALLRANKS:
                self.deck.append(Card(rank, "spade", "blue", code))
                code += 1

        # Set circular mode
        Card.circular = self.circular

        # Initialize piles
        random.shuffle(self.deck)
        self.stock = OneWayStack(False)  # Face-down
        self.foundations = [OneWayStack(True) for _ in range(8)]  # Face-up
        self.waste = [SelectableStack() for _ in range(self.num_waste)]

        # Deal initial layout
        self.stock.extend(self.deck)
        self._deal_initial_layout()

    def _deal_initial_layout(self) -> None:
        """Deal the initial layout of cards to waste piles."""
        # Deal face-down cards
        total_face_down = max(104 - self.num_waste * 6, 0)

        for n in range(total_face_down):
            if not self.stock:
                break
            card = self.stock.pop()
            self.waste[n % self.num_waste].add(card, face_up=self.open)

        # Deal face-up cards
        for n in range(self.num_waste):
            if not self.stock:
                break
            card = self.stock.pop()
            self.waste[n].add(card, True)

    def _load_card_images(self) -> None:
        """Load card images from the assets directory."""
        if self._card_images is not None:
            return  # Already loaded

        self._card_images = {}
        # Use importlib.resources for reliable asset path resolution
        cards_dir = (
            resources.files("gym_v.envs")
            / "assets"
            / "gamerl"
            / "spider_solitaire"
            / "cards"
        )

        # If cards directory doesn't exist, log a warning and continue without images
        if not cards_dir.exists():
            logger.warning(f"Card images directory not found: {cards_dir}")
            logger.warning("Spider Solitaire will render without card images")
            return

        # Load card back
        blue_back_path = cards_dir / "blueBackVert.gif"
        if blue_back_path.exists():
            self._card_images["blue"] = Image.open(blue_back_path).convert("RGBA")

        # Load card faces
        for rank in ALLRANKS:
            filename = f"spade{RANKNAMES[rank]}.gif"
            face_path = cards_dir / filename
            if face_path.exists():
                self._card_images[(rank, "spade")] = Image.open(face_path).convert(
                    "RGBA"
                )

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the game state as a PIL Image"""
        canvas_width = max(745, self.num_waste * XSPACING)
        canvas_height = 410
        img = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
        draw = ImageDraw.Draw(img)

        self._draw_stock(img, draw, canvas_width)
        self._draw_foundations(img, draw)
        self._draw_waste_piles(img, draw, canvas_width)

        return img

    def _draw_stock(self, img: Image.Image, draw: ImageDraw.Draw, canvas_width: int):
        """Draw the stock pile"""
        x, y = MARGIN, 5 * MARGIN
        stock_width = STOCK_DELTA_X * (MAX_STOCK_DISPLAY - 1) + CARDWIDTH

        # Draw rectangle outline
        stock_rect = [x, y, x + stock_width, y + CARDHEIGHT]
        draw.rectangle(stock_rect, outline=OUTLINE)

        # Calculate number of cards to display
        n = len(self.stock)
        deals_left = n // self.num_waste
        display_n = min(deals_left, MAX_STOCK_DISPLAY)

        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            font = ImageFont.load_default()

        if display_n > 0:
            for i in range(display_n):
                pos_x = x + i * STOCK_DELTA_X
                pos_y = y
                if "blue" in self._card_images:
                    img.paste(
                        self._card_images["blue"],
                        (pos_x, pos_y),
                        self._card_images["blue"],
                    )
                else:
                    # Draw a simple rectangle if no image available
                    card_rect = [pos_x, pos_y, pos_x + CARDWIDTH, pos_y + CARDHEIGHT]
                    draw.rectangle(card_rect, fill=(0, 0, 139), outline=(255, 255, 255))

            remaining_deals = deals_left - display_n
            if remaining_deals > 0:
                text = f"+{remaining_deals}"
                text_pos = (
                    x + display_n * STOCK_DELTA_X + 10,
                    y + CARDHEIGHT // 2 - 10,
                )
                draw.text(text_pos, text, fill=(255, 255, 255), font=font)
        else:
            text = "Empty"
            text_pos = (x + CARDWIDTH // 2 - 20, y + CARDHEIGHT // 2 - 10)
            draw.text(text_pos, text, fill=(255, 255, 255), font=font)

    def _draw_foundations(self, img: Image.Image, draw: ImageDraw.Draw):
        """Draw the foundation piles"""
        x, y = MARGIN + XSPACING, 5 * MARGIN

        for k in range(8):
            foundation_rect = [x, y, x + CARDWIDTH, y + CARDHEIGHT]
            draw.rectangle(foundation_rect, outline=OUTLINE)

            # Draw top card if foundation is not empty
            if len(self.foundations[k]) > 0:
                card = self.foundations[k][-1]
                img_key = (card.rank, card.suit)
                if img_key in self._card_images:
                    img.paste(
                        self._card_images[img_key], (x, y), self._card_images[img_key]
                    )
                else:
                    # Draw a simple card representation if no image available
                    card_rect = [x, y, x + CARDWIDTH, y + CARDHEIGHT]
                    draw.rectangle(card_rect, fill=(255, 255, 255), outline=(0, 0, 0))
                    # Draw card rank
                    try:
                        font = ImageFont.truetype("arial.ttf", 12)
                    except OSError:
                        font = ImageFont.load_default()
                    text = RANKNAMES[card.rank]
                    draw.text((x + 5, y + 5), text, fill=(0, 0, 0), font=font)

            x += XSPACING

    def _draw_waste_piles(
        self, img: Image.Image, draw: ImageDraw.Draw, canvas_width: int
    ):
        """Draw the waste piles"""
        # Center the waste piles
        total_width = self.num_waste * XSPACING
        start_x = (canvas_width - total_width) // 2

        x, y = start_x, 5 * MARGIN + YSPACING

        for k in range(self.num_waste):
            # Draw rectangle outline
            waste_rect = [x, y, x + CARDWIDTH, y + CARDHEIGHT]
            draw.rectangle(waste_rect, outline=OUTLINE)

            # Draw cards in the pile
            current_y = y
            for card in self.waste[k]:
                if card.faceUp():
                    img_key = (card.rank, card.suit)
                else:
                    img_key = "blue"

                if img_key in self._card_images:
                    img.paste(
                        self._card_images[img_key],
                        (x, current_y),
                        self._card_images[img_key],
                    )
                else:
                    # Draw a simple card representation if no image available
                    if card.faceUp():
                        # Face-up card: white with rank
                        card_rect = [
                            x,
                            current_y,
                            x + CARDWIDTH,
                            current_y + CARDHEIGHT,
                        ]
                        draw.rectangle(
                            card_rect, fill=(255, 255, 255), outline=(0, 0, 0)
                        )
                        try:
                            font = ImageFont.truetype("arial.ttf", 12)
                        except OSError:
                            font = ImageFont.load_default()
                        text = RANKNAMES[card.rank]
                        draw.text(
                            (x + 5, current_y + 5), text, fill=(0, 0, 0), font=font
                        )
                    else:
                        # Face-down card: blue
                        card_rect = [
                            x,
                            current_y,
                            x + CARDWIDTH,
                            current_y + CARDHEIGHT,
                        ]
                        draw.rectangle(
                            card_rect, fill=(0, 0, 139), outline=(255, 255, 255)
                        )

                # Update y position
                if card.faceUp():
                    current_y += OFFSET2
                else:
                    current_y += OFFSET1

            x += XSPACING

    # ========================================================================
    # Question Generation Methods
    # ========================================================================

    def _dealsLeft(self) -> int:
        """Returns the number of times the stockpile can still deal cards."""
        return len(self.stock) // self.num_waste

    def _downUp(self, pile_index: int) -> tuple[int, int]:
        """Returns (down_count, up_count) for a waste pile."""
        pile = self.waste[pile_index]
        down = sum(1 for card in pile if card.faceDown())
        up = len(pile) - down
        return down, up

    def _downCards(self) -> int:
        """Returns total number of face-down cards across all waste piles."""
        return sum(self._downUp(k)[0] for k in range(self.num_waste))

    def _is_complete_sequence(self, seq: list[Card]) -> bool:
        """Check if the sequence is a complete K→A descending sequence of the same suit."""
        if len(seq) != 13:
            return False
        expected_rank = KING
        suit = seq[0].suit
        for card in seq:
            if card.rank != expected_rank or card.suit != suit:
                return False
            expected_rank -= 1
        return True

    def _find_longest_same_suit_sequence(self) -> tuple[int, int, int] | None:
        """
        Finds the move that forms the longest descending sequence of the same suit.
        Returns (source_pile_index, destination_pile_index, card_index) or None.
        """
        longest_sequence_length = 0
        best_move = None

        for src_index, src_pile in enumerate(self.waste):
            for card_idx in range(len(src_pile)):
                sequence = src_pile[card_idx:]
                if not Card.isDescending(sequence):
                    continue

                # Check if all cards in the sequence are of the same suit
                suit = sequence[0].suit
                if any(card.suit != suit for card in sequence):
                    continue

                # Calculate the length of the sequence
                seq_length = len(sequence)

                # Update if this sequence is longer
                if seq_length > longest_sequence_length:
                    # Find a destination pile where the top card is one higher in rank
                    for dest_index, dest_pile in enumerate(self.waste):
                        if dest_index == src_index:
                            continue
                        if dest_pile.isEmpty() or (
                            dest_pile[-1].rank - sequence[0].rank == 1
                        ):
                            if seq_length > longest_sequence_length:
                                longest_sequence_length = seq_length
                                best_move = (src_index, dest_index, card_idx)
                            break

        return best_move

    def _generate_question_type_0(self) -> dict:
        """How many times can the stockpile still deal cards?"""
        answer = str(self._dealsLeft())
        question = f"{GAME_RULES}\n\n**Question:** How many times can the stockpile still deal cards?"
        analysis = (
            f"We can see that the stockpile has {self._dealsLeft()} stacks of overlapping cards. "
            f"By counting the number of overlapping cards in the stockpile, we know that the stockpile can now be dealt {answer} times"
        )
        return {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "options": None,
        }

    def _generate_question_type_1(self) -> dict:
        """Which card is on the top of waste pile X?"""
        pile_num = random.randint(0, self.num_waste - 1)

        if not self.waste[pile_num].isEmpty():
            top_card = self.waste[pile_num][-1]
            answer = f"{top_card.suit.capitalize()} {RANKNAMES[top_card.rank]}"
            analysis = (
                f"By checking the top card on the {pile_num + 1}-th waste pile, we can know its rank and suit. "
                f"So the top card of waste pile {pile_num} is the {RANKNAMES[top_card.rank]} of {top_card.suit.capitalize()}."
            )
        else:
            answer = "Empty"
            analysis = f"Waste pile {pile_num} is currently empty."

        question = f"{GAME_RULES}\n\n**Question:** Which card is on the top of waste pile {pile_num}?"
        return {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "options": None,
        }

    def _generate_question_type_2(self) -> dict:
        """How many face-down cards are currently in all waste piles?"""
        answer = str(self._downCards())

        # Count face-down cards in each pile
        pile_counts = ", ".join(
            [
                f"waste pile {k} has {self._downUp(k)[0]} face-down cards"
                for k in range(self.num_waste)
            ]
        )

        analysis = (
            f"By counting the face-down cards of each waste pile, we find that {pile_counts}. "
            f"Therefore, there are a total of {self._downCards()} face-down cards across all waste piles."
        )

        question = f"{GAME_RULES}\n\n**Question:** How many face-down cards are currently in all waste piles?"
        return {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "options": None,
        }

    def _generate_question_type_3(self) -> dict:
        """If I click the stockpile for X times, how many face-up cards will be in waste pile Y?"""
        num1 = random.randint(0, 5)
        pile_num = random.randint(0, self.num_waste - 1)

        # Calculate how many times we can actually deal
        possible_deals = min(num1, self._dealsLeft())
        current_face_up = self._downUp(pile_num)[1]
        additional_face_up = possible_deals
        new_face_up = current_face_up + additional_face_up
        answer = str(new_face_up)

        analysis = (
            f"Dealing {num1} time(s) would add {additional_face_up} face-up card(s) to waste pile {pile_num}. "
            f"Currently, there are {current_face_up} face-up card(s) in this pile. "
            f"Therefore, after clicking the stock pile, there would be {new_face_up} face-up card(s) in waste pile {pile_num}."
        )

        question = f"{GAME_RULES}\n\n**Question:** If I click the stockpile for {num1} times, how many face-up cards will be in waste pile {pile_num}?"
        return {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "options": None,
        }

    def _generate_question_type_4(self) -> dict:
        """What will happen if I want to move card X from pile Y to pile Z? (MCQ)"""
        # Select source pile
        source_pile_index = random.randint(0, self.num_waste - 1)
        source_pile = self.waste[source_pile_index]

        # Determine whether to select a face-down card (20% probability)
        if random.random() < 0.2:
            face_down_indices = [
                i for i, card in enumerate(source_pile) if card.faceDown()
            ]
            if not face_down_indices:
                face_up_indices = [
                    i for i, card in enumerate(source_pile) if card.faceUp()
                ]
                card_index = random.choice(face_up_indices) if face_up_indices else -1
            else:
                card_index = random.choice(face_down_indices)
        else:
            face_up_indices = [i for i, card in enumerate(source_pile) if card.faceUp()]
            card_index = random.choice(face_up_indices) if face_up_indices else -1

        # Select destination pile
        if random.random() < 0.75:
            # 75% chance to select a potentially valid destination
            if card_index != -1 and card_index < len(source_pile):
                card_to_move = source_pile[card_index]
                possible_destinations = []
                for idx, pile in enumerate(self.waste):
                    if idx == source_pile_index:
                        continue
                    if pile.isEmpty() or (pile[-1].rank - card_to_move.rank == 1):
                        possible_destinations.append(idx)

                if possible_destinations:
                    destination_pile_index = random.choice(possible_destinations)
                else:
                    destination_pile_index = random.randint(0, self.num_waste - 1)
                    while destination_pile_index == source_pile_index:
                        destination_pile_index = random.randint(0, self.num_waste - 1)
            else:
                destination_pile_index = random.randint(0, self.num_waste - 1)
                while destination_pile_index == source_pile_index:
                    destination_pile_index = random.randint(0, self.num_waste - 1)
        else:
            # 25% chance to select completely random destination
            destination_pile_index = random.randint(0, self.num_waste - 1)
            while destination_pile_index == source_pile_index:
                destination_pile_index = random.randint(0, self.num_waste - 1)

        # Format the question
        question = f"{GAME_RULES}\n\n**Question:** What will happen if I want to move the number {card_index} card of pile {source_pile_index} to pile {destination_pile_index}?"

        # Standard note text
        note_text = (
            f"Note: The number {card_index} card in pile {source_pile_index} is the {card_index + 1}-th card from the bottom. "
            f"Source pile {source_pile_index} is the {source_pile_index + 1}-th pile from the left, and "
            f"destination pile {destination_pile_index} is the {destination_pile_index + 1}-th pile from the left."
        )

        # Determine the correct option
        dest_pile = self.waste[destination_pile_index]

        if card_index == -1 or card_index >= len(source_pile):
            # Invalid card index
            correct_option = "E"
            analysis = f"The specified card does not exist in the source pile. This could be due to selecting an index out of range. {note_text}"
        else:
            card_to_move = source_pile[card_index]

            if card_to_move.faceDown():
                correct_option = "B"
                analysis = (
                    f"The move cannot be made because the selected card is face-down and its value is unknown. "
                    f"In Spider Solitaire, only face-up cards can be moved since their values are visible. {note_text}"
                )
            else:
                # Check cards above
                if card_index < len(source_pile) - 1:
                    cards_to_check = source_pile[card_index:]
                    has_face_down_above = any(
                        card.faceDown() for card in cards_to_check[1:]
                    )

                    # Check descending sequence
                    is_descending = True
                    for i in range(len(cards_to_check) - 1):
                        current_card = cards_to_check[i]
                        next_card = cards_to_check[i + 1]
                        if next_card.rank != current_card.rank - 1:
                            is_descending = False
                            break

                    if has_face_down_above:
                        correct_option = "B"
                        analysis = (
                            f"The move cannot be made because there are face-down cards above the selected card. "
                            f"All cards above the selected card must be face-up to move the sequence. {note_text}"
                        )
                    elif not is_descending:
                        correct_option = "C"
                        analysis = (
                            f"The move cannot be made because the cards above the selected card "
                            f"do not form a valid descending sequence. {note_text}"
                        )
                    else:
                        # Check destination compatibility
                        if dest_pile.isEmpty():
                            correct_option = "A"
                            analysis = (
                                f"Moving the {RANKNAMES[card_to_move.rank]} of {card_to_move.suit.capitalize()} "
                                f"and cards above it from pile {source_pile_index} to the empty pile {destination_pile_index} "
                                f"is successful. {note_text}"
                            )
                        else:
                            dest_top_card = dest_pile[-1]
                            if dest_top_card.rank - card_to_move.rank == 1:
                                correct_option = "A"
                                analysis = (
                                    f"Moving the {RANKNAMES[card_to_move.rank]} of {card_to_move.suit.capitalize()} "
                                    f"and cards above it from pile {source_pile_index} to pile {destination_pile_index} "
                                    f"is successful as it forms a valid descending sequence. {note_text}"
                                )
                            else:
                                correct_option = "D"
                                analysis = (
                                    f"The move cannot be made because the top card of the target pile {destination_pile_index} "
                                    f"does not have a rank equal to this card's rank plus one. {note_text}"
                                )
                else:
                    # Moving a single card
                    if dest_pile.isEmpty():
                        correct_option = "A"
                        analysis = (
                            f"Moving the {RANKNAMES[card_to_move.rank]} of {card_to_move.suit.capitalize()} "
                            f"from pile {source_pile_index} to the empty pile {destination_pile_index} "
                            f"is successful. {note_text}"
                        )
                    else:
                        dest_top_card = dest_pile[-1]
                        if dest_top_card.rank - card_to_move.rank == 1:
                            correct_option = "A"
                            analysis = (
                                f"Moving the {RANKNAMES[card_to_move.rank]} of {card_to_move.suit.capitalize()} "
                                f"from pile {source_pile_index} to pile {destination_pile_index} "
                                f"is successful as it forms a valid descending sequence. {note_text}"
                            )
                        else:
                            correct_option = "D"
                            analysis = (
                                f"The move cannot be made because the top card of the target pile {destination_pile_index} "
                                f"does not have a rank equal to this card's rank plus one. {note_text}"
                            )

        # Define options
        options = [
            "A. The move will be successful, and the cards will be in descending order, following the rules of movement.",
            "B. The move cannot be made because this card is face-down and its value is unknown.",
            "C. The move cannot be made because there is a card above it, and that card does not form a descending order with the selected card.",
            "D. The move cannot be made because the top card of the target pile does not have a rank equal to this card's rank plus one.",
            "E. The move cannot be made because the pile has too few cards, and this card does not exist.",
        ]

        question += "\n\n**Options:**\n" + "\n".join(options)
        answer = correct_option

        return {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "options": options,
        }

    def _generate_question_type_5(self) -> dict:
        """What should I do to reveal the first face-down card in waste pile X? (MCQ)"""
        # Select a waste pile
        candidate_piles = [
            i
            for i, pile in enumerate(self.waste)
            if len(pile) == 1 and pile[-1].faceDown()
        ]

        if candidate_piles:
            waste_pile_num = random.choice(candidate_piles)
            correct_option = "A"
            analysis = (
                f"The first face-down card in waste pile {waste_pile_num} is already at the top and should be face up. "
                f"No action is needed as there are no face-down cards above it."
            )
            options = [
                "A. No action is needed; there are no face-down cards in this pile.",
                "B. There is no immediate way to reveal it; we should move cards from other piles first and wait for more information.",
            ] + [
                f"{chr(67 + i)}. We should move the {random.randint(0, self.num_waste - 1)}-th card of pile {random.randint(0, self.num_waste - 1)} to pile {random.randint(0, self.num_waste - 1)}."
                for i in range(6)
            ]
        else:
            waste_pile_num = random.randint(0, self.num_waste - 1)
            waste_pile = self.waste[waste_pile_num]

            has_face_down = any(card.faceDown() for card in waste_pile)

            if has_face_down:
                selected_top_card = waste_pile[-1]
                possible_targets = [
                    idx
                    for idx, pile in enumerate(self.waste)
                    if idx != waste_pile_num
                    and (
                        pile.isEmpty() or (pile[-1].rank - selected_top_card.rank == 1)
                    )
                ]

                if possible_targets:
                    correct_option = random.choice(["C", "D", "E", "F", "G", "H"])
                    correct_move_pile = random.choice(possible_targets)

                    options = [
                        "A. No action is needed; there are no face-down cards in this pile.",
                        "B. There is no immediate way to reveal it; we should move cards from other piles first and wait for more information.",
                    ] + [
                        f"{chr(67 + i)}. We should move the {random.randint(0, self.num_waste - 1)}-th card of pile {random.randint(0, self.num_waste - 1)} to pile {random.randint(0, self.num_waste - 1)}."
                        for i in range(6)
                    ]

                    correct_option_idx = ord(correct_option) - ord("C")
                    options[correct_option_idx + 2] = (
                        f"{correct_option}. We should move the card of pile {waste_pile_num} to pile {correct_move_pile}."
                    )

                    analysis = (
                        f"To reveal the first face-down card in waste pile {waste_pile_num}, you should move the top card to another pile where it can form a descending sequence. "
                        f"Moving it to pile {correct_move_pile} is a valid move, allowing the face-down card to be revealed."
                    )
                else:
                    correct_option = "B"
                    analysis = (
                        f"In waste pile {waste_pile_num}, there are face-down cards that are not at the top. "
                        f"And there is no operation to move the top cards from this waste pile to another waste pile. "
                        f"So we can't reveal the first face-down card by directly removing the above face-up cards to another waste pile. "
                        f"To reveal the first face-down card, you need to move cards from other piles first and wait for more information."
                    )

                    options = [
                        "A. No action is needed; there are no face-down cards in this pile.",
                        "B. There is no immediate way to reveal it; we should move cards from other piles first and wait for more information.",
                    ] + [
                        f"{chr(67 + i)}. We should move the {random.randint(0, self.num_waste - 1)}-th card of pile {random.randint(0, self.num_waste - 1)} to pile {random.randint(0, self.num_waste - 1)}."
                        for i in range(6)
                    ]
            else:
                correct_option = "A"
                analysis = f"In waste pile {waste_pile_num}, all the cards are face up. There are no face-down cards in waste pile {waste_pile_num}."

                options = [
                    "A. No action is needed; there are no face-down cards in this pile.",
                    "B. There is no immediate way to reveal it; we should move cards from other piles first and wait for more information.",
                ] + [
                    f"{chr(67 + i)}. We should move the {random.randint(1, 5)}-th card of pile {random.randint(0, self.num_waste - 1)} to pile {random.randint(0, self.num_waste - 1)}."
                    for i in range(6)
                ]

        question = f"{GAME_RULES}\n\n**Question:** What should I do if I want to reveal the first face-down card in waste pile {waste_pile_num}?"
        question += "\n\n**Options:**\n" + "\n".join(options)
        answer = correct_option

        return {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "options": options,
        }

    def _generate_question_type_6(self) -> dict:
        """Based on the current board state, what is the optimal strategy? (MCQ)"""
        # Priority 1: Move complete sequences to foundation
        can_move_complete_sequences = False
        target_foundation_index = 0

        for _f_index, foundation in enumerate(self.foundations):
            if len(foundation) == 13:
                continue
            for w_index, waste_pile in enumerate(self.waste):
                if len(waste_pile) >= 13:
                    sequence = waste_pile[-13:]
                    if self._is_complete_sequence(sequence):
                        can_move_complete_sequences = True
                        target_foundation_index = w_index
                        break
            if can_move_complete_sequences:
                break

        # Priority 2: Form longest same-suit sequence
        can_form_descending_same_suit = False
        best_move = None
        if not can_move_complete_sequences:
            best_move = self._find_longest_same_suit_sequence()
            if best_move:
                can_form_descending_same_suit = True

        # Priority 3: Utilize empty piles
        can_utilize_empty_piles = False
        if not can_move_complete_sequences and not can_form_descending_same_suit:
            can_utilize_empty_piles = any(pile.isEmpty() for pile in self.waste)

        # Priority 4: Deal from stock
        can_deal_stock = False
        if (
            not can_move_complete_sequences
            and not can_form_descending_same_suit
            and not can_utilize_empty_piles
        ):
            can_deal_stock = (
                all(not pile.isEmpty() for pile in self.waste) and len(self.stock) > 0
            )

        # Generate question based on priority
        if can_move_complete_sequences:
            correct_option = "H"
            analysis = "There are complete sequences available to move to the foundation piles. Moving them will help progress towards winning the game."

            options = [
                f"{chr(65 + i)}. We should move the {random.randint(1, 5)}-th card of pile {random.randint(0, self.num_waste - 1)} to pile {random.randint(0, self.num_waste - 1)}."
                for i in range(6)
            ] + [
                "G. No cards can be moved; we should click the stockpile to deal cards.",
                f"H. We should move cards from pile {target_foundation_index} to the foundation piles.",
            ]

        elif can_form_descending_same_suit:
            source_pile_index, dest_pile_index, card_idx = best_move
            correct_option_choice = random.choice(["A", "B", "C", "D", "E", "F"])
            correct_option = correct_option_choice

            analysis = (
                f"The optimal strategy is to form a descending sequence of the same suit to maximize potential moves. "
                f"Moving {card_idx} card(s) from pile {source_pile_index} to pile {dest_pile_index} forms the longest possible descending sequence of the same suit. "
                f"So this move is optimal because it forms a sequence longer than any other moves."
            )

            options = [
                f"{chr(65 + i)}. We should move the {random.randint(1, 5)}-th card of pile {random.randint(0, self.num_waste - 1)} to pile {random.randint(0, self.num_waste - 1)}."
                for i in range(6)
            ] + [
                "G. No cards can be moved; we should click the stockpile to deal cards.",
                f"H. We should move cards from pile {random.randint(0, self.num_waste - 1)} to the foundation piles.",
            ]

            correct_option_idx = ord(correct_option) - ord("A")
            options[correct_option_idx] = (
                f"{correct_option}. We should move the {card_idx}-th card of pile {source_pile_index} to pile {dest_pile_index}."
            )

        elif can_utilize_empty_piles:
            correct_option_choice = random.choice(["A", "B", "C", "D", "E", "F"])
            correct_option = correct_option_choice

            empty_piles = [i for i, pile in enumerate(self.waste) if pile.isEmpty()]
            target_pile = random.choice(empty_piles)
            source_pile = random.randint(0, self.num_waste - 1)
            while len(self.waste[source_pile]) == 0:
                source_pile = random.randint(0, self.num_waste - 1)
            num_cards = len(self.waste[source_pile])

            analysis = f"There is an empty pile 'pile {target_pile}' from which we can move our cards to. Utilizing empty waste piles provides more flexibility in organizing cards and can help in creating more opportunities for valid moves."

            options = [
                f"{chr(65 + i)}. We should move the {random.randint(1, 5)}-th card of pile {random.randint(0, self.num_waste - 1)} to pile {random.randint(0, self.num_waste - 1)}."
                for i in range(6)
            ] + [
                "G. No cards can be moved; we should click the stockpile to deal cards.",
                f"H. We should move cards from pile {random.randint(0, self.num_waste - 1)} to the foundation piles.",
            ]

            correct_option_idx = ord(correct_option) - ord("A")
            options[correct_option_idx] = (
                f"{correct_option}. We should move the {num_cards}-th card of pile {source_pile} to pile {target_pile}."
            )

        elif can_deal_stock:
            correct_option = "G"
            analysis = "In the current game, no move can form a descending card order. So no immediate moves are available. Dealing cards from the stockpile will uncover new cards and create new opportunities for moves."

            options = [
                f"{chr(65 + i)}. We should move the {random.randint(1, 5)}-th card of pile {random.randint(0, self.num_waste - 1)} to pile {random.randint(0, self.num_waste - 1)}."
                for i in range(6)
            ] + [
                "G. No cards can be moved; we should click the stockpile to deal cards.",
                f"H. We should move cards from pile {random.randint(0, self.num_waste - 1)} to the foundation piles.",
            ]

        else:
            correct_option = "G"
            analysis = (
                "The current game state does not clearly indicate an optimal strategy."
            )

            options = [
                f"{chr(65 + i)}. We should move the {random.randint(1, 5)}-th card of pile {random.randint(0, self.num_waste - 1)} to pile {random.randint(0, self.num_waste - 1)}."
                for i in range(6)
            ] + [
                "G. No cards can be moved; we should click the stockpile to deal cards.",
                f"H. We should move cards from pile {random.randint(0, self.num_waste - 1)} to the foundation piles.",
            ]

        question = f"{GAME_RULES}\n\n**Question:** Based on the current board state, what is the optimal strategy we should adopt?"
        question += "\n\n**Options:**\n" + "\n".join(options)
        answer = correct_option

        return {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "options": options,
        }

    def _generate_question(self, question_type: int) -> None:
        """Generate a question of the specified type - sets _question, _options, _oracle_answer"""
        if question_type == 0:
            result = self._generate_question_type_0()
        elif question_type == 1:
            result = self._generate_question_type_1()
        elif question_type == 2:
            result = self._generate_question_type_2()
        elif question_type == 3:
            result = self._generate_question_type_3()
        elif question_type == 4:
            result = self._generate_question_type_4()
        elif question_type == 5:
            result = self._generate_question_type_5()
        elif question_type == 6:
            result = self._generate_question_type_6()
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        # Extract to instance variables
        self._question = result["question"]
        self._options = result.get("options")
        self._oracle_answer = result["answer"]

    # ========================================================================
    # Gym-v Interface
    # ========================================================================

    def _get_state_text(self) -> str:
        """Generate text description of current Spider Solitaire game state.

        Returns a text representation matching the rendered image.
        """
        lines = []

        # Stock pile
        stock_count = len(self.stock) if self.stock else 0
        lines.append(f"Stock: {stock_count} cards")

        # Foundation piles (completed suits)
        completed_suits = sum(1 for f in self.foundations if len(f) > 0)
        lines.append(f"Foundations: {completed_suits} completed suits")

        # Waste piles (tableau piles)
        for i, pile in enumerate(self.waste, 1):
            if pile.isEmpty():
                lines.append(f"Pile {i}: empty")
            else:
                cards = list(pile)
                faceup = [c for c in cards if c.faceUp]
                hidden_count = len(cards) - len(faceup)
                if hidden_count > 0:
                    cards_str = f"[{hidden_count} hidden], " + ", ".join(
                        [f"{c.rank}{c.suit[0].upper()}" for c in faceup]
                    )
                else:
                    cards_str = ", ".join(
                        [f"{c.rank}{c.suit[0].upper()}" for c in faceup]
                    )
                lines.append(f"Pile {i}: {cards_str}")

        return "\n".join(lines)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """Reset the environment"""
        super().reset(seed=seed)

        logger.info(
            f"Reset Spider Solitaire QA (num_waste={self.num_waste}, circular={self.circular}, open={self.open})"
        )

        self._initialize_game()
        self._load_card_images()

        # Generate random question type or use specified
        if self._question_type_param is not None:
            self._question_type_idx = self._question_type_param
        else:
            self._question_type_idx = random.randint(0, 6)
        q_type = self.QUESTION_TYPES[self._question_type_idx]

        # Generate question - sets _question, _options, _oracle_answer
        self._generate_question(self._question_type_idx)

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
        """Execute an action (answer the question)"""
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]

        reward = self._score_answer(action_str)
        correct = reward == 1.0

        if correct:
            response = "Correct!"
        else:
            response = f"Incorrect. The correct answer is: {self._oracle_answer}"

        obs = Observation(image=self.render(), text=response)

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

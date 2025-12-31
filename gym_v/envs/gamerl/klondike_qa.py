"""Klondike Solitaire QA environment for gym-v.

Klondike is the classic solitaire card game with:
- Stock pile (draw pile) - 24 cards initially
- Waste pile - cards drawn from stock
- 4 Foundation piles - build up from Ace to King by suit
- 7 Tableau piles - build down in alternating colors

This environment provides 5 question types about game moves and strategy.
"""

from __future__ import annotations

import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation

# Game Rules
KLONDIKE_RULES = """The given image represents the interface of the game Klondike Solitaire. The user interface consists of a board with 52 playing cards divided into four distinct areas:

1. **Stock Pile (Draw Pile):** Initially composed of 24 face-down cards. The player can draw one card at a time to reveal its face.

2. **Waste Pile (Dump Pile):** This pile holds the cards drawn from the Stock Pile that have not been moved to other areas. Only the topmost card in the Waste Pile is available for play.

3. **Foundation Piles:** These four piles are designated for each suit (hearts, diamonds, clubs, and spades, but not necessarily following this order). From left to right, they are referred to as foundation 1 through foundation 4. Players must build up the foundation starting with the Ace and then place cards in ascending order (2 through King) of the same suit.

4. **Tableau Piles:** There are seven tableau piles. From left to right, these piles are referred to as Tab 1 through Tab 7, and initially contain an increasing number of cards from 1 to 7. Only the topmost cards in each pile are face-up and built in descending order, alternating colors (red and black suits). Only when the topmost cards are removed to some other place (e.g. another tableau pile or the foundation pile) will the hidden card beneath be revealed. Only a King can be placed on an empty tableau pile unless it starts there at the beginning of the game.

**Objective:**
The goal of Klondike Solitaire is to move all cards to the Foundation Piles, organized by suit in ascending order from Ace to King."""


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
                                f"Move from Tab{from_idx+1} to Tab{to_idx+1}"
                            )

        # Waste to Tableau
        if self.waste:
            waste_card = self.waste[-1]
            for tab_idx in range(7):
                if self.can_move_to_tableau(waste_card, self.tableau[tab_idx]):
                    valid_moves.append(f"Move from Waste Pile to Tab{tab_idx+1}")

        # Waste to Foundation
        if self.waste:
            waste_card = self.waste[-1]
            for found_idx in range(4):
                if self.can_move_to_foundation(waste_card, self.foundation[found_idx]):
                    valid_moves.append(
                        f"Move from Waste Pile to Foundation {found_idx+1}"
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
                            f"Move from Tab{tab_idx+1} to Foundation {found_idx+1}"
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


def render_klondike_board(game: KlondikeGame) -> Image.Image:
    """Render Klondike board using PIL."""
    width = 800
    height = 600
    img = Image.new("RGB", (width, height), (0, 100, 0))  # Green felt
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    card_width = 70
    card_height = 100
    padding = 10

    # Draw Stock pile
    stock_x, stock_y = 20, 20
    if game.stock:
        draw.rectangle(
            [stock_x, stock_y, stock_x + card_width, stock_y + card_height],
            fill="blue",
            outline="white",
            width=2,
        )
        draw.text(
            (stock_x + 10, stock_y + 40), f"{len(game.stock)}", fill="white", font=font
        )
    else:
        draw.rectangle(
            [stock_x, stock_y, stock_x + card_width, stock_y + card_height],
            outline="white",
            width=2,
        )
    draw.text(
        (stock_x, stock_y + card_height + 5), "Stock", fill="white", font=font_small
    )

    # Draw Waste pile
    waste_x = stock_x + card_width + padding
    if game.waste:
        card = game.waste[-1]
        color = "red" if card.color == "red" else "black"
        draw.rectangle(
            [waste_x, stock_y, waste_x + card_width, stock_y + card_height],
            fill="white",
            outline="black",
            width=2,
        )
        draw.text(
            (waste_x + 10, stock_y + 40),
            f"{card.suit}\n{card.rank}",
            fill=color,
            font=font,
        )
    else:
        draw.rectangle(
            [waste_x, stock_y, waste_x + card_width, stock_y + card_height],
            outline="white",
            width=2,
        )
    draw.text(
        (waste_x, stock_y + card_height + 5), "Waste", fill="white", font=font_small
    )

    # Draw Foundation piles
    found_x_start = waste_x + card_width + padding * 3
    for i in range(4):
        found_x = found_x_start + i * (card_width + padding)
        if game.foundation[i]:
            card = game.foundation[i][-1]
            color = "red" if card.color == "red" else "black"
            draw.rectangle(
                [found_x, stock_y, found_x + card_width, stock_y + card_height],
                fill="white",
                outline="black",
                width=2,
            )
            draw.text(
                (found_x + 10, stock_y + 40),
                f"{card.suit}\n{card.rank}",
                fill=color,
                font=font,
            )
        else:
            draw.rectangle(
                [found_x, stock_y, found_x + card_width, stock_y + card_height],
                outline="white",
                width=2,
            )
            draw.text(
                (found_x + 20, stock_y + 40), f"F{i+1}", fill="white", font=font_small
            )

    # Draw Tableau piles
    tab_y = stock_y + card_height + 50
    for i in range(7):
        tab_x = 20 + i * (card_width + padding)

        # Draw label
        draw.text((tab_x, tab_y - 20), f"Tab{i+1}", fill="white", font=font_small)

        if game.tableau[i]:
            y_offset = 0
            for _, card in enumerate(game.tableau[i]):
                card_y = tab_y + y_offset

                if card.faceup:
                    color = "red" if card.color == "red" else "black"
                    draw.rectangle(
                        [tab_x, card_y, tab_x + card_width, card_y + card_height],
                        fill="white",
                        outline="black",
                        width=2,
                    )
                    draw.text(
                        (tab_x + 10, card_y + 40),
                        f"{card.suit}\n{card.rank}",
                        fill=color,
                        font=font,
                    )
                else:
                    draw.rectangle(
                        [tab_x, card_y, tab_x + card_width, card_y + card_height],
                        fill="blue",
                        outline="white",
                        width=2,
                    )

                y_offset += 25  # Overlap cards
        else:
            # Empty tableau
            draw.rectangle(
                [tab_x, tab_y, tab_x + card_width, tab_y + card_height],
                outline="white",
                width=2,
            )

    return img


# ============================================================================
# Klondike QA Environment
# ============================================================================


class GameRLKlondikeQAEnv(Env):
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

    def __init__(self, question_type: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self._question_type = question_type
        self._current_question = None
        self._game = None

    @property
    def description(self) -> str:
        return f"Klondike Solitaire QA\n\n{KLONDIKE_RULES}"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
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
        q_type = (
            self._question_type
            if self._question_type is not None
            else random.randint(0, 4)
        )

        # Generate question
        if q_type == 0:
            self._current_question = self._generate_board_state_question()
        elif q_type == 1:
            self._current_question = self._generate_deadlock_question()
        elif q_type == 2:
            self._current_question = self._generate_effectiveness_question()
        elif q_type == 3:
            self._current_question = self._generate_validity_question()
        elif q_type == 4:
            self._current_question = self._generate_foundation_question()

        # Render
        img = render_klondike_board(self._game)
        obs = Observation(image=img, text=self._current_question["question"])
        return obs, {}

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        # Normalize answer
        answer_normalized = action.strip().lower()
        correct_answer = str(self._current_question["answer"]).strip().lower()

        # Check if correct
        correct = answer_normalized == correct_answer
        reward = 1.0 if correct else 0.0

        # Generate response
        if correct:
            response = "Correct!"
        else:
            response = f"Incorrect. The correct answer is: {self._current_question['answer']}\n\n{self._current_question['analysis']}"

        # Re-render
        img = render_klondike_board(self._game)
        obs = Observation(image=img, text=response)
        return obs, reward, True, False, {}

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

        # Build question
        question = f"{KLONDIKE_RULES}\n\nWhich of the following moves is valid?\n"
        for i, opt in enumerate(options):
            question += f"{i+1}. {opt}\n"

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
            "analysis": analysis,
        }

    def _generate_deadlock_question(self) -> dict:
        """Type 1: Is the game in a deadlock?"""
        is_deadlock = self._game.is_deadlock()

        question = f"{KLONDIKE_RULES}\n\nIs the current game state in a deadlock?\n1. Yes\n2. No"

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

        return {"question": question, "answer": answer, "analysis": analysis}

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
                else f"Move from Tab{random.randint(1,7)} to Tab{random.randint(1,7)}"
            )
            if other_move not in options:
                options.append(other_move)

        random.shuffle(options)
        answer = str(options.index(best_move) + 1)

        question = f"{KLONDIKE_RULES}\n\nWhich move is most effective?\n"
        for i, opt in enumerate(options):
            question += f"{i+1}. {opt}\n"

        analysis = f"The most effective move is: {best_move}\n"
        if "Foundation" in best_move:
            analysis += "Foundation moves are always effective as they progress toward the goal."
        else:
            analysis += "This move helps reveal hidden cards or creates strategic opportunities."

        return {"question": question, "answer": answer, "analysis": analysis}

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

        question = f"{KLONDIKE_RULES}\n\nIs the following move valid?\n{test_move}\n\n1. Yes\n2. No"

        answer = "1" if is_valid else "2"

        analysis = f"The move '{test_move}' is {'valid' if is_valid else 'invalid'}.\n"
        if is_valid:
            analysis += "This move follows the game rules."
        else:
            analysis += "This move violates the game rules."

        return {"question": question, "answer": answer, "analysis": analysis}

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
                            card_desc = f"{card.suit} {card.rank} from Tab{tab_idx+1}"
                            break
                if can_move:
                    break

        question = f"{KLONDIKE_RULES}\n\nCan any card be moved to a foundation pile?\n1. Yes\n2. No"

        answer = "1" if can_move else "2"

        analysis = ""
        if can_move:
            analysis = f"Yes, {card_desc} can be moved to a foundation pile."
        else:
            analysis = "No cards can currently be moved to foundation piles."

        return {"question": question, "answer": answer, "analysis": analysis}

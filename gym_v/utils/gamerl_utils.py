"""GameRL environment utility functions."""

from __future__ import annotations

import re
from textwrap import dedent


def build_description(
    *,
    game_name: str,
    rules: str,
    question: str,
    options: list[str] | None = None,
    oracle_answer: str | None = None,
    answer_format: str | None = None,
) -> str:
    """Build a standardized description for GameRL environments.

    This function takes clean components and formats them into a standard template.
    The question should be pure question text (without embedded rules or options).
    The options should be a clean list (without prefixes like A., B., 1., 2.).

    Standard output format:
        You are playing {game_name}.

        **Rules:**
        {rules}

        **Question:**
        {question}

        **Options:**
        A. option1
        B. option2
        ...

        **Answer Format:**
        - For multiple choice: Reply with only the letter/number
        - For numbers: Reply with only the number

        Do not include any explanation or extra text.

    Args:
        game_name: Name of the game (e.g., "Tic-Tac-Toe")
        rules: Game rules text (GAME_RULES constant)
        question: Pure question text (no embedded rules/options)
        options: Clean list of options (no prefixes), or None for fill-in-blank
        oracle_answer: Expected answer (used to determine prefix format)

    Returns:
        Formatted description string
    """
    # Clean up rules (remove leading "Rules:" header if present)
    clean_rules = _strip_rules_header(rules) if rules else ""

    # Clean up question (remove any accidentally embedded content)
    clean_question = _clean_question_text(question, rules) if question else ""

    # Format options with appropriate prefix
    formatted_options = None
    if options:
        formatted_options = _format_options(options, oracle_answer)

    # Generate answer format prompt
    answer_format = (
        answer_format.strip()
        if answer_format
        else _generate_answer_format(formatted_options, oracle_answer)
    )

    # Build the description
    parts = [
        f"You are playing {game_name}.",
        "",
        "**Rules:**",
        clean_rules,
        "",
        "**Question:**",
        clean_question,
    ]

    if formatted_options:
        parts.extend(
            [
                "",
                "**Options:**",
                "\n".join(formatted_options),
            ]
        )

    parts.extend(
        [
            "",
            "**Answer Format:**",
            answer_format,
            "",
            "Do not include any explanation or extra text.",
        ]
    )

    return "\n".join(parts).strip()


def _strip_rules_header(rules: str) -> str:
    """Remove leading Rules header from rules text."""
    if not rules:
        return ""

    text = rules.strip()
    # Remove common header patterns
    patterns = [
        r"^\*\*Rules:?\*\*\s*\n?",
        r"^Rules:?\s*\n?",
        r"^Game Rules:?\s*\n?",
        r"^Key Rules:?\s*\n?",
        r"^Puzzle Rules:?\s*\n?",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


def _clean_question_text(question: str, rules: str | None = None) -> str:
    """Extract pure question text, removing any embedded rules/options/format."""
    if not question:
        return ""

    text = question.strip()

    # Remove embedded rules if present
    if rules:
        rules_text = rules.strip()
        if rules_text and rules_text in text:
            text = text.replace(rules_text, "").strip()

    # Extract content after "Question:" marker if present
    for marker in ("**Question:**", "Question:"):
        if marker in text:
            text = text.split(marker)[-1].strip()
            break

    # Remove embedded Options section
    for marker in ("**Options:**", "Options:"):
        if marker in text:
            text = text.split(marker)[0].strip()
            break

    # Remove embedded Answer Format section
    for marker in ("**Answer Format:**", "Answer Format:"):
        if marker in text:
            text = text.split(marker)[0].strip()
            break

    # Remove inline formatting instructions that belong in Answer Format.
    text = re.sub(
        r"\bPlease answer with a number\.?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(
        r"Write the answer as a list of three lists:.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    text = re.sub(
        r"Example format:.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # Remove trailing "Do not include..." instruction
    text = re.sub(
        r"\n?Do not include any explanation or extra text\.?$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Remove any option-like lines at the end (A., B., 1., 2., etc.)
    lines = text.split("\n")
    while lines and re.match(r"^(?:[A-Za-z]|\d+)[\.:)\-]\s+", lines[-1].strip()):
        lines.pop()
    text = "\n".join(lines).strip()

    return text


def _format_options(options: list[str], oracle_answer: str | None = None) -> list[str]:
    """Format options with consistent prefixes (A., B., C. or 1., 2., 3.).

    Args:
        options: List of option strings (may or may not have prefixes)
        oracle_answer: Expected answer to determine prefix type

    Returns:
        List of formatted options with consistent prefixes
    """
    if not options:
        return []

    # Determine prefix type based on oracle_answer
    use_letter_prefix = True
    if oracle_answer:
        answer_val = str(oracle_answer).strip()
        if answer_val.isdigit():
            use_letter_prefix = False
        elif len(answer_val) == 1 and answer_val.isalpha():
            use_letter_prefix = True

    formatted = []
    for i, opt in enumerate(options):
        # Remove any existing prefix
        clean_opt = _strip_option_prefix(str(opt).strip())

        # Add new prefix
        if use_letter_prefix:
            prefix = chr(ord("A") + i)
            formatted.append(f"{prefix}. {clean_opt}")
        else:
            formatted.append(f"{i + 1}. {clean_opt}")

    return formatted


def _strip_option_prefix(option: str) -> str:
    """Remove prefix from an option string (A., B., 1., 2., etc.)."""
    # Handle multiple prefixes (e.g., "A. A. text" -> "text")
    text = option.strip()
    while True:
        match = re.match(r"^[A-Za-z\d]+[\.:)\-]\s*(.*)$", text)
        if match:
            new_text = match.group(1).strip()
            if new_text and new_text != text:
                text = new_text
            else:
                break
        else:
            break
    return text


def _generate_answer_format(
    formatted_options: list[str] | None = None,
    oracle_answer: str | None = None,
) -> str:
    """Generate answer format instructions based on options and answer type."""
    # First check for multi-select answers (even when options exist)
    if oracle_answer:
        answer_val = str(oracle_answer).strip()
        # Multi-letter answer (e.g., "BD", "ABCD" for multi-select)
        if len(answer_val) > 1 and answer_val.isalpha() and answer_val.isupper():
            mc_hint = "Reply with one or more letters, e.g., A or BD or ABC"
            return f"- For multiple choice: {mc_hint}"

    # Determine if using letter or number format
    if formatted_options:
        # Check first option's prefix
        first_opt = formatted_options[0] if formatted_options else ""
        if re.match(r"^[A-Z]\.", first_opt):
            mc_hint = "Reply with only the letter (A, B, C, etc.)"
        else:
            mc_hint = "Reply with only the number (1, 2, 3, etc.)"
        return f"- For multiple choice: {mc_hint}"
    elif oracle_answer:
        answer_val = str(oracle_answer).strip()
        if answer_val.isdigit():
            return "- For numbers: Reply with only the number (1, 2, 3, etc.)"
        # Detect special answer types and provide specific format with example
        answer_type = _detect_answer_type(answer_val)
        mc_hint = _get_answer_hint(answer_type, answer_val)
        return f"- {mc_hint}"
    return "- Reply with the required format (e.g., A, (2, 3), e4, or clear 2 3)."


def _detect_answer_type(answer: str) -> str:
    """Detect the type of answer based on its format."""
    if not answer:
        return "unknown"

    # Coordinate pattern: (row, col) or (x, y, z)
    if re.match(r"^\(\s*-?\d+\s*,\s*-?\d+\s*(,\s*-?\d+\s*)?\)$", answer):
        return "coordinate"

    # PyramidChess position pattern: "[x, y] at level z"
    if re.match(
        r"^\[\s*-?\d+\s*,\s*-?\d+\s*\]\s+at\s+level\s+-?\d+$",
        answer,
        re.IGNORECASE,
    ):
        return "pyramidchess_position"

    # Ultra TicTacToe coordinate pattern: "(i, j, row, col)"
    if re.match(
        r"^\(\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\)$",
        answer,
    ):
        return "ultra_tictactoe_coord"

    # Command pattern: word followed by numbers/directions
    command_patterns = [
        r"^(clear|swap|move|push|pull|place|pick|drop)\s+",
        r"^\w+\s+\d+\s+\d+",
    ]
    for pattern in command_patterns:
        if re.match(pattern, answer, re.IGNORECASE):
            return "command"

    # Card name pattern: "Suit Rank" (e.g., "Spade 3", "Spade Jack", "Heart Ace")
    suits = {"spade", "heart", "diamond", "club"}
    ranks = {
        "ace",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "jack",
        "queen",
        "king",
    }
    parts = answer.lower().split()
    if len(parts) == 2 and parts[0] in suits and parts[1] in ranks:
        return "card"

    # Multi-letter answer pattern (e.g., "BD", "ABCD" for multi-select)
    if len(answer) > 1 and answer.isalpha() and answer.isupper():
        return "multi_letter"

    if len(answer) == 1 and answer.isalpha():
        return "single_letter"

    # Color pattern
    colors = {
        "red",
        "yellow",
        "blue",
        "green",
        "orange",
        "purple",
        "pink",
        "white",
        "black",
        "none",
    }
    if answer.lower() in colors:
        return "color"

    # Chess coordinate pattern (e.g., e4)
    if re.match(r"^[a-h][1-8]$", answer, re.IGNORECASE):
        return "chess_square"

    # Matrix/list pattern (e.g., [[0, 1, 0], [1, 1, 0], [0, 1, 1]])
    if re.match(
        r"^\[\s*\[\s*\d+\s*(,\s*\d+\s*)+\]\s*(,\s*\[\s*\d+\s*(,\s*\d+\s*)+\]\s*)+\]$",
        answer,
    ):
        return "matrix"

    # Yes/No compound pattern: "Yes, X, Y" or "No, X, Y"
    if re.match(r"^(Yes|No),\s*\d+,\s*\d+$", answer, re.IGNORECASE):
        return "compound_yes_no"

    # Sentence pattern: starts with capital, ends with punctuation or multiple words
    if len(answer.split()) > 3 or answer.endswith("."):
        return "sentence"

    # Direction pattern
    directions = {"up", "down", "left", "right", "north", "south", "east", "west"}
    if answer.lower() in directions:
        return "direction"

    return "unknown"


def _get_answer_hint(answer_type: str, example: str) -> str:
    """Get the appropriate hint text based on answer type.

    Note: We use generic examples instead of the actual answer to avoid revealing it.
    """
    # Use a format-consistent example that differs from the oracle answer.
    hints = {
        "coordinate": f"Reply with coordinates in the format (row, col) (e.g., {_coordinate_example(example)})",
        "command": f"Reply with only the command (e.g., {_command_example(example)})",
        "card": f"Reply with only the card name (e.g., {_card_example(example)})",
        "multi_letter": f"Reply with one or more letters (e.g., {_multi_letter_example(example)})",
        "single_letter": f"Reply with only the letter (e.g., {_single_letter_example(example)})",
        "color": f"Reply with only the color name (e.g., {_color_example(example)})",
        "chess_square": f"Reply with the board position (e.g., {_chess_square_example(example)})",
        "matrix": f"Reply with a list of three lists (e.g., {_matrix_example(example)})",
        "compound_yes_no": f"Reply in the format '<Yes/No>, <number>, <number>' (e.g., {_compound_yes_no_example(example)})",
        "sentence": "Reply with the complete answer as stated in the question",
        "direction": f"Reply with only the direction (e.g., {_direction_example(example)})",
        "pyramidchess_position": f"Reply with the position in the format [x, y] at level z (e.g., {_pyramidchess_example(example)})",
        "ultra_tictactoe_coord": f"Reply with coordinates as (i, j, row, col) (e.g., {_ultra_tictactoe_example(example)})",
        "unknown": "Reply with only the answer (e.g., example)",
    }
    return hints.get(answer_type, "Reply with only the answer")


def _coordinate_example(answer: str) -> str:
    match = re.match(r"^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*(,\s*(-?\d+)\s*)?\)$", answer)
    if not match:
        return "(2, 3)"
    x = int(match.group(1))
    y = int(match.group(2))
    if match.group(4) is not None:
        z = int(match.group(4))
        return f"({x + 1}, {y}, {z})"
    return f"({x + 1}, {y})"


def _command_example(answer: str) -> str:
    parts = answer.split()
    if not parts:
        return "clear 2 3"
    verb = parts[0]
    numbers = []
    rest = []
    for part in parts[1:]:
        if part.isdigit():
            numbers.append(str(int(part) + 1))
        else:
            rest.append(part)
    if verb == "swap" and rest:
        directions = ["up", "down", "left", "right"]
        rest[-1] = next(d for d in directions if d != rest[-1])
    return " ".join([verb] + numbers + rest).strip() or "clear 2 3"


def _card_example(answer: str) -> str:
    parts = answer.split()
    if len(parts) != 2:
        return "Spade 3"
    suits = ["Spade", "Heart", "Diamond", "Club"]
    ranks = [
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
    suit = next(s for s in suits if s.lower() != parts[0].lower())
    rank = next(r for r in ranks if r.lower() != parts[1].lower())
    return f"{suit} {rank}"


def _multi_letter_example(answer: str) -> str:
    example = "BD"
    if answer == example:
        example = "AC"
    return example


def _single_letter_example(answer: str) -> str:
    letters = ["A", "B", "C", "D"]
    for letter in letters:
        if letter != answer.upper():
            return letter
    return "B"


def _color_example(answer: str) -> str:
    colors = [
        "red",
        "yellow",
        "blue",
        "green",
        "orange",
        "purple",
        "pink",
        "white",
        "black",
        "none",
    ]
    for color in colors:
        if color != answer.lower():
            return color
    return "red"


def _chess_square_example(answer: str) -> str:
    files = "abcdefgh"
    ranks = "12345678"
    if not re.match(r"^[a-h][1-8]$", answer, re.IGNORECASE):
        return "e4"
    file_idx = (files.index(answer[0].lower()) + 1) % 8
    rank_idx = (ranks.index(answer[1]) + 1) % 8
    return f"{files[file_idx]}{ranks[rank_idx]}"


def _matrix_example(answer: str) -> str:
    try:
        import ast

        matrix = ast.literal_eval(answer)
        if isinstance(matrix, list) and matrix and isinstance(matrix[0], list):
            matrix = [row[:] for row in matrix]
            matrix[0][0] = 0 if matrix[0][0] else 1
            return str(matrix)
    except (ValueError, SyntaxError):
        pass
    return "[[0, 1, 0], [1, 1, 0], [0, 1, 1]]"


def _compound_yes_no_example(answer: str) -> str:
    match = re.match(r"^(Yes|No),\s*(\d+),\s*(\d+)$", answer, re.IGNORECASE)
    if not match:
        return "Yes, 3, 2"
    yn = "No" if match.group(1).lower() == "yes" else "Yes"
    return f"{yn}, {int(match.group(2)) + 1}, {int(match.group(3))}"


def _direction_example(answer: str) -> str:
    directions = ["up", "down", "left", "right"]
    for direction in directions:
        if direction != answer.lower():
            return direction
    return "up"


def _pyramidchess_example(answer: str) -> str:
    match = re.match(
        r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s+at\s+level\s+(-?\d+)$",
        answer,
        re.IGNORECASE,
    )
    if not match:
        return "[2, 1] at level 0"
    x = int(match.group(1))
    y = int(match.group(2))
    level = int(match.group(3))
    return f"[{x + 1}, {y}] at level {level}"


def _ultra_tictactoe_example(answer: str) -> str:
    match = re.match(
        r"^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$",
        answer,
    )
    if not match:
        return "(1, 1, 2, 3)"
    i = int(match.group(1))
    j = int(match.group(2))
    row = int(match.group(3))
    col = int(match.group(4))
    return f"({i}, {j}, {row + 1}, {col})"


# ============================================================================
# Legacy functions for backward compatibility
# ============================================================================


def extract_clean_question(question: str, game_rules: str) -> str:
    """Extract clean question text (legacy function, use _clean_question_text)."""
    return _clean_question_text(question, game_rules)


def detect_option_prefix_type(options: list[str] | None) -> str | None:
    """Detect the prefix type used in options (letter or number)."""
    if not options:
        return None

    first = options[0].strip()
    if re.match(r"^[A-Za-z][.:\s)]", first):
        return "letter"
    if re.match(r"^\d+[.:\s)]", first):
        return "number"
    return None


def generate_answer_format_prompt(
    options: list[str] | None = None,
    oracle_answer: str | None = None,
) -> str:
    """Generate an ANSWER_FORMAT_PROMPT (legacy function)."""
    prefix_type = detect_option_prefix_type(options)

    if prefix_type == "letter":
        mc_hint = "Reply with only the letter (A, B, C, etc.)"
    elif prefix_type == "number":
        mc_hint = "Reply with only the number (1, 2, 3, etc.)"
    else:
        if oracle_answer:
            answer_val = str(oracle_answer).strip()
            if len(answer_val) == 1 and answer_val.isalpha():
                mc_hint = "Reply with only the letter (A, B, C, etc.)"
            elif answer_val.isdigit():
                mc_hint = "Reply with only the number (1, 2, 3, etc.)"
            else:
                mc_hint = "Reply with only the letter (A, B, C, etc.)"
        else:
            mc_hint = "Reply with only the letter (A, B, C, etc.)"

    return dedent(f"""
        **Answer Format:**
        - For multiple choice: {mc_hint}
        - For numbers: Reply with only the number

        Do not include any explanation or extra text.
    """).strip()


# ============================================================================
# Scoring helper functions for unified answer evaluation
# ============================================================================


def score_coordinate(answer: str, oracle_answer: str) -> float:
    """Score coordinate answer like (3, 5) or (1, 2, 3).

    Tolerates whitespace differences and optional parentheses.

    Args:
        answer: User's answer string
        oracle_answer: Expected answer string

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    # Try 3D coordinate first
    match_3d = re.search(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", answer)
    oracle_3d = re.search(
        r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", oracle_answer
    )
    if match_3d and oracle_3d:
        try:
            user_coords = tuple(int(match_3d.group(i)) for i in range(1, 4))
            oracle_coords = tuple(int(oracle_3d.group(i)) for i in range(1, 4))
            return 1.0 if user_coords == oracle_coords else 0.0
        except ValueError:
            pass

    # Try 2D coordinate
    match_2d = re.search(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", answer)
    oracle_2d = re.search(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", oracle_answer)
    if match_2d and oracle_2d:
        try:
            user_coords = (int(match_2d.group(1)), int(match_2d.group(2)))
            oracle_coords = (int(oracle_2d.group(1)), int(oracle_2d.group(2)))
            return 1.0 if user_coords == oracle_coords else 0.0
        except ValueError:
            pass

    return 0.0


def score_number(answer: str, oracle_answer: str) -> float:
    """Score numeric answer.

    Extracts the first number from the answer and compares with oracle.

    Args:
        answer: User's answer string
        oracle_answer: Expected answer string

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    match = re.search(r"-?\d+", answer)
    if not match:
        return 0.0

    try:
        num = int(match.group())
        oracle = int(oracle_answer)
        return 1.0 if num == oracle else 0.0
    except ValueError:
        return 0.0


def score_choice(answer: str, oracle_answer: str) -> float:
    """Score multiple choice answer (letter-based: A, B, C, D, etc.).

    Looks for standalone option letters (A-Z) that appear as choices.
    Prioritizes letters at the start of the answer or after common prefixes.

    Args:
        answer: User's answer string
        oracle_answer: Expected answer string (single letter)

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    answer = answer.strip()
    oracle = oracle_answer.strip().upper()

    # First try: single letter answer
    if len(answer) == 1 and answer.upper().isalpha():
        return 1.0 if answer.upper() == oracle else 0.0

    # Second try: letter at the start (possibly with punctuation)
    match = re.match(r"^([A-Za-z])[\s.:)\-]", answer)
    if match:
        return 1.0 if match.group(1).upper() == oracle else 0.0

    # Third try: standalone letter in the answer
    match = re.search(r"(?:^|[\s(])([A-Za-z])(?:[\s).,:]|$)", answer)
    if match:
        return 1.0 if match.group(1).upper() == oracle else 0.0

    # Fallback: first letter (for simple cases)
    if answer and answer[0].upper().isalpha():
        return 1.0 if answer[0].upper() == oracle else 0.0

    return 0.0


def score_number_choice(answer: str, oracle_answer: str) -> float:
    """Score multiple choice answer (number-based: 1, 2, 3, 4).

    Extracts the first number from the answer and compares with oracle.

    Args:
        answer: User's answer string
        oracle_answer: Expected answer string (single digit)

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    match = re.search(r"\d+", answer)
    if not match:
        return 0.0

    try:
        choice = int(match.group())
        oracle = int(oracle_answer)
        return 1.0 if choice == oracle else 0.0
    except ValueError:
        return 0.0


def score_exact(answer: str, oracle_answer: str) -> float:
    """Score exact match answer (case-insensitive, whitespace-tolerant).

    Args:
        answer: User's answer string
        oracle_answer: Expected answer string

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    return 1.0 if answer.strip().lower() == oracle_answer.strip().lower() else 0.0

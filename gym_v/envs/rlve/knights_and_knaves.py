"""Knights and Knaves environment for gym-v (self-contained)."""

from __future__ import annotations

import copy
import enum
import itertools
from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


# Knights and Knaves logic solving functions
def find_solution(statements):
    """Find solutions given a list of statements."""
    n_people = len(statements)
    single_statement = ('and',) + tuple(('<=>', ('telling-truth', i), statements[i])
                                        for i in range(len(statements)))
    solutions = []
    for assignments in itertools.product([True, False], repeat=n_people):
        if test_satisfiability(single_statement, assignments):
            solutions.append(assignments)
    return solutions


def test_satisfiability(statement, assignments):
    """Test if statement is satisfied with given assignments."""
    if statement[0] == 'telling-truth':
        return assignments[statement[1]]
    if statement[0] == 'lying':
        return not assignments[statement[1]]
    if statement[0] == 'not':
        return not test_satisfiability(statement[1], assignments)
    if statement[0] == 'and':
        return np.all([test_satisfiability(statement[i], assignments)
                       for i in range(1, len(statement))])
    if statement[0] == 'or':
        return np.any([test_satisfiability(statement[i], assignments)
                       for i in range(1, len(statement))])
    if statement[0] == '->':
        val1 = test_satisfiability(statement[1], assignments)
        val2 = test_satisfiability(statement[2], assignments)
        return (not val1) or val2
    if statement[0] == '<=>':
        val1 = test_satisfiability(statement[1], assignments)
        val2 = test_satisfiability(statement[2], assignments)
        return (val1 and val2) or ((not val1) and (not val2))
    raise KeyError(f'Unknown statement: {statement}')


class KKProblemSampler:
    """Problem Sampler for Knight and Knave."""

    def __init__(self, rng: np.random.Generator, n_people: int,
                 depth_constraint: int = 2, width_constraint: int = 2):
        self.rng = rng
        self.n_people = n_people
        self.depth_constraint = depth_constraint
        self.width_constraint = width_constraint

    def sample(self):
        """Sample a single K&K problem."""
        statements = tuple(self._sample_statement(person_id, self.depth_constraint)
                           for person_id in range(self.n_people))
        return self._immutable_statements(statements)

    def sample_valid_problems(self, n_problems: int, max_retry: int = 1000,
                              skip_no_solution: bool = True,
                              skip_multiple_solutions: bool = True):
        """Sample valid problems with unique solution."""
        problems = []
        unique_statements = set()
        for i_problem in range(n_problems):
            for _ in range(max_retry):
                statements = self.sample()
                if statements in unique_statements:
                    continue
                solutions = find_solution(statements)
                if len(solutions) == 0 and skip_no_solution:
                    continue
                if len(solutions) > 1 and skip_multiple_solutions:
                    continue
                sol = solutions[0] if len(solutions) > 0 else None
                problems.append({'statements': statements, 'solution': sol,
                                 'all_solutions': solutions})
                unique_statements.add(statements)
                break
            if i_problem + 1 > len(problems):
                raise RuntimeError(f'Failed to generate a valid problem after {max_retry} retries.')
        return problems

    def _immutable_statements(self, mutable_statements):
        """Change list back to tuples."""
        def _make_immutable(x):
            if isinstance(x, (list, tuple)):
                return tuple(_make_immutable(child) for child in x)
            if isinstance(x, np.str_):
                return str(x)
            if isinstance(x, np.int64):
                return int(x)
            return x
        return tuple(_make_immutable(s) for s in mutable_statements)

    def _sample_statement(self, person_id: int, depth_constraint: int):
        """Sample a single statement."""
        dice = self.rng.integers(0, 6)
        if depth_constraint == 1 or dice == 0:
            while True:
                knight_or_knave = self.rng.choice(['telling-truth', 'lying'])
                person = self.rng.integers(0, self.n_people)
                if not (knight_or_knave == 'lying' and person == person_id):
                    return (knight_or_knave, person)

        if dice == 1:
            return ('not', self._sample_statement(person_id, depth_constraint-1))
        if dice in [2, 3]:
            operator = ['and', 'or'][dice-2]
            n_substatements = self.rng.integers(2, self.width_constraint+1)
            return (operator,) + self._sample_substatements(person_id, depth_constraint, n_substatements)
        if dice in [4, 5]:
            operator = ['->', '<=>'][dice-4]
            return (operator,) + self._sample_substatements(person_id, depth_constraint, 2)

    def _sample_substatements(self, person_id: int, depth_constraint: int, count: int, dedup: bool = True):
        """Sample substatements for an operator."""
        sub_statements = []
        dedup_set = set()
        while True:
            stmt = self._sample_statement(person_id, depth_constraint-1)
            if dedup:
                if stmt in dedup_set:
                    continue
                dedup_set.add(stmt)
            sub_statements.append(stmt)
            if len(sub_statements) == count:
                break
        return tuple(sub_statements)


COMMON_NAMES = ['Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'Ethan', 'Sophia',
                'Mason', 'Isabella', 'William', 'Mia', 'James', 'Charlotte',
                'Benjamin', 'Amelia', 'Lucas', 'Harper', 'Henry', 'Evelyn',
                'Alexander', 'Abigail', 'Michael', 'Emily', 'Daniel', 'Elizabeth']

TEMPLATES = [
    '{name} said that {content}.',
    '{name} told you that {content}.',
    '{name} said, "{content}."',
    '{name} stated, "{content}".',
    'According to {name}, "{content}".',
    '''In {name}'s words: "{content}".''',
    '{name} remarked, "{content}".',
    '"{content}," {name} declared.',
]


def format_knight_knave(names, knight_knave, statement, negation=False):
    """Format a knight/knave statement."""
    assert statement[0] in ('telling-truth', 'lying')
    text = names[statement[1]] + ' is '
    if negation:
        text += 'not '
    text += {'telling-truth': knight_knave['a_knight'],
             'lying': knight_knave['a_knave']}[statement[0]]
    return text


def format_statement(names, knight_knave, statement):
    """Format a statement in natural language."""
    if statement[0] == 'not':
        return format_knight_knave(names, knight_knave, statement[1], negation=True)
    if statement[0] in ['and', 'or']:
        text = (' ' + statement[0] + ' ').join(
            format_knight_knave(names, knight_knave, sub_stmt) for sub_stmt in statement[1:])
        return text
    if statement[0] == '->':
        return ('If ' + format_knight_knave(names, knight_knave, statement[1]) + ' then ' +
                format_knight_knave(names, knight_knave, statement[2]))
    if statement[0] == '<=>':
        return (format_knight_knave(names, knight_knave, statement[1]) + ' if and only if ' +
                format_knight_knave(names, knight_knave, statement[2]))
    return format_knight_knave(names, knight_knave, statement)


class KKProblemFormatter:
    """Format Knights and Knaves problem."""

    def __init__(self, rng: np.random.Generator, problem):
        self.rng = rng
        self.problem = problem

    def format_problem(self, random_names=True, random_saying_template=True):
        """Format the problem in natural language."""
        statements = copy.deepcopy(self.problem['statements'])
        n_people = len(statements)

        names = COMMON_NAMES[:n_people]
        if random_names:
            names = list(self.rng.choice(COMMON_NAMES, size=n_people, replace=False))
        names = [str(x) for x in names]

        knight_knave = {
            'knight': 'knight',
            'knave': 'knave',
            'a_knight': 'a knight',
            'a_knave': 'a knave',
            'Knight': 'Knight',
            'Knave': 'Knave'
        }

        text = 'A very special island is inhabited only by knights and knaves. '
        text += 'Knights always tell the truth, and knaves always lie. '
        text += f'You meet {n_people} inhabitants: '
        text += ', '.join(names[:-1]) + ', and ' + names[-1] + '. '

        for i, stmt in enumerate(statements):
            tmpl = TEMPLATES[0]
            if random_saying_template:
                tmpl = self.rng.choice(TEMPLATES)
            content = format_statement(names, knight_knave, stmt)
            text += tmpl.format(name=names[i], content=content) + ' '

        text += 'So who is a knight and who is a knave?'

        if self.problem['solution'] is None:
            solution_text = 'No valid solution exists.'
        else:
            solution_stmts = []
            for name, indicator in zip(names, self.problem['solution']):
                if indicator:
                    solution_stmts.append(name + ' is ' + knight_knave['a_knight'])
                else:
                    solution_stmts.append(name + ' is ' + knight_knave['a_knave'])
            solution_text = ', '.join(solution_stmts[:-1]) + ', and ' + solution_stmts[-1] + '.'

        return {
            'quiz': text,
            'names': names,
            'knight_knave': knight_knave,
            'solution': self.problem['solution'],
            'solution_text': solution_text,
            'statements': statements
        }


class RLVEKnightsAndKnavesEnv(Env):
    """RLVE Knights and Knaves as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        N: int = 3,
        depth_constraint: int = 2,
        width_constraint: int = 2,
        image_width: int = 1000,
        image_height: int = 800,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}
        self._N = N
        self._depth_constraint = depth_constraint
        self._width_constraint = width_constraint
        self._image_width = image_width
        self._image_height = image_height

        self._names: list[str] | None = None
        self._statements: tuple | None = None
        self._solution: tuple[bool, ...] | None = None
        self._quiz_text: str | None = None
        self._prompt: str | None = None
        self._reference_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        """Return description adapted for visual input."""
        return dedent(
            """
            Knights and Knaves Logic Puzzle:

            Rules:
            1) Knights always tell the truth
            2) Knaves always lie
            3) Each person makes a statement about themselves or others
            4) Use logic to determine who is a knight and who is a knave

            In the image:
            - Each person is shown with their position
            - Knights are represented with shields (🛡️)
            - Knaves are represented with masks (🎭)
            - Question marks (❓) indicate unknown identities
            - Statements are shown below each person
            - Use logical deduction to solve the puzzle

            Output format: State the identity of each person, e.g.,
            "Emma is a knight, Liam is a knave, and Olivia is a knight."
            """
        ).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._generate()
        self._prompt = self._prompt_generate()
        self._last_image = self.render()

        obs = Observation(
            image=self._last_image,
            text=self._prompt,
            metadata={
                "rlve_prompt": self._prompt,
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
        }
        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

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
        reward = float(self._score_answer(action_str))

        obs = Observation(
            image=self._last_image,
            text=None,
            metadata={
                "rlve_prompt": self._prompt,
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {"reference_answer": self._reference_answer}

        terminated = True
        truncated = False

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

    def _generate(self) -> None:
        """Generate problem instance."""
        sampler = KKProblemSampler(
            rng=self.np_random,
            n_people=self._N,
            depth_constraint=self._depth_constraint,
            width_constraint=self._width_constraint
        )

        problems = sampler.sample_valid_problems(
            n_problems=1,
            max_retry=1000,
            skip_no_solution=True,
            skip_multiple_solutions=True
        )

        if not problems:
            raise RuntimeError("Failed to generate a valid problem")

        problem = problems[0]

        formatter = KKProblemFormatter(rng=self.np_random, problem=problem)
        formatted = formatter.format_problem()

        self._names = formatted["names"]
        self._statements = formatted["statements"]
        self._solution = formatted["solution"]
        self._quiz_text = formatted["quiz"]
        self._reference_answer = formatted["solution_text"]

    def _prompt_generate(self) -> str:
        """Generate the prompt for the problem instance."""
        return self._quiz_text

    def _process(self, answer: str) -> dict[str, str] | None:
        """Parse model's answer into status dictionary."""
        if not isinstance(answer, str):
            return None

        knight_count = answer.lower().count('knight')
        knave_count = answer.lower().count('knave')

        # Basic check: should have N assignments
        if knight_count + knave_count != self._N:
            return None

        import re
        status_dict = {}
        for name in self._names:
            # More strict pattern: name should be at word boundary or after punctuation
            # and followed by "is a knight" or "is a knave"
            pattern = re.compile(
                rf'(?:^|[,\s]){re.escape(name)}\s+is\s+a\s+(knight|knave)',
                re.IGNORECASE
            )
            match = pattern.search(answer)

            if match:
                role = match.group(1).lower()
                status_dict[name] = role
            else:
                return None

        return status_dict

    def _score_answer(self, answer: str) -> float:
        """Score the model's output."""
        processed_result = self._process(answer)

        if processed_result is None:
            return -1.0

        correct_solution = {
            name: 'knight' if is_knight else 'knave'
            for name, is_knight in zip(self._names, self._solution)
        }

        if processed_result == correct_solution:
            return 1.0

        # Partial credit: count how many identities are correct
        correct_count = sum(
            1 for name in self._names
            if processed_result[name] == correct_solution[name]
        )

        # Return a negative score proportional to mistakes
        return -0.5

    def render(self) -> Image.Image:
        """Render the Knights and Knaves puzzle as a beautiful visualization."""
        img = Image.new("RGB", (self._image_width, self._image_height), (250, 245, 235))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            title_font = ImageFont.truetype(str(font_path), 36)
            name_font = ImageFont.truetype(str(font_path), 28)
            statement_font = ImageFont.truetype(str(font_path), 18)
            small_font = ImageFont.truetype(str(font_path), 16)
        else:
            title_font = ImageFont.load_default()
            name_font = title_font
            statement_font = title_font
            small_font = title_font

        # Title
        title = "Knights and Knaves Logic Puzzle"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(((self._image_width - title_width) // 2, 30), title,
                  fill=(40, 40, 100), font=title_font)

        # Legend
        legend_y = 90
        legend_text = "🛡️ = Knight (always tells truth)    🎭 = Knave (always lies)    ❓ = Unknown"
        legend_bbox = draw.textbbox((0, 0), legend_text, font=small_font)
        legend_width = legend_bbox[2] - legend_bbox[0]
        draw.text(((self._image_width - legend_width) // 2, legend_y),
                  legend_text, fill=(80, 80, 80), font=small_font)

        # Calculate layout for people
        person_width = self._image_width // (self._N + 1)
        start_y = 180
        person_spacing = self._image_width // (self._N + 1)

        knight_knave = {
            'knight': 'knight',
            'knave': 'knave',
            'a_knight': 'a knight',
            'a_knave': 'a knave'
        }

        for i, name in enumerate(self._names):
            x = person_spacing * (i + 1)

            # Draw person circle with unknown status (question mark)
            circle_radius = 50
            draw.ellipse(
                [x - circle_radius, start_y - circle_radius,
                 x + circle_radius, start_y + circle_radius],
                fill=(200, 200, 230),
                outline=(100, 100, 150),
                width=3
            )

            # Draw question mark (unknown identity)
            question_mark = "❓"
            qm_bbox = draw.textbbox((0, 0), question_mark, font=name_font)
            qm_width = qm_bbox[2] - qm_bbox[0]
            qm_height = qm_bbox[3] - qm_bbox[1]
            draw.text((x - qm_width // 2, start_y - qm_height // 2),
                      question_mark, fill=(100, 100, 150), font=name_font)

            # Draw name below circle
            name_bbox = draw.textbbox((0, 0), name, font=name_font)
            name_width = name_bbox[2] - name_bbox[0]
            draw.text((x - name_width // 2, start_y + circle_radius + 10),
                      name, fill=(40, 40, 40), font=name_font)

            # Draw statement in a box below
            statement_text = format_statement(self._names, knight_knave, self._statements[i])

            # Wrap statement text
            max_width = person_width - 20
            words = statement_text.split()
            lines = []
            current_line = []

            for word in words:
                test_line = ' '.join(current_line + [word])
                bbox = draw.textbbox((0, 0), test_line, font=statement_font)
                if bbox[2] - bbox[0] <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))

            # Draw statement box
            statement_y = start_y + circle_radius + 50
            box_height = len(lines) * 25 + 20
            box_top = statement_y
            box_bottom = statement_y + box_height

            draw.rectangle(
                [x - person_width // 2 + 10, box_top,
                 x + person_width // 2 - 10, box_bottom],
                fill=(255, 255, 240),
                outline=(150, 150, 150),
                width=2
            )

            # Draw statement text
            for j, line in enumerate(lines):
                line_bbox = draw.textbbox((0, 0), line, font=statement_font)
                line_width = line_bbox[2] - line_bbox[0]
                draw.text((x - line_width // 2, box_top + 10 + j * 25),
                          line, fill=(40, 40, 40), font=statement_font)

        # Add instruction at bottom
        instruction = "Determine who is a knight and who is a knave using logical deduction."
        inst_bbox = draw.textbbox((0, 0), instruction, font=small_font)
        inst_width = inst_bbox[2] - inst_bbox[0]
        draw.text(((self._image_width - inst_width) // 2, self._image_height - 40),
                  instruction, fill=(80, 80, 80), font=small_font)

        return img

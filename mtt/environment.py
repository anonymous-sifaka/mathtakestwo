import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
import numpy as np
import re
import random

from .environment_data import shape_codes
from .environment_data import shapes
from .environment_data import number_set

from .environment_data import markov_transition_probs

from .environment_data import edge_cases

from .environment_data import practice_questions
from .environment_data import practice_assignments
from .environment_data import practice_rinds

from .environment_data import test_questions
from .environment_data import test_assignments
from .environment_data import test_rinds


# Class for sorting and returning set of 3x3 shapes - could be generalzed to e.g. 4x4

class ShapeLibrary:
    def __init__(self, mode='precond'):

        assert mode in {'precond', 'practice', 'test'}

        self.mode = mode

        self.shape_codes = shape_codes()
        self.shapes = shapes()

        self.shape_dict = {code: np.array(shape) for code, shape in zip(self.shape_codes, self.shapes)}

        # Restrict shape library in benchmark phase
        if self.mode == 'precond':
            self.shape_codes = self.shape_codes[:8]
            self.shape_dict = {c: self.shape_dict[c] for c in self.shape_codes}

        if self.mode == 'practice':
            self.shape_codes = self.shape_codes[:10]
            self.shape_dict = {c: self.shape_dict[c] for c in self.shape_codes}

        if self.mode == 'test':
            self.shape_dict = {c: self.shape_dict[c] for c in self.shape_codes}

    def get_shape(self, code):
        """Return the shape matrix for a given code."""
        return self.shape_dict.get(code, None)

    def list_codes(self):
        """Return all available shape codes."""
        return list(self.shape_dict.keys())

    def all_shapes(self):
        """Return the full dictionary of shape codes to matrices."""
        return self.shape_dict


# Library of numbers restricted to training mode
class NumberLibrary:
    def __init__(self, mode='precond'):

        assert mode in {'precond', 'practice', 'test'}

        self.mode = mode
        self.numbers = number_set()

        # Restrict shape library to benchmark phase
        if self.mode == 'precond':
            self.numbers = self.numbers[:6]

        if self.mode == 'practice':
            self.numbers = self.numbers[:7]

    def list_numbers(self):
        """Return all available shape codes."""
        return self.numbers


## Class for drawing shapes on a canvas

class ShapeDrawer:
    """
    A utility for drawing 3x3 (or custom-sized) shapes on a 2D canvas.
    Shapes are placed on a grid and sourced from a provided shape library.
    """

    def __init__(self, shape_lib, max_shapes=10, max_canvas_width=47):
        """
        Initialize the canvas and shape configuration.

        Args:
            shape_lib: An object with methods `get_shape(code)` and `list_codes()`
                       that returns shapes as numpy arrays.
            max_shapes_per_col (int): Maximum number of shapes per column.
            max_canvas_width (int): Width of the canvas (in pixels).
        """
        self.shape_lib = shape_lib
        self.max_shapes = max_shapes

        # Determine shape size from first shape
        sample_shape = self.shape_lib.get_shape(self.shape_lib.list_codes()[0])
        self.shape_size = sample_shape.shape[0]

        # Canvas dimensions
        self.canvas_height = self.max_shapes * (self.shape_size + 1) + 1
        self.canvas_width = max_canvas_width
        self.canvas = np.zeros((self.canvas_width, self.canvas_height))

        # Drawing state
        self.offset = 0
        self.max_x = 0
        self.shape_set = set(self.shape_lib.list_codes())

    def draw_shape(self, x, y, shape_code):
        """
        Draw a single shape at a grid position (x, y).

        Args:
            x (int): Row index.
            y (int): Column index.
            shape_code (str): Shape identifier in shape_lib.
        """
        if shape_code not in self.shape_set:
            raise ValueError(f"Shape '{shape_code}' is not valid.")

        # Apply horizontal offset for first shape in new group
        if x == 0 and y == 0:
            self.offset = self.max_x + 1

        px = x * (self.shape_size + 1) + self.offset
        py = y * (self.shape_size + 1) + 1

        shape = self.shape_lib.get_shape(shape_code)
        self.canvas[px:px + self.shape_size, py:py + self.shape_size] = shape

        # Track farthest x-drawn for offsetting next column group
        self.max_x = max(self.max_x, px + self.shape_size + 2)

    def draw_all(self):
        """
        Draw all shapes from the shape library in grid order.
        """
        for idx, code in enumerate(self.shape_lib.list_codes()):
            x = idx // self.max_shapes
            y = idx % self.max_shapes
            self.draw_shape(x, y, code)

    def draw_image(self, figsize=(5, 5)):
        """
        Display the canvas using matplotlib.

        Args:
            figsize (tuple): Size of the matplotlib figure.
        """
        plt.figure(figsize=figsize)
        plt.imshow(self.canvas.T, cmap='gray')
        plt.axis('off')
        plt.show()

    def clear_canvas(self):
        """
        Reset the canvas and drawing state.
        """
        self.canvas = np.zeros((self.canvas_width, self.canvas_height))
        self.offset = 0
        self.max_x = 0


# Class for rendering shape programs

class ShapeProgramParser:
    def __init__(self, drawer):
        """
        Initialize with a ShapeDrawer instance.
        """
        self.drawer = drawer

    @staticmethod
    def parse_trinary(s):
        """Convert a trinary (base-3) string to int."""
        return int(s, 3)

    def render(self, prog_str, show_image=True, figsize=(5, 5)):
        """
        Parses a program string and uses the ShapeDrawer to render shapes.
        Supported commands:
          - SHAPEcc       → vertical stack of SHAPE with count=trinary(cc)
          - SHAPE*cc      → horizontal row of SHAPE, width=trinary(cc)
          - SHAPEcc*cc    → grid of SHAPE, rows × cols in trinary
          - SHAPE         → single shape
        """
        self.drawer.clear_canvas()
        tokens = prog_str.split('+')

        grid = 0  # Starting grid index

        for token in tokens:
            token = token.strip()

            # Pattern: SHAPErr*cc (grid)
            match_grid = re.fullmatch(r'([A-Z]{1,2})([012]+)\*([012]+)', token)
            if match_grid:
                shape, row_bits, col_bits = match_grid.groups()
                rows = self.parse_trinary(row_bits)
                cols = self.parse_trinary(col_bits)
                for i in range(rows):
                    for j in range(cols):
                        self.drawer.draw_shape(j, i, shape)
                continue

            # Pattern: SHAPE*cc (horizontal row)
            match_row = re.fullmatch(r'([A-Z]{1,2})\*([012]+)', token)
            if match_row:
                shape, col_bits = match_row.groups()
                cols = self.parse_trinary(col_bits)
                for j in range(cols):
                    self.drawer.draw_shape(j, 0, shape)
                continue

            # Pattern: SHAPEcc (vertical line)
            match_col = re.fullmatch(r'([A-Z]{1,2})([012]+)', token)
            if match_col:
                shape, count_bits = match_col.groups()
                count = self.parse_trinary(count_bits)
                for i in range(count):
                    self.drawer.draw_shape(0, i, shape)
                continue

            # Single shape
            if token in self.drawer.shape_set:
                self.drawer.draw_shape(0, 0, token)
                continue

            raise ValueError(f"Unrecognized shape token: {token}")

        if show_image:
            self.drawer.draw_image(figsize=figsize)

        return self.drawer.canvas


# Class for generating random valid shape programs based on Markov state model

class MarkovProgramGenerator:

    def __init__(self, shape_lib, number_lib, max_length=8, transitions=None):

        self.max_length = max_length

        self.shapes = shape_lib.list_codes()
        self.numbers = number_lib.list_numbers()

        if transitions == None:
            self.transitions_nm = self._build_transitions_nm()
        else:
            self.transitions_nm = transitions

        self.shape_lib = shape_lib
        self.transitions_m = None

    def _build_transitions_nm(self):

        """
        Import Markov Transition Probabilities

        """

        return markov_transition_probs(self.shapes, self.numbers)

    def _build_transitions_m(self):

        """
        Remove * option after use until next char values

        """

        transitions_m = dict()

        for k, v in self.transitions_nm.items():
            transitions_m[k] = [ve for ve in v if ve[0] != '*']

        return transitions_m

    def _sample_next_token(self, current, array_mode):

        """
        Samples the next token based on current state and mode.

        Args:
            current_state (str): Current token/state.
            array_mode (bool): Whether array mode is active.

        Returns:
            str: Next token.
        """

        transitions = self.transitions_m if array_mode else self.transitions_nm

        options = transitions.get(current, [('END', 1.0)])
        tokens, probs = zip(*options)

        choice = random.choices(tokens, weights=probs)[0]

        shape_set = set(self.shapes)

        if choice in shape_set:
            shape_set.remove(choice)
            self.shapes = list(shape_set)
            transitions['+'] = [(s, 1.0 / len(self.shapes)) for s in self.shapes]

        return choice

    def generate_program(self):

        """
        Generate a single program string.

        Returns:
            str: Generated program.
        """

        program = []
        current = 'START'
        array_mode = False

        while len(program) < self.max_length:

            token = self._sample_next_token(current, array_mode)
            if token == 'END':
                break

            if current == '*':
                array_mode = True

                if self.transitions_m == None:
                    self.transitions_m = self._build_transitions_m()

            if current in {'A', 'B', 'C'}:
                array_mode = False

            # Add characters from token (multi-char support)
            for ch in token:
                if len(program) < self.max_length:
                    program.append(ch)

            current = token[-1]  # base next state on last char

        self.shapes = self.shape_lib.list_codes()
        return ''.join(program).strip('*+')

    def generate_batch(self, count=20):
        return [self.generate_program() for _ in range(count)]


# Class for creating test question based on input program

class ProgramQuestionGenerator:
    """
    Generates multiple-choice visual questions by mutating or randomly generating
    symbolic programs based on shape and number vocabularies.
    """

    def __init__(self, parser, shape_lib, number_lib, generator=None,
                 prob_shape=0.3, prob_number=0.5, prob_both=0.2):
        """
        Initialize the question generator.

        Args:
            parser: Object with a `.render(program_str)` method.
            shape_lib: Object with `.list_codes()` method.
            number_lib: Object with `.list_numbers()` method.
            generator: Optional program generator with `.generate_batch(n)` method.
            prob_shape (float): Probability of shape-only mutation.
            prob_number (float): Probability of number-only mutation.
            prob_both (float): Probability of shape + number mutation.
        """
        assert abs(prob_shape + prob_number + prob_both - 1.0) < 1e-6, "Probabilities must sum to 1.0"

        self.parser = parser
        self.generator = generator

        self.shapes = shape_lib.list_codes()
        self.numbers = number_lib.list_numbers()

        self.prob_shape = prob_shape
        self.prob_number = prob_number
        self.prob_both = prob_both

    def mutate_shape_token(self, program_str):
        """
        Replace a shape token in the program with a different one.
        """
        pattern = re.compile(r'(' + '|'.join(sorted(self.shapes, key=len, reverse=True)) + r')')
        matches = list(pattern.finditer(program_str))

        if not matches:
            return program_str

        match = random.choice(matches)
        old_shape = match.group(1)
        alternatives = [s for s in self.shapes if s != old_shape]

        if not alternatives:
            return program_str

        new_shape = random.choice(alternatives)
        return program_str[:match.start()] + new_shape + program_str[match.end():]

    def mutate_number_token(self, program_str):
        """
        Replace a number token in the program with a different one of the same length.
        """
        number_sorted = sorted(self.numbers, key=len, reverse=True)
        pattern = re.compile(r'(' + '|'.join(map(re.escape, number_sorted)) + r')')
        matches = list(pattern.finditer(program_str))

        if not matches:
            return program_str

        match = random.choice(matches)
        old_number = match.group(1)
        alternatives = [n for n in self.numbers if n != old_number and len(n) == len(old_number)]

        if not alternatives:
            return program_str

        new_number = random.choice(alternatives)
        return program_str[:match.start()] + new_number + program_str[match.end():]

    def mutate_program(self, base_program, existing_programs):
        """
        Mutate a given program in a way that it's not in the existing list.
        """
        mutation_type = random.choices(
            ['shape', 'number', 'both'],
            weights=[self.prob_shape, self.prob_number, self.prob_both]
        )[0]

        mutated = base_program
        attempt = 0

        while mutated in existing_programs:
            attempt += 1
            if mutation_type == 'shape':
                mutated = self.mutate_shape_token(mutated)
            elif mutation_type == 'number':
                mutated = self.mutate_number_token(mutated)
            else:  # 'both' or fallback
                mutated = self.mutate_shape_token(mutated)
                mutated = self.mutate_number_token(mutated)

            if attempt > 25:
                return None  # Give up if mutation fails too many times

        return mutated

    def generate_question(self, base_program, n_mutate=2, n_random=1,
                          show_image=False, return_text=False, options=None, rinds=None):
        """
        Create a multiple-choice visual question with one correct and several distractor options.

        Args:
            base_program (str): The correct program.
            n_mutate (int): Number of mutated options.
            n_random (int): Number of randomly generated distractors.
            show_image (bool): Whether to display the images.
            return_text (bool): Whether to return program strings.
            options (list): Optional pre-defined options.
            rinds (array-like): Permutation for shuffling options.

        Returns:
            Tuple:
              - imgs: List of rendered images
              - answer: Boolean array indicating which is correct
              - [optional] options: The corresponding program strings
        """
        assert n_mutate + n_random == 3, f"Expected 3 options besides base; got {n_mutate + n_random}"

        if options is None:
            options = [base_program]

            # Generate mutated variants
            for _ in range(n_mutate):
                mutated = self.mutate_program(base_program, options)
                if mutated is None:
                    # Fallback to random if mutation fails
                    options += self.generator.generate_batch(n_random)
                else:
                    options.append(mutated)

            # Add random distractors
            options += self.generator.generate_batch(n_random)

        # Shuffle and determine correct answer location
        rinds = np.random.permutation(4) if rinds is None else rinds
        shuffled_options = [options[r] for r in rinds]
        answer = (rinds == 0)

        imgs = []
        for i, prog in enumerate(shuffled_options):
            img = self.parser.render(prog, show_image=False)
            imgs.append(img)

        if show_image:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            for i, (ax, img) in enumerate(zip(axes.flat, imgs)):
                ax.imshow(img.T, cmap='gray')
                ax.set_title(f"Option {i + 1}")
                ax.axis('off')
            plt.tight_layout()
            plt.show()

        return (imgs, answer, shuffled_options) if return_text else (imgs, answer)


class ExampleGenerator:
    """
    Generates examples, questions, and visualizations for symbolic shape-number programs.
    Used for testing symbolic reasoning or vision-language models.
    """

    def __init__(self, mode='precond', valid_num=500, figsize=(5, 5)):
        """
        Initialize the generator with rendering tools, program generators, and question generator.

        Args:
            mode (str): Currently unused but can be extended for multiple logic types.
            valid_num (int): Number of pre-generated validation examples.
            figsize (tuple): Figure size for rendering.
        """
        shape_lib = ShapeLibrary()
        number_lib = NumberLibrary()
        drawer = ShapeDrawer(shape_lib, max_shapes=10)

        self.parser = ShapeProgramParser(drawer)
        self.generator = MarkovProgramGenerator(shape_lib, number_lib, max_length=8)
        self.questions_generator = ProgramQuestionGenerator(
            self.parser, shape_lib, number_lib, generator=self.generator
        )

        self.valid_num = valid_num
        self.valid_cases = self.generator.generate_batch(valid_num)
        self.valid_set = set(self.valid_cases)
        self.valid_counter = 0
        self.figsize = figsize

    def show_example(self, return_figure=False):
        """
        Render a random example and a multiple-choice question.
        """
        example = self.generator.generate_batch(1)[0]

        if return_figure:
            img = self.parser.render(example, figsize=self.figsize, show_image=False)
            imgs = self.questions_generator.generate_question(example, show_image=False)

            return img, imgs

        else:
            print('Example Image:')
            self.parser.render(example, figsize=self.figsize)
            print('Example Question: Which answer is correct?')
            self.questions_generator.generate_question(example, show_image=True)

    def show_edge_cases(self, return_figure=False):
        """
        Display predefined edge cases using the parser.
        """
        if return_figure:
            imgs = []
            for case in edge_cases():
                imgs.append(self.parser.render(case, figsize=self.figsize, show_image=False))
            return imgs

        else:
            for case in edge_cases():
                self.parser.render(case, figsize=self.figsize)

    def generate_q_batch(self, program_str):
        """
        Generate a multiple-choice question and return image batch + answer.

        Args:
            program_str (str): The base program string.

        Returns:
            tuple: (image_batch: [4, 1, H, W], answer: [bool array of length 4])
        """
        images, answer = self.questions_generator.generate_question(program_str, show_image=False)
        images_with_channel = [img[np.newaxis, ...] for img in images]  # Add channel dim
        batch = np.stack(images_with_channel, axis=0)
        return batch, answer

    def get_qna(self):
        """
        Generate a new question-answer pair with an unseen program.

        Returns:
            tuple: (single image, 4-option image batch, answer array)
        """
        example = self._sample_new_example()
        img = self.parser.render(example, show_image=False)
        batch, answer = self.generate_q_batch(example)
        return img, batch, answer

    def get_qna_valid(self):
        """
        Retrieve a validation set question-answer pair.

        Returns:
            tuple: (single image, 4-option image batch, answer array)
        """
        self.valid_counter = (self.valid_counter + 1) % self.valid_num
        program = self.valid_cases[self.valid_counter]
        img = self.parser.render(program, show_image=False)
        batch, answer = self.generate_q_batch(program)
        return img, batch, answer

    def get_image(self):
        """
        Get a new rendered image (not from validation set).

        Returns:
            np.ndarray: Rendered image.
        """
        program = self._sample_new_example()
        return self.parser.render(program, show_image=False)

    def get_image_valid(self):
        """
        Get the next validation image.

        Returns:
            np.ndarray: Rendered image.
        """
        self.valid_counter = (self.valid_counter + 1) % self.valid_num
        program = self.valid_cases[self.valid_counter]
        return self.parser.render(program, show_image=False)

    def _sample_new_example(self):
        """
        Generate a program that is not in the validation set.

        Returns:
            str: A unique program string.
        """
        while True:
            example = self.generator.generate_batch(1)[0]
            if example not in self.valid_set:
                return example


import numpy as np


class QuizGenerator:
    """
    Orchestrates practice and test phases for a communication task between
    a 'describer' and a 'visualizer'. Each player receives assigned questions
    from a shared pool and sees either a rendered image or multiple choice options.
    """

    def __init__(self, player_n, player_type, figsize=(5, 5)):
        """
        Initialize a quiz session for a given player and role.

        Args:
            player_n (int): Player index (0–9), defines the question subset assigned.
            player_type (str): Role of the player ('describer' or 'visualizer').
            figsize (tuple): Size of the figures to be rendered.
        """
        self.player_n = player_n
        self.player_type = player_type  # 'describer' or 'visualizer'
        self.figsize = figsize

        # === Initialize practice shape/number libraries and tools ===
        self.shape_lib_practice = ShapeLibrary(mode='practice')
        self.number_lib_practice = NumberLibrary(mode='practice')
        drawer_practice = ShapeDrawer(self.shape_lib_practice, max_shapes=10)
        self.parser_practice = ShapeProgramParser(drawer_practice)
        self.practice_generator = ProgramQuestionGenerator(
            self.parser_practice, self.shape_lib_practice, self.number_lib_practice
        )

        # === Initialize test shape/number libraries and tools ===
        self.shape_lib_test = ShapeLibrary(mode='test')
        self.number_lib_test = NumberLibrary(mode='test')
        drawer_test = ShapeDrawer(self.shape_lib_test, max_shapes=10)
        self.parser_test = ShapeProgramParser(drawer_test)
        self.test_generator = ProgramQuestionGenerator(
            self.parser_test, self.shape_lib_test, self.number_lib_test
        )

        # === Load question sets and assignment metadata ===
        self.practice_programs, self.practice_questions = practice_questions()
        self.practice_as = practice_assignments()
        self.practice_rinds = practice_rinds()

        self.test_programs, self.test_questions = test_questions()
        self.test_as = test_assignments()
        self.test_rinds = test_rinds()

        # === Progress tracking ===
        self.practice_count = 0
        self.test_count = 0
        self.prev_rind = []

    def get_next_practice(self, return_figure=False):
        """
        Advance to the next practice question and render based on player role.
        Describer sees an image; visualizer sees options.
        """
        if self.practice_count >= 10:
            if self.player_type == 'visualizer' and not return_figure:
                print('The answer to the last question is:',
                      np.where(np.asarray(self.prev_rind) == 0)[0][0] + 1)
            print('✅ Practice Complete - Please move onto the next stage')
            return None

        set_key = f'set_{self.practice_count}'
        idx = self.practice_as[set_key][self.player_n]
        program = self.practice_programs[set_key][idx]

        if self.practice_count >= 1 and self.player_type == 'visualizer' and not return_figure:
            print('The answer to the last question is:',
                  np.where(np.asarray(self.prev_rind) == 0)[0][0] + 1)

        if self.player_type == 'describer':
            if return_figure:
                img = self.parser_practice.render(program, figsize=self.figsize, show_image=False)
                self.practice_count += 1
                return img
            else:
                print(f'Question {self.practice_count} (Describer) - Image:\n')
                self.parser_practice.render(program, figsize=self.figsize)
                self.practice_count += 1
                return None

        elif self.player_type == 'visualizer':
            options = self.practice_questions[set_key][idx]
            rinds = self.practice_rinds[set_key][idx]

            if return_figure:
                imgs, answer = self.practice_generator.generate_question(
                    program, show_image=False, options=options, rinds=rinds
                )
                self.practice_count += 1
                return imgs
            else:
                print(f'Question {self.practice_count} (Visualizer): Which image is the describer seeing?\n')
                self.practice_generator.generate_question(
                    program, show_image=True, options=options, rinds=rinds
                )
                self.practice_count += 1

            self.prev_rind = rinds
            return None
        return None

    def get_next_test(self, return_figure=False):
        """
        Advance to the next test question and render based on player role.
        Describer sees image; visualizer sees options. No answers are shown.
        """
        if self.test_count >= 10:
            print('✅ Test Complete - Congratulations!')
            return

        set_key = f'set_{self.test_count}'
        idx = self.test_as[set_key][self.player_n]
        program = self.test_programs[set_key][idx]

        if self.player_type == 'describer':

            if return_figure:
                img = self.parser_test.render(program, figsize=self.figsize, show_image=False)
                self.test_count += 1
                return img
            else:
                print(f'Question {self.test_count} (Describer) - Image:\n')
                self.parser_test.render(program, figsize=self.figsize)
                self.test_count += 1

        elif self.player_type == 'visualizer':
            options = self.test_questions[set_key][idx]
            rinds = self.test_rinds[set_key][idx]

            if return_figure:
                imgs, answer = self.test_generator.generate_question(
                    program, show_image=False, options=options, rinds=rinds
                )
                self.test_count += 1
                return imgs
            else:
                print(f'Question {self.test_count} (Visualizer): Which image is the describer seeing?\n')
                self.test_generator.generate_question(
                    program, show_image=True, options=options, rinds=rinds
                )
                self.test_count += 1
            return None
        return None


import numpy as np


class QuizGeneratorML:
    """
    Handles programmatic access to practice/test questions for ML or human-based evaluation.
    Supports both model-driven ("ml_version") and player-driven ("human_version") use cases.
    """

    def __init__(self, mode='ml_version', player_index=None, player_role=None):

        """
        Initialize a quiz session for either ML evaluation or human testing.

        Args:
            mode (str): Either 'ml_version' or 'human_version'.
            player_index (int): Index of the human player (0–9). Required for 'human_version'.
            player_role (str): Role of the player ('describer' or 'visualizer'). Required for 'human_version'.
        """

        assert mode in {'ml_version', 'human_version'}
        self.mode = mode

        if mode == 'human_version':
            assert player_index is not None and player_role in {'describer', 'visualizer'}
            self.player_index = player_index
            self.player_role = player_role

        # === Initialize shape/number libraries ===
        self.shape_lib_practice = ShapeLibrary(mode='practice')
        self.number_lib_practice = NumberLibrary(mode='practice')

        self.shape_lib_test = ShapeLibrary(mode='test')
        self.number_lib_test = NumberLibrary(mode='test')

        # === Initialize parsers and question generators ===
        self.parser_practice = ShapeProgramParser(ShapeDrawer(self.shape_lib_practice, max_shapes=10))
        self.practice_generator = ProgramQuestionGenerator(
            self.parser_practice, self.shape_lib_practice, self.number_lib_practice
        )

        self.parser_test = ShapeProgramParser(ShapeDrawer(self.shape_lib_test, max_shapes=10))
        self.test_generator = ProgramQuestionGenerator(
            self.parser_test, self.shape_lib_test, self.number_lib_test
        )

        # === Load questions and assignments ===
        self.programs = {
            'practice': practice_questions()[0],
            'test': test_questions()[0]
        }
        self.questions = {
            'practice': practice_questions()[1],
            'test': test_questions()[1]
        }
        self.assignments = {
            'practice': practice_assignments(),
            'test': test_assignments()
        }
        self.rinds = {
            'practice': practice_rinds(),
            'test': test_rinds()
        }

        # === Track counts and total lengths ===
        self.counts = {'practice': 0, 'test': 0}
        self.lengths = {
            'practice': sum(len(v) for v in self.assignments['practice'].values()),
            'test': sum(len(v) for v in self.assignments['test'].values())
        }

    def get_qna(self, phase='practice'):
        """
        Return a question-answer tuple for the given phase.

        Args:
            phase (str): One of {'practice', 'test'}.

        Returns:
            tuple:
                - img (np.ndarray): Rendered image of the correct program.
                - options (List[np.ndarray]): List of 4 rendered answer images.
                - answer (np.ndarray): Boolean array indicating which image is correct.
        """
        assert phase in {'practice', 'test'}, "Phase must be 'practice' or 'test'"

        # Get index: assume 10 programs per 'set'
        set_index = self.counts[phase] // 10
        item_index = self.counts[phase] % 10
        set_key = f'set_{set_index}'

        program = self.programs[phase][set_key][item_index]
        options = self.questions[phase][set_key][item_index]
        rinds = np.asarray(self.rinds[phase][set_key][item_index])

        # Render and generate question
        if phase == 'practice':
            img = self.parser_practice.render(program, show_image=False)
            option_imgs, answer_mask = self.practice_generator.generate_question(
                program, show_image=False, options=options, rinds=rinds
            )
        else:
            img = self.parser_test.render(program, show_image=False)
            option_imgs, answer_mask = self.test_generator.generate_question(
                program, show_image=False, options=options, rinds=rinds
            )

        # Update internal counter
        self.counts[phase] += 1
        if self.counts[phase] >= self.lengths[phase]:
            self.counts[phase] = 0

        return img, option_imgs, answer_mask, program, options


class EvalLogger:
    def __init__(self, mode_1, mode_2):

        assert mode_1 in {'precond', 'practice', 'test'}
        assert mode_1 in {'precond', 'practice', 'test'}

        mode_1_codes = set(ShapeLibrary(mode=mode_1).shape_codes)
        mode_2_codes = set(ShapeLibrary(mode=mode_2).shape_codes)
        self.new_codes = list(mode_2_codes.difference(mode_1_codes))

        mode_1_numbers = set(NumberLibrary(mode=mode_1).numbers)
        mode_2_numbers = set(NumberLibrary(mode=mode_2).numbers)
        self.new_numbers = list(mode_2_numbers.difference(mode_1_numbers))

        self.logging_dict = {
            'ood_symbol_question': 0,
            'ood_number_question': 0,
            'ood_both_question': 0,
            'ood_symbol_answer': 0,
            'ood_number_answer': 0,
            'ood_both_answer': 0,
        }

        self.correct_dict = self.logging_dict.copy()
        self.count, self.correct = 0, 0

    def reset(self, ):

        self.logging_dict = {
            'ood_symbol_question': 0,
            'ood_number_question': 0,
            'ood_both_question': 0,
            'ood_symbol_answer': 0,
            'ood_number_answer': 0,
            'ood_both_answer': 0,
        }
        self.correct_dict = logging_dict.copy()

    def test_qna(self, answer, prediction, program, options):

        # High level scores updates

        self.count += len(prediction)

        correct = prediction == answer
        correct = np.asarray(correct.cpu())
        self.correct += correct.sum()

        for zz in range(len(program)):
            # Test if question is out of distribution

            for v in self.new_codes:
                if v in program[zz]:
                    self.logging_dict['ood_symbol_question'] += 1
                    self.correct_dict['ood_symbol_question'] += correct[zz]

            for v in self.new_numbers:
                if v in program[zz]:
                    self.logging_dict['ood_number_question'] += 1
                    self.correct_dict['ood_number_question'] += correct[zz]

            for v in self.new_codes:
                for w in self.new_numbers:
                    if v in program[zz] and w in program[0]:
                        self.logging_dict['ood_both_question'] += 1
                        self.correct_dict['ood_both_question'] += correct[zz]

                        # Test if answer is out of distribution

            flag1, flag2, flag3 = False, False, False

            for v in self.new_codes:
                for p in options:
                    if v in p[zz]:
                        flag1 = True
            if flag1:
                self.logging_dict['ood_symbol_answer'] += 1
                self.correct_dict['ood_symbol_answer'] += correct[zz]

            for v in self.new_numbers:
                for p in options:
                    if v in p[zz]:
                        flag2 = True
            if flag2:
                self.logging_dict['ood_number_answer'] += 1
                self.correct_dict['ood_number_answer'] += correct[zz]

            for v in self.new_codes:
                for w in self.new_numbers:
                    for p in options:
                        if v in p[zz] and w in p[0]:
                            flag3 = True
            if flag3:
                self.logging_dict['ood_both_answer'] += 1
                self.correct_dict['ood_both_answer'] += correct[zz]

    def return_results(self):

        self.acc_dict = dict()
        for k, v in self.logging_dict.items():
            if v > 0:
                self.acc_dict[k] = self.correct_dict[k] / v
            else:
                self.acc_dict[k] = 0

        self.eval_results = {'question_n': self.count,
                             'correct_n': self.correct,
                             'accuracy': self.correct / self.count,
                             'logging_dict': self.logging_dict,
                             'correct_dict': self.correct_dict,
                             'accuracy_dict': self.acc_dict}
        return self.eval_results

        
        
        
        
        
        
        
    

    
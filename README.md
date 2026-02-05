
# Symbolic Communication Benchmark

This repository contains a modular framework for generating and evaluating symbolic reasoning tasks between agents (or models), focusing on visual communication using structured shape-number programs.

The system supports:

* Human-in-the-loop experiments (describer ↔ visualizer)
* ML-based benchmarking with QnA-style data
* Preconditioned image-only and QnA datasets for training/validation

---

## Key Components

### `QuizGenerator`

Used for **human-based evaluation**, supporting two roles:

* **Describer**: views a symbolic image and generates a description.
* **Visualizer**: receives multiple options and selects the correct image.

Each player is assigned a unique subset of questions for **practice** and **test** phases.

**Methods**:

* `get_next_practice()`: Steps through a practice question for the player.
* `get_next_test()`: Steps through a test question for the player.

---

### `QuizGeneratorML`

A model-friendly version of `QuizGenerator` for automated evaluation.

**Use case**: programmatically access batches of (image, question, answer) triplets for training/testing ML models.

**Methods**:

* `get_qna(phase='practice')`: Returns a single (image, 4-option choices, correct answer) tuple for `phase ∈ {'practice', 'test'}`.

---

### `ExampleGenerator`

Utility class for debugging, exploration, and demoing symbolic tasks.

**Features**:

* `show_example()`: Display a random symbolic image and related QnA.
* `get_qna() / get_qna_valid()`: Return (img, questions, answer) triplet.
* `get_image() / get_image_valid()`: Return symbolic image alone.

---

### `PrecondDataset` (PyTorch)

Dataset wrapper for both image-only and QnA-based symbolic data. Supports training/validation splits.

**Modes**:

* `img_train` / `img_val`: Image-only modes for vision-based pretraining.
* `qna_train` / `qna_val`: Full symbolic QnA format for supervised learning.

**Returns**:

* Image mode: `Tensor [1, H, W]`
* QnA mode: `(Tensor [1, H, W], Tensor [4, 1, H, W], answer_index)`

---

## Dependencies

Make sure the following are installed:

```bash
pip install numpy torch torchvision matplotlib
```

---

## Usage Examples

### Image Pretraining (PyTorch)

```python
train_ds = PrecondDataset(generator=ExampleGenerator(), mode='img_train')
img = train_ds[0]  # Tensor shape: [1, H, W]
```

### QnA Task with ML Generator

```python
quiz_ml = QuizGeneratorML()
img, options, answer = quiz_ml.get_qna(phase='test')
```

### Human Describer Round

```python
quiz = QuizGenerator(player_n=2, player_type='describer')
quiz.get_next_practice()  # Renders image
```

---

## Project Structure

```plaintext
├── generators/
│   ├── QuizGenerator.py
│   ├── QuizGeneratorML.py
│   ├── ExampleGenerator.py
│   └── PrecondDataset.py
├── libs/
│   ├── ShapeLibrary.py
│   ├── NumberLibrary.py
│   └── ShapeDrawer.py
├── assignments/
│   ├── practice_questions.py
│   ├── test_questions.py
│   └── practice_rinds.py
```

---

## Evaluation Protocol

* **Practice Phase**: Visual feedback is provided.
* **Test Phase**: No feedback, designed for evaluation.
* Each player sees a different subset of questions via assignment matrices.

---


## License

MIT License. Attribution appreciated.








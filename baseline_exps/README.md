# MLLM Training Setup

This directory contains the updated MLLM (Multimodal Large Language Model) training setup for the MathTakesTwo benchmark.

## Files Overview

### Core Components

- **`mllm_models.py`**: Contains the MLLM model architectures
  - `VisionEncoder`: Vision Transformer for image encoding
  - `SymbolicTransformerDecoder`: Decodes vision features to symbolic representations
  - `SymbolicTransformerEncoder`: Encodes symbolic sequences
  - `MLLMQnAModel`: Complete Q&A model that combines vision and symbolic processing
  - `MLLMStack`: Full stack for image comparison tasks

- **`mllm_trainers.py`**: Training utilities
  - `MLLMQnATrainer`: Handles training loop, validation, and early stopping

- **`dataloaders.py`**: Data loading utilities
  - `PrecondDataset`: Dataset for preconditioned training data
  - `PracTestDataset`: Dataset for practice/test data

### Training Scripts

- **`mtt_mllm_updated.ipynb`**: Clean, updated notebook for MLLM training
- **`test_mllm_setup.py`**: Test script to verify the setup works correctly

## Key Changes Made

### 1. Model Architecture Consistency
- Added `MLLMQnAModel` class that properly handles Q&A tasks
- Fixed model interfaces to be consistent across components
- Ensured proper tensor shapes and data flow

### 2. Data Loading Consistency
- Updated dataloaders to handle different data structures consistently
- Fixed tensor conversion to handle both numpy arrays and tensors
- Improved error handling for batch processing

### 3. Training Pipeline
- Updated trainer to handle flexible batch structures
- Fixed accuracy calculation to use actual batch sizes
- Improved early stopping and model saving

## Usage

### Quick Start

1. **Test the setup**:
   ```bash
   cd baseline_exps
   python test_mllm_setup.py
   ```

2. **Run training**:
   ```bash
   # Open the notebook
   jupyter notebook mtt_mllm_updated.ipynb
   ```

### Model Configuration

The `MLLMQnAModel` accepts the following parameters:
- `embed_dim`: Embedding dimension (default: 512)
- `num_symbols`: Number of symbolic tokens (default: 8)
- `seq_len`: Sequence length for symbolic processing (default: 8)
- `num_answers`: Number of possible answers (default: 4)

### Training Configuration

The training configuration includes:
- `num_epochs`: Number of training epochs
- `patience`: Early stopping patience
- `save_path`: Path to save the best model
- `device`: Training device (cuda/cpu)

## Data Environment

The training uses the MathTakesTwo environment:
- **Preconditioned data**: Generated using `env.ExampleGenerator(mode="precond")`
- **Practice/Test data**: Generated using `env.QuizGeneratorML()`
- **Data modes**: `qna_train`, `qna_val`, `qna_prac`, `qna_test`

## Model Architecture

The `MLLMQnAModel` follows this pipeline:
1. **Vision Encoding**: Images → Vision Transformer → Features
2. **Symbolic Decoding**: Features → Symbolic Transformer → Symbol sequences
3. **Symbolic Encoding**: Symbol sequences → Encoded representations
4. **Question Processing**: Questions → Question embeddings
5. **Cross-Attention**: Message-question attention
6. **Answer Prediction**: Combined features → Answer logits

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure the project root is in your Python path
2. **CUDA errors**: Check if CUDA is available and models are on the correct device
3. **Data shape errors**: Verify that the data format matches expected shapes

### Debugging

Use the test script to isolate issues:
```bash
python test_mllm_setup.py
```

This will test:
- Model creation and forward pass
- Data loading
- Training step execution

## Performance Notes

- The model uses a frozen Vision Transformer for efficiency
- Symbolic processing is lightweight and fast
- Cross-attention mechanism allows for flexible question-answer matching
- Early stopping prevents overfitting

## Future Improvements

- Add support for different vision encoders
- Implement curriculum learning
- Add more sophisticated attention mechanisms
- Support for variable-length symbolic sequences 
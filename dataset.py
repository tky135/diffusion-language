"""
Dataset loading and creation helpers used by main.py.
"""

import csv

import torch

# HuggingFace datasets for text8
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


def create_simple_dataset():
    """
    Create a simple sequential dataset with sequences of length 4.
    Data: [0,1,2,3], [1,2,3,4], [2,3,4,5], ..., [9,0,1,2]
    """
    data = []
    for i in range(10):
        seq = [(i + j) % 10 for j in range(4)]
        data.append(seq)

    data = torch.tensor(data, dtype=torch.int64)
    print(f"[simple] Dataset shape: {data.shape}")
    print(f"Dataset (first 5 rows):\n{data[:5]}")
    return data


def load_sudoku_dataset_csv(csv_path: str, input_type: str = 'solution_only',
                            sep_token_id: int = 11, eof_token_id: int = 12):
    """
    Load Sudoku dataset from CSV file with columns 'quizzes' and 'solutions'.

    Args:
        csv_path: Path to CSV file
        input_type: Input format for training data
            - 'solution_only': Returns only solutions [N, 81] (default)
            - 'quiz_solution': Returns quiz+SEP+solution+EOF [N, 164]
        sep_token_id: Token ID for separator between quiz and solution (default: 11)
        eof_token_id: Token ID for end-of-sequence marker (default: 12)

    Returns:
        Training files:
            - If input_type='solution_only': tensor of solutions (shape [N, 81])
            - If input_type='quiz_solution': tensor of quiz+SEP+solution+EOF (shape [N, 164])
        Test files: Always returns tuple (quiz_tensor, solution_tensor), each [N, 81]
    """
    solutions = []
    quizes = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            solution_str = row['solutions']
            solution = [int(c) for c in solution_str]
            solutions.append(solution)

            quiz_str = row['quizzes'].replace(".", "0")
            quiz = [int(c) for c in quiz_str]
            quizes.append(quiz)

    solution_data = torch.tensor(solutions, dtype=torch.int64)
    quiz_data = torch.tensor(quizes, dtype=torch.int64)

    print(f"[sudoku] Loaded from {csv_path}")

    if "test" in csv_path.lower():
        # Test data: always return (quiz, solution) separately for evaluation
        print(f"Test dataset shape: quiz={quiz_data.shape}, solution={solution_data.shape}")
        print(f"First quiz:\n{quiz_data[0].reshape(9, 9)}")
        print(f"First solution:\n{solution_data[0].reshape(9, 9)}")
        return quiz_data, solution_data
    else:
        # Training data: format depends on input_type
        if input_type == 'solution_only':
            data = solution_data
            print(f"Training dataset shape (solution only): {data.shape}")
            print(f"First solution:\n{data[0].reshape(9, 9)}")
            return data
        elif input_type == 'quiz_solution':
            # Concatenate: quiz (81) + SEP (1) + solution (81) + EOF (1) = 164 tokens
            N = quiz_data.shape[0]
            sep_tokens = torch.full((N, 1), sep_token_id, dtype=torch.int64)
            eof_tokens = torch.full((N, 1), eof_token_id, dtype=torch.int64)

            data = torch.cat([quiz_data, sep_tokens, solution_data, eof_tokens], dim=1)  # [N, 164]

            print(f"Training dataset shape (quiz+SEP+solution+EOF): {data.shape}")
            print(f"First example:")
            print(f"  Quiz part:\n{data[0, :81].reshape(9, 9)}")
            print(f"  SEP token: {data[0, 81].item()}")
            print(f"  Solution part:\n{data[0, 82:163].reshape(9, 9)}")
            print(f"  EOF token: {data[0, 163].item()}")
            return data
        else:
            raise ValueError(f"input_type must be 'solution_only' or 'quiz_solution', got '{input_type}'")


def tokenize_text8(text: str):
    """
    Tokenize text for text8 dataset using 27-token vocabulary.

    Vocabulary:
        0-25: lowercase letters a-z
        26: whitespace (space)

    Args:
        text: Input text string

    Returns:
        List of token indices
    """
    text_lower = text.lower()
    tokens = []
    for char in text_lower:
        if 'a' <= char <= 'z':
            tokens.append(ord(char) - ord('a'))  # 0-25
        elif char == ' ':
            tokens.append(26)  # space token
        # Skip other characters (punctuation, digits, etc.)

    return tokens


def load_text8_dataset(split='train', seq_len=64, stride=None):
    """
    Load text8 dataset from HuggingFace and prepare sequences with sliding window.

    Args:
        split: Dataset split ('train', 'validation', or 'test')
        seq_len: Sequence length (default: 64)
        stride: Sliding window stride (default: seq_len // 2 for 50% overlap)

    Returns:
        torch.tensor of shape [N, seq_len] with dtype=int64
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError(
            "datasets library required for text8 dataset. "
            "Install with: pip install datasets"
        )

    if stride is None:
        stride = seq_len // 2  # 50% overlap by default

    print(f"[text8] Loading dataset split '{split}' with seq_len={seq_len}, stride={stride}")
    print(f"[text8] Vocabulary: 27 tokens (a-z: 0-25, space: 26)")

    # Load dataset from HuggingFace (single row with full text)
    dataset = load_dataset("afmck/text8")
    text = dataset[split][0]['text']

    # Tokenize
    print(f"[text8] Tokenizing text...")
    tokens = tokenize_text8(text)

    # Create sequences with sliding window
    print(f"[text8] Creating sequences with sliding window...")
    sequences = []
    for i in range(0, len(tokens) - seq_len + 1, stride):
        sequences.append(tokens[i:i + seq_len])

    # Convert to tensor
    data = torch.tensor(sequences, dtype=torch.int64)

    print(f"[text8] Split: {split}")
    print(f"[text8] Dataset shape: {data.shape}")
    print(f"[text8] Total tokens: {len(tokens):,}, Sequences: {data.shape[0]:,}")

    # Decode first sequence for verification
    first_decoded = decode_text8_tokens(data[0])
    if len(first_decoded) > 80:
        first_decoded = first_decoded[:80] + "..."
    print(f"[text8] First sequence: {first_decoded}")

    return data


def decode_text8_tokens(tokens):
    """
    Decode text8 token indices back to text.

    Args:
        tokens: List or tensor of token indices

    Returns:
        Decoded text string
    """
    if torch.is_tensor(tokens):
        tokens = tokens.tolist()

    chars = []
    for token in tokens:
        if 0 <= token <= 25:
            chars.append(chr(ord('a') + token))
        elif token == 26:
            chars.append(' ')
        else:
            chars.append('?')  # Unknown token
    return ''.join(chars)

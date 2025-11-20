"""
Dataset loading and creation helpers used by main.py.
"""

import csv

import torch


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


def load_sudoku_dataset(csv_path: str):
    """
    Load Sudoku dataset from CSV file with columns 'quizzes' and 'solutions'.

    Returns:
        Training files -> tensor of solutions (shape [N, 81])
        Test files     -> tuple (quiz_tensor, solution_tensor), each [N, 81]
    """
    solutions = []
    quizes = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            solution_str = row['solutions']
            solution = [int(c) for c in solution_str]
            solutions.append(solution)

            quiz_str = row['quizzes']
            quiz = [int(c) for c in quiz_str]
            quizes.append(quiz)

    data = torch.tensor(solutions, dtype=torch.int64)
    print(f"[sudoku] Loaded from {csv_path}")
    print(f"Dataset shape: {data.shape}")
    print(f"First solution:\n{data[0].reshape(9, 9)}")

    if "test" in csv_path.lower():
        quiz_data = torch.tensor(quizes, dtype=torch.int64)
        print(f"First quiz:\n{quiz_data[0].reshape(9, 9)}")
        return quiz_data, data
    else:
        return data

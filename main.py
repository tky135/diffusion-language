"""

Datasets supported:
- simple:        sequences length 4, [i, i+1, i+2, i+3] [~100%]
- sudoku:        full Sudoku solutions [0%]
- sudoku_simple: first row  [93.40%] python main.py --dataset sudoku_simple --lr 1e-3 --batch_size 512 --steps 60000 --lr_warmup_steps 5000 --positional_encoding sinusoidal --embed_dim 2
- sudoku_tiny:   all permutations of {0,1,2,3} [99.2%]
"""

import fire
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import math
import os
import shutil
import glob
from datetime import datetime
from typing import Dict, List, Optional
from itertools import permutations
from torch.utils.tensorboard import SummaryWriter

from lib import ops as lib_ops


def setup_experiment_dir(exp_name: Optional[str] = None, base_dir: str = "experiments") -> str:
    """
    Create experiment directory with timestamp and optional name.
    Also creates a backup folder with all .py files.

    Args:
        exp_name: Optional experiment name to append to timestamp
        base_dir: Base directory for experiments

    Returns:
        Path to the created experiment directory
    """
    # Create timestamp in format: YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create experiment folder name
    if exp_name:
        exp_folder = f"{timestamp}_{exp_name}"
    else:
        exp_folder = timestamp

    # Full path to experiment directory
    exp_dir = os.path.join(base_dir, exp_folder)
    os.makedirs(exp_dir, exist_ok=True)

    # Create backup directory
    backup_dir = os.path.join(exp_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)

    # Find and backup all .py files recursively
    py_files = glob.glob("*.py", recursive=True)
    py_files.extend(glob.glob("lib/*.py", recursive=True))

    for py_file in py_files:
        # Skip files in virtual environments, __pycache__, etc.
        if any(skip in py_file for skip in ["venv", "env", "__pycache__", ".git", "site-packages"]):
            continue

        # Create subdirectories in backup if needed
        dest_path = os.path.join(backup_dir, py_file)
        dest_dir = os.path.dirname(dest_path)
        if "experiments" in dest_dir:
            continue

        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        try:
            shutil.copy2(py_file, dest_path)
        except Exception as e:
            print(f"Warning: Could not backup {py_file}: {e}")

    print(f"\n{'='*60}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Backed up {len([f for f in py_files if not any(skip in f for skip in ['venv', 'env', '__pycache__', '.git', 'site-packages'])])} Python files to {backup_dir}")
    print(f"{'='*60}\n")

    # Create README for experiment directory
    readme_path = os.path.join(exp_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"# Experiment: {exp_folder}\n\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Directory Structure\n\n")
        f.write("```\n")
        f.write(f"{exp_folder}/\n")
        f.write("├── backup/          # Backup of all Python source files\n")
        f.write("├── runs/            # TensorBoard logs\n")
        f.write("├── config.txt       # Experiment configuration\n")
        f.write("├── checkpoint.pt    # Model checkpoint\n")
        f.write("├── output.txt       # Training/sampling scores\n")
        f.write("├── digit_embeddings.png     # Visualization of learned embeddings\n")
        f.write("├── training_loss_*.png      # Loss curves\n")
        f.write("└── README.md        # This file\n")
        f.write("```\n\n")
        f.write("## How to View Results\n\n")
        f.write("### TensorBoard\n")
        f.write("```bash\n")
        f.write(f"tensorboard --logdir {os.path.join(exp_dir, 'runs')}\n")
        f.write("```\n\n")
        f.write("### Configuration\n")
        f.write("See `config.txt` for full experiment configuration.\n\n")
        f.write("### Checkpoint\n")
        f.write(f"Model checkpoint is saved at `checkpoint.pt`\n\n")
        f.write("To load:\n")
        f.write("```python\n")
        f.write("checkpoint = torch.load('checkpoint.pt')\n")
        f.write("model.load_state_dict(checkpoint['model_state_dict'])\n")
        f.write("embedding.load_state_dict(checkpoint['embedding_state_dict'])\n")
        f.write("```\n")

    return exp_dir


def get_dispersion_loss(x_feature):
    # get random permutation of indices
    rand_indices = torch.randperm(x_feature.shape[0])
    x_feature_comp = x_feature[rand_indices]  # (B, L*D)

    D = 1 - F.cosine_similarity(x_feature, x_feature_comp)  # (B, L*D)

    # dispersion loss
    disp_loss = torch.log(torch.exp(-D / 0.5).mean())
    return disp_loss

def visualize_embeddings(embedding_weights: torch.Tensor, save_path: Optional[str] = None):
    """Plot the learned digit embeddings in 2D using PCA."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover
        print("matplotlib not available; skipping embedding visualization.")
        return

    emb = embedding_weights.detach().cpu()
    if emb.ndim != 2 or emb.size(0) == 0:
        print("Embedding tensor has unexpected shape; skipping visualization.")
        return

    mean = emb.mean(dim=0, keepdim=True)
    centered = emb - mean
    # Perform PCA via SVD for stability (not strictly needed for the simple plots below)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    # components = centered @ Vh[:, :2]

    x = emb[:, 0].numpy()
    y = emb[:, 1].numpy() if emb.size(1) > 1 else np.zeros_like(x)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color="tab:blue", edgecolors="black")
    for idx, (px, py) in enumerate(zip(x, y)):
        plt.text(px, py, str(idx), fontsize=12, ha="center", va="center", color="white",
                 bbox=dict(facecolor="tab:blue", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3"))

    plt.title("Digit Embeddings")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(alpha=0.25)
    plt.axhline(0, color="grey", linewidth=0.5)
    plt.axvline(0, color="grey", linewidth=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved embedding visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_loss_series(loss_dict: Dict[str, List[float]], base_path: Optional[str] = None, show: bool = False) -> bool:
    """Plot each loss series in its own figure.

    Returns True if at least one plot is shown or saved.
    """
    if not loss_dict:
        print("No loss series provided; skipping loss plots.")
        return False

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover
        print("matplotlib not available; skipping loss plots.")
        return False

    emitted = False
    ext = ".png"
    base_dir = None
    base_name = None

    if base_path:
        base_dir, filename = os.path.split(base_path)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        name, candidate_ext = os.path.splitext(filename)
        if candidate_ext:
            ext = candidate_ext
        base_name = name or "training_loss"
    else:
        base_name = "training_loss"

    for key, values in loss_dict.items():
        if not values:
            continue

        steps = np.arange(1, len(values) + 1)
        losses_np = np.asarray(values, dtype=np.float32)

        plt.figure(figsize=(7, 4))
        plt.plot(steps, losses_np, linewidth=1.5, color="tab:orange")
        plt.title(f"{key.title()} Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(alpha=0.25)

        filename = None
        if base_name:
            safe_key = key.lower().replace(" ", "_")
            filename = f"{base_name}_{safe_key}{ext}"
            output_path = os.path.join(base_dir or "", filename)
            plt.savefig(output_path, bbox_inches="tight")
            print(f"Saved {key} loss plot to {output_path}")
            emitted = True

        if show:
            plt.show()
            emitted = True

        plt.close()

    if not emitted:
        print("No loss plots were generated.")

    return emitted


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

def create_randpair_dataset():
    data = []
    for i in range(10):
        for j in range(10):
            data.append([i, (i + 1) % 10, j, (j + 1) % 10])
    data = torch.tensor(data, dtype=torch.int64)
    print(f"[simple] Dataset shape: {data.shape}")
    print(f"Dataset (first 5 rows):\n{data[:5]}")
    return data
def load_sudoku_dataset(csv_path):
    """
    Load Sudoku dataset from CSV file.
    CSV format: quizzes,solutions
    Each puzzle/solution is an 81-character string (9x9 grid flattened).
    0 represents empty cells in quizzes.

    Args:
        csv_path: Path to CSV file

    Returns:
        torch.Tensor: Sudoku solutions of shape [N, 81] with values 1-9
    """
    import csv

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
        print(f"First quiz:\n{torch.tensor(quizes, dtype=torch.int64)[0].reshape(9, 9)}")
        return quiz_data, data
    else:
        return data


def load_sudoku_first_row_dataset(csv_path: str) -> torch.Tensor:
    """
    Load only the first row (9 digits) from each Sudoku solution.
    Returns shape [N, 9], values 1..9.
    """
    full = load_sudoku_dataset(csv_path)  # [N, 81]
    rows = full[:, :9].contiguous()       # first 9 digits = first row
    print(f"[sudoku_simple] Using only first row: shape {rows.shape}")
    return rows


def create_sudoku_tiny_dataset(repeat: int = 1, shuffle: bool = True) -> torch.Tensor:
    """
    Tiny 'sudoku' dataset: every sequence is a permutation of {0,1,2,3}.
    Returns shape [24*repeat, 4].
    """
    base = list(permutations([0, 1, 2, 3], 4))  # 24 tuples
    data = torch.tensor(base, dtype=torch.int64)
    if repeat > 1:
        data = data.repeat(repeat, 1)
    if shuffle:
        idx = torch.randperm(len(data))
        data = data[idx]
    print(f"[sudoku_tiny] Dataset shape: {data.shape} (24 perms x repeat={repeat})")
    print(data[:8])
    return data


class EmbeddingMatrix(nn.Module):
    """Embedding matrix with per-row normalization."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        matrix = torch.randn(vocab_size, embed_dim)
        with torch.no_grad():
            matrix /= matrix.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        self.matrix = nn.Parameter(matrix)

    def forward(self, tokens=None):
        norm = torch.linalg.norm(self.matrix, dim=1, keepdim=True)
        normalized = self.matrix / (norm + 1e-8)
        if tokens is None:
            return normalized
        return normalized[tokens]


class OneHotEmbedding(nn.Module):
    """Fixed one-hot embedding matrix."""

    def __init__(self, vocab_size):
        super().__init__()
        matrix = torch.eye(vocab_size, dtype=torch.float32)
        self.register_buffer("matrix", matrix, persistent=False)

    def forward(self, tokens=None):
        if tokens is None:
            return self.matrix
        return self.matrix[tokens]


class UnitSphereEmbedding(nn.Module):
    """Fixed embedding matrix with digits uniformly distributed on a 2D unit circle."""

    def __init__(self, vocab_size):
        super().__init__()
        # Create uniformly distributed points on unit circle
        # For vocab_size digits, place them at angles: 2π * i / vocab_size
        angles = torch.arange(vocab_size, dtype=torch.float32) * (2.0 * math.pi / vocab_size)
        
        # Convert to cartesian coordinates on unit circle
        matrix = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        
        self.register_buffer("matrix", matrix, persistent=False)

    def forward(self, tokens=None):
        if tokens is None:
            return self.matrix
        return self.matrix[tokens]


class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=False),
        )

    def forward(self, x):  # x: [B, T, dim]
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        residual = x
        x = self.norm1(x)

        qkv = self.attn_qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]             # [B, H, T, D]

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # [B, H, T, D]
        out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, T, C)        # [B, T, C]
        x = residual + self.attn_out(out)

        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class SimpleDiffusionModel(nn.Module):
    """
    Simplified diffusion model for discrete sequences.
    Takes noisy embeddings and predicts clean embeddings.
    """
    def __init__(self, embed_dim, hidden_dim, n_blocks, n_heads, vocab_size, seq_len, 
                 positional_encoding: str = "learned", dataset_type: str = "simple"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.positional_encoding = positional_encoding.lower()
        self.dataset_type = dataset_type.lower()

        # Project embedding to hidden dimension
        self.input_proj = nn.Linear(embed_dim, hidden_dim, bias=False)

        # Positional embeddings (learned, sinusoidal 1D, or sinusoidal 2D)
        if self.dataset_type == "sudoku" and self.positional_encoding == "sinusoidal":
            # Use 2D positional encoding for sudoku (9x9 grid = 81 positions)
            print(f"Using 2D sinusoidal positional encoding for sudoku dataset")
            pe = self._build_2d_sinusoidal_embedding(9, 9, hidden_dim)
            self.register_buffer("pos_embedding", pe, persistent=False)
        elif self.positional_encoding == "learned":
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        elif self.positional_encoding == "sinusoidal":
            pe = self._build_sinusoidal_embedding(seq_len, hidden_dim)
            self.register_buffer("pos_embedding", pe, persistent=False)
        else:
            raise ValueError(f"Unknown positional_encoding '{positional_encoding}'. Use 'learned' or 'sinusoidal'.")

        # Time/noise level embedding (using sinusoidal encoding)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(hidden_dim, n_heads)
            for _ in range(n_blocks)
        ])

        # Output projection
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=True)

    @staticmethod
    def _build_sinusoidal_embedding(seq_len: int, dim: int) -> torch.Tensor:
        """Build 1D sinusoidal positional embedding."""
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / max(dim, 1)))
        pe = torch.zeros(seq_len, dim, dtype=torch.float32)
        sinusoid = position * div_term
        pe[:, 0::2] = torch.sin(sinusoid)
        if dim > 1:
            cos_columns = pe[:, 1::2].shape[1]
            pe[:, 1::2] = torch.cos(sinusoid[:, :cos_columns])
        return pe.unsqueeze(0)

    @staticmethod
    def _build_2d_sinusoidal_embedding(height: int, width: int, dim: int) -> torch.Tensor:
        """
        Build 2D sinusoidal positional embedding for grid-structured data.
        
        Args:
            height: Number of rows in the grid (e.g., 9 for Sudoku)
            width: Number of columns in the grid (e.g., 9 for Sudoku)
            dim: Embedding dimension
            
        Returns:
            Tensor of shape [1, height*width, dim]
        """
        # Split dimension between row and column encodings
        assert dim % 2 == 0, "Embedding dimension must be even for 2D positional encoding"
        d_model = dim // 2
        
        # Create position indices
        pe = torch.zeros(height, width, dim, dtype=torch.float32)
        
        # Generate row encodings (first half of dimensions)
        row_pos = torch.arange(height, dtype=torch.float32).unsqueeze(1)  # [height, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                            (-math.log(9.0) / d_model))
        
        row_sinusoid = row_pos * div_term  # [height, d_model//2]
        pe[:, :, 0:d_model:2] = torch.sin(row_sinusoid).unsqueeze(1).repeat(1, width, 1)
        pe[:, :, 1:d_model:2] = torch.cos(row_sinusoid).unsqueeze(1).repeat(1, width, 1)
        
        # Generate column encodings (second half of dimensions)
        col_pos = torch.arange(width, dtype=torch.float32).unsqueeze(1)  # [width, 1]
        col_sinusoid = col_pos * div_term  # [width, d_model//2]
        pe[:, :, d_model::2] = torch.sin(col_sinusoid).unsqueeze(0).repeat(height, 1, 1)
        pe[:, :, d_model+1::2] = torch.cos(col_sinusoid).unsqueeze(0).repeat(height, 1, 1)
        
        # Flatten spatial dimensions: [height, width, dim] -> [height*width, dim] -> [1, height*width, dim]
        pe = pe.view(height * width, dim).unsqueeze(0)
        
        return pe

    def get_time_embedding(self, gamma, dim):
        """Create sinusoidal time embeddings."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=gamma.device) * -emb)
        emb = gamma[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

    def forward(self, z, gamma):
        """
        Args:
            z: noisy embeddings [batch, seq_len, embed_dim]
            gamma: noise level [batch]
        Returns:
            logits: predicted token logits [batch, seq_len, vocab_size]
        """
        x = self.input_proj(z)  # [B, T, hidden_dim]

        # Positional information
        pos_emb = self.pos_embedding[:, :x.size(1), :]
        x = x + pos_emb

        # Time embedding
        time_emb = self.get_time_embedding(gamma, self.hidden_dim)  # [B, hidden_dim]
        time_emb = self.time_mlp(time_emb)                          # [B, hidden_dim]
        x = x + time_emb[:, None, :]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        x = self.norm_out(x)
        logits = self.output_proj(x)  # [B, T, vocab_size]
        return logits


def main(**args):
    # Default arguments
    def _coerce_bool(value, default):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    # Setup experiment directory
    exp_name = args.get('exp_name', None)
    exp_dir = setup_experiment_dir(exp_name=exp_name, base_dir="experiments")

    # Dataset selection
    dataset_type = str(args.get('dataset', 'simple')).lower()
    sudoku_train_path = args.get('sudoku_train_path', 'data/sudoku_train.csv')
    sudoku_test_path = args.get('sudoku_test_path', 'data/sudoku_test.csv')

    batch_size = args.get('batch_size', 512)
    lr = args.get('lr', 1e-4)
    lr_decay = args.get('lr_decay', True)
    lr_decay_end = args.get('lr_decay_end', 1e-5)
    lr_warmup_steps = args.get('lr_warmup_steps', 1000)
    lr_decay_steps = args.get('lr_decay_steps', None)
    steps = args.get('steps', 20000)
    print_freq = args.get('print_freq', 100)
    embed_dim = args.get('embed_dim', 4)
    hidden_dim = args.get('hidden_dim', 32)
    n_blocks = args.get('n_blocks', 4)
    n_heads = args.get('n_heads', 4)

    # Setup paths relative to experiment directory
    embed_plot_path = args.get('embed_plot_path', os.path.join(exp_dir, 'digit_embeddings.png'))
    loss_plot_path = args.get('loss_plot_path', os.path.join(exp_dir, 'training_loss.png'))
    checkpoint_path = args.get('checkpoint_path', os.path.join(exp_dir, 'checkpoint.pt'))
    load_checkpoint_path = args.get('resume_from', None)
    tensorboard_log_dir = args.get('tensorboard_log_dir', os.path.join(exp_dir, 'runs'))
    score_output_file = os.path.join(exp_dir, 'output.txt')

    plot_loss_curve = _coerce_bool(args.get('plot_loss_curve', True), True)
    show_loss_plot = _coerce_bool(args.get('show_loss_plot', False), False)
    embedding_type = str(args.get('embedding_type', 'learned')).lower()
    positional_encoding = str(args.get('positional_encoding', 'learned')).lower()
    sampling_only = args.get('sampling_only', False)
    resume = args.get('resume', False)
    
    if load_checkpoint_path is not None:
        resume = True
        
    if resume is True and load_checkpoint_path is None:
        load_checkpoint_path = checkpoint_path

    # DDPM-style noise schedule parameters
    num_timesteps = args.get('num_timesteps', 1000)
    beta_start = args.get('beta_start', 0.0001)
    beta_end = args.get('beta_end', 0.02)

    # Precompute DDPM noise schedule
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device='cpu')
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    eps = 1e-6
    alpha_sq = alphas_cumprod.clamp(min=eps, max=1.0 - eps)
    sigma_sq = (1.0 - alphas_cumprod).clamp(min=eps, max=1.0 - eps)
    gamma_table = torch.log(sigma_sq / alpha_sq)
    denom = max(num_timesteps - 1, 1)
    dt = 1.0 / denom
    gamma_prime_table = torch.zeros_like(gamma_table)
    if num_timesteps > 1:
        gamma_prime_table[1:-1] = (gamma_table[2:] - gamma_table[:-2]) / (2 * dt)
        gamma_prime_table[0] = (gamma_table[1] - gamma_table[0]) / dt
        gamma_prime_table[-1] = (gamma_table[-1] - gamma_table[-2]) / dt

    alpha_1_scalar = sqrt_alphas_cumprod[-1]
    sigma_1_scalar = sqrt_one_minus_alphas_cumprod[-1]

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset / problem setup
    if dataset_type == 'sudoku':
        vocab_size = 10  # Digits 0-9 (but we'll use 1-9 for Sudoku)
        seq_len = 81     # 9x9 grid
        data = None
        test_data = None
        if not sampling_only:
            data = load_sudoku_dataset(sudoku_train_path)
            test_quiz, test_data = load_sudoku_dataset(sudoku_test_path)

    elif dataset_type == 'sudoku_simple':
        # Only the first row (9 digits) from each Sudoku solution
        vocab_size = 10
        seq_len = 9
        data = None
        test_data = None
        if not sampling_only:
            data = load_sudoku_first_row_dataset(sudoku_train_path)   # [N, 9]
            test_data = load_sudoku_first_row_dataset(sudoku_test_path)

    elif dataset_type == 'sudoku_tiny':
        # Permutations of {0,1,2,3}: seq_len=4, vocab_size=4
        vocab_size = 4
        seq_len = 4
        data = None
        test_data = None
        if not sampling_only:
            tiny_repeat = int(args.get('tiny_repeat', 1))   # optional oversampling
            data = create_sudoku_tiny_dataset(repeat=tiny_repeat, shuffle=True)
            test_data = data.clone()

    elif dataset_type == 'sequential':
        # original 4-token toy sequence
        vocab_size = 10
        seq_len = 4
        data = None
        test_data = None
        if not sampling_only:
            data = create_simple_dataset()

    elif dataset_type == 'randompair':
        vocab_size = 10
        seq_len = 4
        data = create_randpair_dataset()
        
    else:
        raise Exception

    print("="*60)
    print("Simple Diffusion Model for Sequential Data (DDPM)")
    print("="*60)
    print(f"Experiment directory: {exp_dir}")
    print(f"dataset: {dataset_type}")
    print(f"vocab_size: {vocab_size}, seq_len: {seq_len}")
    print(f"batch_size: {batch_size}")
    print(f"lr: {lr}")
    print(f"steps: {steps}")
    print(f"embed_dim: {embed_dim}, hidden_dim: {hidden_dim}")
    print(f"n_blocks: {n_blocks}, n_heads: {n_heads}")
    print(f"num_timesteps: {num_timesteps}")
    print(f"beta_start: {beta_start}, beta_end: {beta_end}")
    print(f"embedding_type: {embedding_type}")
    print(f"positional_encoding: {positional_encoding}")
    print("="*60)
    print()

    # Open output file for scores
    out_f = open(score_output_file, 'w')

    # Save experiment configuration
    config_file = os.path.join(exp_dir, 'config.txt')
    with open(config_file, 'w') as cf:
        cf.write("="*60 + "\n")
        cf.write("Experiment Configuration\n")
        cf.write("="*60 + "\n")
        cf.write(f"Experiment directory: {exp_dir}\n")
        cf.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        cf.write(f"\nDataset Configuration:\n")
        cf.write(f"  dataset: {dataset_type}\n")
        cf.write(f"  vocab_size: {vocab_size}\n")
        cf.write(f"  seq_len: {seq_len}\n")
        cf.write(f"\nTraining Configuration:\n")
        cf.write(f"  batch_size: {batch_size}\n")
        cf.write(f"  lr: {lr}\n")
        cf.write(f"  lr_decay: {lr_decay}\n")
        cf.write(f"  lr_decay_end: {lr_decay_end}\n")
        cf.write(f"  lr_warmup_steps: {lr_warmup_steps}\n")
        cf.write(f"  steps: {steps}\n")
        cf.write(f"\nModel Configuration:\n")
        cf.write(f"  embed_dim: {embed_dim}\n")
        cf.write(f"  hidden_dim: {hidden_dim}\n")
        cf.write(f"  n_blocks: {n_blocks}\n")
        cf.write(f"  n_heads: {n_heads}\n")
        cf.write(f"  embedding_type: {embedding_type}\n")
        cf.write(f"  positional_encoding: {positional_encoding}\n")
        cf.write(f"\nDiffusion Configuration:\n")
        cf.write(f"  num_timesteps: {num_timesteps}\n")
        cf.write(f"  beta_start: {beta_start}\n")
        cf.write(f"  beta_end: {beta_end}\n")
        cf.write(f"\nFile Paths:\n")
        cf.write(f"  checkpoint: {checkpoint_path}\n")
        cf.write(f"  tensorboard_logs: {tensorboard_log_dir}\n")
        cf.write(f"  embed_plot: {embed_plot_path}\n")
        cf.write(f"  loss_plot: {loss_plot_path}\n")
        cf.write("="*60 + "\n")

    # Setup embedding
    if embedding_type == "onehot":
        embedding = OneHotEmbedding(vocab_size)
        embed_dim = vocab_size
        print(f"Using one-hot embeddings (embed_dim overridden to {embed_dim})")
    elif embedding_type == "unitsphere":
        embedding = UnitSphereEmbedding(vocab_size)
        embed_dim = 2  # Always 2D for unit circle
        print(f"Using unit sphere embeddings (digits uniformly on 2D circle, embed_dim overridden to {embed_dim})")
    else:
        embedding = EmbeddingMatrix(vocab_size, embed_dim)
        print(f"Using learned embeddings with embed_dim={embed_dim}")

    embedding = embedding.to(device)

    # Setup model - PASS dataset_type to model constructor
    model = SimpleDiffusionModel(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        n_heads=n_heads,
        vocab_size=vocab_size,
        seq_len=seq_len,
        positional_encoding=positional_encoding,
        dataset_type=dataset_type  # NEW: Pass dataset_type to model
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in embedding.parameters()) + \
                   sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    if resume:
        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        checkpoint_config = checkpoint.get('config', {})
        saved_positional = str(checkpoint_config.get('positional_encoding', positional_encoding)).lower()
        if saved_positional != positional_encoding:
            raise ValueError(
                "Checkpoint positional_encoding='" + saved_positional + "' does not match requested positional_encoding='" + positional_encoding + "'."
            )
        saved_embedding_type = str(checkpoint_config.get('embedding_type', embedding_type)).lower()
        if saved_embedding_type != embedding_type:
            raise ValueError(
                "Checkpoint embedding_type='" + saved_embedding_type + "' does not match requested embedding_type='" + embedding_type + "'."
            )
        embedding.load_state_dict(checkpoint['embedding_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])

    # Optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(embedding.parameters()),
        lr=lr,
        weight_decay=1e-5
    )

    # Learning rate scheduler with warmup
    total_steps = steps if steps > 0 else 1
    if lr_decay_steps is None:
        lr_decay_steps = max(total_steps - lr_warmup_steps, 1)

    def lr_lambda(step: int):
        if step < lr_warmup_steps:
            return max(step / max(lr_warmup_steps, 1), 1e-8)
        if not lr_decay:
            return 1.0
        progress = min(step - lr_warmup_steps, lr_decay_steps) / max(lr_decay_steps, 1)
        target_ratio = lr_decay_end / lr
        return target_ratio + (1.0 - target_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Move noise schedule to device
    dtype = torch.float32
    alphas_cumprod = alphas_cumprod.to(device=device, dtype=dtype)
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=device, dtype=dtype)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=device, dtype=dtype)
    gamma_table = gamma_table.to(device=device, dtype=dtype)
    gamma_prime_table = gamma_prime_table.to(device=device, dtype=dtype)
    alpha_1_scalar = alpha_1_scalar.to(device=device, dtype=dtype)
    sigma_1_scalar = sigma_1_scalar.to(device=device, dtype=dtype)

    alpha_1_tensor = alpha_1_scalar.view(1, 1, 1)
    sigma_1_tensor = sigma_1_scalar.view(1, 1, 1)
    zero_tensor = torch.tensor(0.0, device=device, dtype=dtype)
    one_tensor = torch.tensor(1.0, device=device, dtype=dtype)

    if not sampling_only:
        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        print(f"TensorBoard logging to: {tensorboard_log_dir}")

        # Log hyperparameters
        hparams = {
            'dataset': dataset_type,
            'batch_size': batch_size,
            'lr': lr,
            'lr_decay': lr_decay,
            'lr_warmup_steps': lr_warmup_steps,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'n_blocks': n_blocks,
            'n_heads': n_heads,
            'num_timesteps': num_timesteps,
            'beta_start': beta_start,
            'beta_end': beta_end,
            'embedding_type': embedding_type,
            'positional_encoding': positional_encoding,
            'vocab_size': vocab_size,
            'seq_len': seq_len,
        }
        # Add hyperparameters to TensorBoard (note: metrics will be added at the end)

        # Training loop
        print("Starting training...")
        print(f"{'Step':<10} {'Loss':<12} {'Acc@t=0':<12}")
        print("-" * 40)

        total_losses: List[float] = []
        recon_losses: List[float] = []
        diffusion_losses: List[float] = []
        prior_losses: List[float] = []

        for step in range(steps):
            # Sample batch
            indices = torch.randint(0, len(data), (batch_size,))
            x = data[indices].to(device)  # [batch, seq_len]

            # get clean embeddings
            x_embed = embedding(x)  # [batch, seq_len, embed_dim]

            # select reconstruction subset (need time = 0 to calculate reconstruction loss)
            reconst_bs = max(1, batch_size // 4)
            reconst_bs = min(reconst_bs, batch_size)
            t = torch.randint(0, num_timesteps, (batch_size,), device=device)
            t[:reconst_bs] = 0

            # noise schedule values for these timesteps
            sqrt_alpha_t = sqrt_alphas_cumprod[t][:, None, None]
            sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t][:, None, None]

            # Add noise in DDPM style: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
            noise = torch.randn_like(x_embed)
            z = sqrt_alpha_t * x_embed + sqrt_one_minus_alpha_t * noise

            # Convert discrete timestep to continuous [0, 1] for model input
            t_continuous = t.float() / denom

            # Predict logits
            logits = model(z, t_continuous)  # [batch, seq_len, vocab_size]

            # Predicted embedding reconstruction
            probs = F.softmax(logits, dim=-1)
            embedding_matrix = embedding()
            x_reconst = probs @ embedding_matrix

            # Reconstruction loss (first reconst_bs elements)
            if reconst_bs > 0:
                reconst_terms = lib_ops.cross_entropy(logits[:reconst_bs], x[:reconst_bs]).mean(dim=1)
                reconst_loss = reconst_terms.mean()
            else:
                reconst_terms = torch.empty(0, device=device)
                reconst_loss = torch.tensor(0.0, device=device)

            gamma_t = gamma_table[t]
            gamma_prime_t = gamma_prime_table[t]
            snr_prime = -torch.exp(-gamma_t) * gamma_prime_t
            diff_base = (x_embed - x_reconst).pow(2).mean(dim=1).sum(dim=1)
            diffusion_vals = -0.5 * snr_prime * diff_base
            diffusion_vals = diff_base
            diffusion_tail = diffusion_vals[reconst_bs:] if reconst_bs < batch_size else torch.empty(0, device=device)
            diffusion_loss = diffusion_tail.mean() if diffusion_tail.numel() > 0 else torch.tensor(0.0, device=device)



            # prior loss at t=1(most noisy)
            prior_loss = lib_ops.gaussian_kl(
                alpha_1_tensor * x_embed,
                sigma_1_tensor,
                zero_tensor,
                one_tensor
            ).sum(dim=2).mean()

            loss = prior_loss
            if reconst_bs > 0:
                loss = loss + reconst_loss
            if diffusion_tail.numel() > 0:
                loss = loss + diffusion_loss
            
            dispersive_loss = get_dispersion_loss(embedding().repeat(batch_size, 1))

            loss = loss + dispersive_loss * 1e1
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_losses.append(float(loss.detach()))
            recon_losses.append(float(reconst_loss.detach()))
            diffusion_losses.append(float(diffusion_loss.detach()))
            prior_losses.append(float(prior_loss.detach()))

            # Print progress and log to TensorBoard
            if step % print_freq == 0 or step == steps - 1:
                total_diffusion = diffusion_vals.mean().item()
                reconst_val = reconst_loss.item() if reconst_bs > 0 else 0.0
                diff_tail_val = diffusion_loss.item() if diffusion_tail.numel() > 0 else 0.0
                # Compute accuracy at t=0 (clean reconstruction)
                with torch.no_grad():
                    t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
                    sqrt_alpha_0 = sqrt_alphas_cumprod[t_zero][:, None, None]
                    z_clean = sqrt_alpha_0 * x_embed
                    t_zero_continuous = t_zero.float() / num_timesteps
                    logits_clean = model(z_clean, t_zero_continuous)
                    preds = logits_clean.argmax(dim=-1)
                    acc = (preds == x).float().mean().item()

                # TensorBoard logging
                writer.add_scalar('Loss/total', loss.item(), step)
                writer.add_scalar('Loss/reconstruction', reconst_val, step)
                writer.add_scalar('Loss/diffusion', diff_tail_val, step)
                writer.add_scalar('Loss/prior', prior_loss.item(), step)
                writer.add_scalar('Loss/dispersion', dispersive_loss.item(), step)
                writer.add_scalar('Loss/diffusion_mean', total_diffusion, step)
                writer.add_scalar('Metrics/accuracy_t0', acc, step)
                writer.add_scalar('Hyperparameters/learning_rate', scheduler.get_last_lr()[0], step)

                # Log embeddings periodically
                if step % (print_freq * 10) == 0:
                    emb_matrix = embedding().detach().cpu()
                    writer.add_embedding(
                        emb_matrix,
                        metadata=[str(i) for i in range(vocab_size)],
                        global_step=step,
                        tag='embeddings'
                    )

                # Log model parameter histograms periodically
                if step % (print_freq * 10) == 0:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            writer.add_histogram(f'Model/{name}', param.data, step)
                            if param.grad is not None:
                                writer.add_histogram(f'Model/{name}.grad', param.grad, step)

                print(embedding())
                print(f"{step:>6} | recon={reconst_val:.4f} diff_tail={diff_tail_val:.4f} prior={prior_loss.item():.4f} "
                      f"loss={loss.item():.4f} (diff_mean={total_diffusion:.4f}) disp_loss={dispersive_loss:.4f} acc={acc:.4f}")


            if step % 10000 == 0:
                n_samples = args.get('n_samples', 10000)
                sampling_steps = args.get('sampling_steps', num_timesteps)
                sampling_eta = args.get('sampling_eta', 0.0)
                sampling_start_t = args.get('sampling_start_t', num_timesteps - 1)
                if isinstance(sampling_start_t, float):
                    sampling_start_t = int(round(sampling_start_t))
                sampling_start_t = int(sampling_start_t)
                sampling_start_t = max(0, min(num_timesteps - 1, sampling_start_t))

                with torch.no_grad():
                    # Start from user-specified noise level
                    alpha_start = alphas_cumprod[sampling_start_t]
                    sqrt_one_minus_alpha_start = torch.sqrt(torch.clamp(1.0 - alpha_start, min=1e-8))
                    sqrt_alpha_start = torch.sqrt(alpha_start)
                    # Start from pure noise
                    z = torch.randn(n_samples, seq_len, embed_dim, device=device) * sqrt_one_minus_alpha_start
                    embedding_matrix = embedding()

                    # Load test quiz data for guided sampling
                    test_quiz = None
                    if dataset_type == 'sudoku':
                        test_quiz_path = args.get('test_quiz_path', 'data/sudoku_test.csv')
                        if os.path.exists(test_quiz_path):
                            test_quiz, _ = load_sudoku_dataset(test_quiz_path)
                            test_quiz = test_quiz[:n_samples].to(device)
                            print(f"Loaded test quiz data from {test_quiz_path} for guided sampling")
                            print(f"Quiz data shape: {test_quiz.shape}")

                    schedule = torch.linspace(float(sampling_start_t), 0.0, sampling_steps, device=device)
                    schedule = torch.round(schedule).to(torch.long)
                    schedule = torch.unique_consecutive(schedule)

                    print(f"Denoising over {sampling_steps} steps from t={sampling_start_t} to t=0...")

                    # Create reverse timestep schedule
                    for step_idx, t_discrete in enumerate(schedule.tolist()):
                        t_tensor = torch.full((n_samples,), t_discrete, dtype=torch.long, device=device)
                        t_continuous = t_tensor.float() / denom

                        logits = model(z, t_continuous)
                        probs = F.softmax(logits, dim=-1)
                        x_reconst = probs @ embedding_matrix
                        pred_tokens = logits.argmax(dim=-1)

                        ### no clamping
                        x_embed_disc = x_reconst
                        ### clamping
                        # x_embed_disc = embedding_matrix[pred_tokens]

                        # Inject ground truth quiz values at known positions
                        # if test_quiz is not None:
                        #     # Create mask for known positions (where quiz != 0)
                        #     quiz_mask = (test_quiz != 0).unsqueeze(-1)  # shape: [n_samples, seq_len, 1]
                        #     # Get ground truth embeddings for quiz values
                        #     quiz_embeddings = embedding_matrix[test_quiz]  # shape: [n_samples, seq_len, embed_dim]
                        #     # Replace denoised embeddings with ground truth at known positions
                        #     x_embed_disc = torch.where(quiz_mask, quiz_embeddings, x_embed_disc)

                        alpha_t = alphas_cumprod[t_tensor].view(n_samples, 1, 1)
                        sqrt_alpha_t = torch.sqrt(alpha_t)
                        sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
                        eps_pred = (z - sqrt_alpha_t * x_embed_disc) / (sqrt_one_minus_alpha_t + 1e-8)

                        if t_discrete > 0:
                            next_index = schedule[min(step_idx + 1, schedule.numel() - 1)].item()
                            alpha_next = alphas_cumprod[next_index].view(1, 1, 1).to(device=device, dtype=dtype)
                            alpha_next = alpha_next.expand(n_samples, 1, 1)
                            sqrt_alpha_next = torch.sqrt(alpha_next)
                            sqrt_one_minus_alpha_next = torch.sqrt(torch.clamp(1.0 - alpha_next, min=1e-8))

                            if sampling_eta > 0:
                                noise = torch.randn_like(z)
                                eps_mix = (1.0 - sampling_eta) * eps_pred + sampling_eta * noise
                            else:
                                eps_mix = eps_pred

                            z = sqrt_alpha_next * x_embed_disc + sqrt_one_minus_alpha_next * eps_mix
                        else:
                            z = x_embed_disc

                        if step_idx in [0, max(1, schedule.numel() // 4), max(1, schedule.numel() // 2),
                                        max(1, 3 * schedule.numel() // 4), schedule.numel() - 1]:
                            print(f"  Step {step_idx:3d} (t={t_discrete:4d}): Sample 1 = {pred_tokens[0].tolist()}, test_quiz = {test_quiz[0].tolist() if test_quiz is not None else 'N/A'}")

                    # Final prediction at t=0
                    t_zero = torch.zeros(n_samples, dtype=torch.long, device=device)
                    t_zero_continuous = t_zero.float() / denom
                    final_logits = model(z, t_zero_continuous)
                    final_preds = final_logits.argmax(dim=-1)

                    print("\nFinal generated sequences (after all denoising steps):")

                    # Log sample outputs to TensorBoard
                    sample_text = []
                    if dataset_type == 'sudoku':
                        for i in range(min(10, n_samples)):
                            print(f"  Sample {i+1}:")
                            grid_str = str(final_preds[i].reshape(9, 9).cpu().numpy())
                            print(final_preds[i].reshape(9, 9))
                            sample_text.append(f"Sample {i+1}:\n{grid_str}\n")
                            print()

                    elif dataset_type == 'sudoku_simple':
                        for i in range(min(20, n_samples)):
                            sample_str = f"Sample {i+1} (row): {final_preds[i].tolist()}"
                            print(f"  {sample_str}")
                            sample_text.append(sample_str)

                    elif dataset_type == 'sudoku_tiny':
                        for i in range(min(30, n_samples)):
                            sample_str = f"Sample {i+1}: {final_preds[i].tolist()}"
                            print(f"  {sample_str}")
                            sample_text.append(sample_str)

                    else:
                        for i in range(min(50, n_samples)):
                            sample_str = f"Sample {i+1}: {final_preds[i].tolist()}"
                            print(f"  {sample_str}")
                            sample_text.append(sample_str)

                    # Log samples to TensorBoard
                    if sample_text:
                        writer.add_text('Samples/generated', '\n'.join(sample_text[:10]), step)

                    # Pattern analysis
                    print("\nPattern analysis:")
                    if dataset_type == 'sudoku':
                        # For Sudoku, check if generated sequences are valid
                        def is_valid_sudoku(grid):
                            """Check if a 9x9 Sudoku grid is valid (ignoring zeros)."""
                            grid = grid.reshape(9, 9)
                            score = 0
                            
                            valid = True

                            # Check rows
                            for i in range(9):
                                row = grid[i][grid[i] != 0]
                                if len(row) != len(set(row.tolist())):
                                    valid = False
                                else:
                                    score += 1

                            # Check columns
                            for j in range(9):
                                col = grid[:, j][grid[:, j] != 0]
                                if len(col) != len(set(col.tolist())):
                                    valid = False
                                else:
                                    score += 1

                            # Check 3x3 boxes
                            for box_i in range(3):
                                for box_j in range(3):
                                    box = grid[box_i*3:(box_i+1)*3, box_j*3:(box_j+1)*3].flatten()
                                    box = box[box != 0]
                                    if len(box) != len(set(box.tolist())):
                                        valid = False
                                    else:
                                        score += 1
                            score = score / 27.0  # average per unit

                            return valid, score

                        valid_patterns = []
                        score_list = []
                        for i in range(n_samples):
                            seq = final_preds[i]
                            is_valid, score = is_valid_sudoku(seq.cpu())
                            valid_patterns.append(is_valid)
                            score_list.append(score)

                        valid_count = sum(valid_patterns)
                        print(f"\nValid Sudoku grids: {valid_count}/{n_samples}")
                        accuracy_pct = 100.0 * valid_count / n_samples if n_samples > 0 else 0.0
                        avg_score = np.mean(score_list)
                        print(f"Sampling accuracy: {accuracy_pct:.2f}%", file=out_f, flush=True)
                        print(f"Sampling score: ", avg_score, file=out_f, flush=True)

                        # TensorBoard logging
                        writer.add_scalar('Sampling/accuracy', accuracy_pct, step)
                        writer.add_scalar('Sampling/score', avg_score, step)
                        writer.add_scalar('Sampling/valid_count', valid_count, step)

                    elif dataset_type == 'sudoku_simple':
                        # Valid Sudoku row: digits 1..9 exactly once
                        valid = []
                        target_set = set(range(1, 10))
                        for i in range(n_samples):
                            row = final_preds[i].tolist()
                            is_valid = (len(row) == 9) and (set(row) == target_set)
                            valid.append(is_valid)
                        valid_count = sum(valid)
                        print(f"\nValid Sudoku first rows: {valid_count}/{n_samples}")
                        accuracy_pct = 100.0 * valid_count / n_samples if n_samples > 0 else 0.0
                        print(f"Sampling accuracy (row uniqueness 1..9): {accuracy_pct:.2f}%")

                        # TensorBoard logging
                        writer.add_scalar('Sampling/accuracy', accuracy_pct, step)
                        writer.add_scalar('Sampling/valid_count', valid_count, step)

                    elif dataset_type == 'sudoku_tiny':
                        # Valid row = permutation of {0,1,2,3}
                        target = set([0, 1, 2, 3])
                        valid = []
                        for i in range(n_samples):
                            row = final_preds[i].tolist()
                            is_perm = (len(row) == 4) and (set(row) == target) and (len(set(row)) == 4)
                            valid.append(is_perm)
                        valid_count = sum(valid)
                        print(f"\nValid permutations: {valid_count}/{n_samples}")
                        accuracy_pct = 100.0 * valid_count / n_samples if n_samples > 0 else 0.0
                        print(f"Sampling accuracy (perm of 0..3): {accuracy_pct:.2f}%")

                        # TensorBoard logging
                        writer.add_scalar('Sampling/accuracy', accuracy_pct, step)
                        writer.add_scalar('Sampling/valid_count', valid_count, step)
                    elif dataset_type == 'randompair':
                        valid_patterns = []
                        for i in range(n_samples):
                            seq = final_preds[i].tolist()
                            is_sequential = (seq[1] == (seq[0] + 1) % 10) and (seq[3] == (seq[2] + 1) % 10)
                            valid_patterns.append(is_sequential)

                        valid_count = sum(valid_patterns)
                        print(f"\nValid sequential patterns: {valid_count}/{n_samples}")
                        accuracy_pct = 100.0 * valid_count / n_samples if n_samples > 0 else 0.0
                        print(f"Sampling accuracy: {accuracy_pct:.2f}%")

                        # TensorBoard logging
                        writer.add_scalar('Sampling/accuracy', accuracy_pct, step)
                        writer.add_scalar('Sampling/valid_count', valid_count, step)
                    elif dataset_type == 'sequential':
                        # Original sequential pattern check
                        valid_patterns = []
                        for i in range(n_samples):
                            seq = final_preds[i].tolist()
                            is_sequential = all(seq[j] == (seq[0] + j) % 10 for j in range(len(seq)))
                            valid_patterns.append(is_sequential)

                        valid_count = sum(valid_patterns)
                        print(f"\nValid sequential patterns: {valid_count}/{n_samples}")
                        accuracy_pct = 100.0 * valid_count / n_samples if n_samples > 0 else 0.0
                        print(f"Sampling accuracy: {accuracy_pct:.2f}%")

                        # TensorBoard logging
                        writer.add_scalar('Sampling/accuracy', accuracy_pct, step)
                        writer.add_scalar('Sampling/valid_count', valid_count, step)

        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)

        if plot_loss_curve:
            loss_series = {
                "total": total_losses,
                "reconstruction": recon_losses,
                "diffusion": diffusion_losses,
                "prior": prior_losses,
            }
            plot_loss_series(loss_series, loss_plot_path, show=show_loss_plot)

        if checkpoint_path:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    'embedding_state_dict': embedding.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'embed_dim': embed_dim,
                        'hidden_dim': hidden_dim,
                        'n_blocks': n_blocks,
                        'n_heads': n_heads,
                        'vocab_size': vocab_size,
                        'seq_len': seq_len,
                        'positional_encoding': positional_encoding,
                        'embedding_type': embedding_type,
                    },
                },
                checkpoint_path
            )
            print(f"Saved checkpoint to {checkpoint_path}")

        # Evaluation: Test on sequences
        print("\nEvaluating on test sequences:")
        print("-" * 60)

        model.eval()
        embedding.eval()

        with torch.no_grad():
            all_correct = 0
            total_tokens = 0

            if dataset_type == 'sudoku':
                # Test on a subset of test data
                num_test_samples = min(10, len(test_data))
                test_indices = torch.randperm(len(test_data))[:num_test_samples]

                for idx in test_indices:
                    seq = test_data[idx:idx+1].to(device)           # [1, 81]
                    x_embed = embedding(seq)

                    t_test = torch.tensor([0], dtype=torch.long, device=device)
                    sqrt_alpha_test = sqrt_alphas_cumprod[t_test][:, None, None]
                    z_test = sqrt_alpha_test * x_embed
                    t_test_continuous = t_test.float() / denom

                    logits = model(z_test, t_test_continuous)
                    preds = logits[0].argmax(dim=-1)

                    correct = (preds == seq[0]).sum().item()
                    all_correct += correct
                    total_tokens += seq.shape[1]

                    print(f"Test sample {idx.item()}:")
                    print(f"Ground truth:\n{seq[0].reshape(9, 9)}")
                    print(f"Predicted:\n{preds.reshape(9, 9)}")
                    print(f"Correct digits: {correct}/{seq.shape[1]}\n")

            elif dataset_type == 'sudoku_simple':
                # First-row-only evaluation (9 digits)
                num_test_samples = min(10, len(test_data))
                test_indices = torch.randperm(len(test_data))[:num_test_samples]

                for idx in test_indices:
                    seq = test_data[idx:idx+1].to(device)           # [1, 9]
                    x_embed = embedding(seq)

                    t_test = torch.tensor([0], dtype=torch.long, device=device)
                    sqrt_alpha_test = sqrt_alphas_cumprod[t_test][:, None, None]
                    z_test = sqrt_alpha_test * x_embed
                    t_test_continuous = t_test.float() / denom

                    logits = model(z_test, t_test_continuous)
                    preds = logits[0].argmax(dim=-1)

                    correct = (preds == seq[0]).sum().item()
                    all_correct += correct
                    total_tokens += seq.shape[1]

                    print(f"Test sample {idx.item()} first row:")
                    print(f"Ground truth: {seq[0].tolist()}")
                    print(f"Predicted   : {preds.tolist()}")
                    print(f"Correct digits: {correct}/{seq.shape[1]}\n")

            elif dataset_type == 'sudoku_tiny':
                # Evaluate on the tiny set (up to 24 rows)
                num_test = min(len(test_data), 24)
                for i in range(num_test):
                    seq = test_data[i:i+1].to(device)  # [1, 4]
                    x_embed = embedding(seq)

                    t_test = torch.tensor([0], dtype=torch.long, device=device)
                    sqrt_alpha_test = sqrt_alphas_cumprod[t_test][:, None, None]
                    z_test = sqrt_alpha_test * x_embed
                    t_test_continuous = t_test.float() / denom

                    logits = model(z_test, t_test_continuous)
                    preds = logits[0].argmax(dim=-1)

                    correct = (preds == seq[0]).sum().item()
                    all_correct += correct
                    total_tokens += seq.shape[1]

                    print(f"GT: {seq[0].tolist()}  |  Pred: {preds.tolist()}  |  correct {correct}/4")

            elif dataset_type == 'randompair':
                pass
            elif dataset_type == 'sequential':
                # Original 4-digit evaluation
                for i in range(10):
                    seq = torch.tensor([[i, (i+1)%10, (i+2)%10, (i+3)%10]], device=device)
                    x_embed = embedding(seq)

                    t_test = torch.tensor([0], dtype=torch.long, device=device)
                    sqrt_alpha_test = sqrt_alphas_cumprod[t_test][:, None, None]
                    z_test = sqrt_alpha_test * x_embed
                    t_test_continuous = t_test.float() / denom

                    logits = model(z_test, t_test_continuous)
                    probs = F.softmax(logits[0], dim=-1)
                    preds = logits[0].argmax(dim=-1)

                    correct = (preds == seq[0]).sum().item()
                    all_correct += correct
                    total_tokens += seq.shape[1]

                    print(f"Input: {seq[0].tolist()}")
                    print(f"Predicted: {preds.tolist()}")
                    print(f"Match: {preds.tolist() == seq[0].tolist()}")

                    for pos in range(4):
                        true_token = seq[0, pos].item()
                        pred_token = preds[pos].item()
                        true_prob = probs[pos, true_token].item()
                        pred_prob = probs[pos, pred_token].item()
                        match = "✓" if true_token == pred_token else "✗"
                        print(f"  Pos {pos}: True={true_token} (p={true_prob:.4f}), "
                              f"Pred={pred_token} (p={pred_prob:.4f}) {match}")
                    print()
            else:
                raise Exception
            accuracy = all_correct / total_tokens
            print(f"Overall Accuracy: {accuracy:.2%} ({all_correct}/{total_tokens} tokens correct)")
            print(embedding())

            # TensorBoard logging for final evaluation
            writer.add_scalar('Evaluation/final_accuracy', accuracy * 100, steps)
            writer.add_scalar('Evaluation/correct_tokens', all_correct, steps)
            writer.add_scalar('Evaluation/total_tokens', total_tokens, steps)

            # Log hyperparameters with final metrics
            metric_dict = {
                'hparam/final_accuracy': accuracy * 100,
                'hparam/final_loss': total_losses[-1] if total_losses else 0.0,
            }
            writer.add_hparams(hparams, metric_dict)

            # Log embedding visualization
            visualize_embeddings(embedding(), embed_plot_path)
            if os.path.exists(embed_plot_path):
                try:
                    import matplotlib.pyplot as plt
                    img = plt.imread(embed_plot_path)
                    writer.add_image('Embeddings/visualization', img, steps, dataformats='HWC')
                except Exception as e:
                    print(f"Could not log embedding image to TensorBoard: {e}")

        print("\n" + "="*60)
        print("Test complete!")
        print("="*60)

        # Close TensorBoard writer
        writer.close()
        print(f"TensorBoard logs saved to: {tensorboard_log_dir}")
    else:
        if not load_checkpoint_path:
            raise ValueError("sampling_only=True requires 'load_checkpoint_path' to be specified")
        if not os.path.exists(load_checkpoint_path):
            raise FileNotFoundError(f"Checkpoint '{load_checkpoint_path}' not found for sampling-only mode")

        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        checkpoint_config = checkpoint.get('config', {})
        saved_positional = str(checkpoint_config.get('positional_encoding', positional_encoding)).lower()
        if saved_positional != positional_encoding:
            raise ValueError(
                "Checkpoint positional_encoding='" + saved_positional + "' does not match requested positional_encoding='" + positional_encoding + "'."
            )
        saved_embedding_type = str(checkpoint_config.get('embedding_type', embedding_type)).lower()
        if saved_embedding_type != embedding_type:
            raise ValueError(
                "Checkpoint embedding_type='" + saved_embedding_type + "' does not match requested embedding_type='" + embedding_type + "'."
            )
        embedding.load_state_dict(checkpoint['embedding_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        embedding.eval()

        print("\nSampling-only mode: loaded checkpoint and skipping training/evaluation.")
        print(f"Loaded weights from {load_checkpoint_path}")

    # Optional: Test sampling (denoising from pure noise)
    run_sampling = sampling_only or args.get('test_sampling', True)
    if run_sampling:
        print("\nTesting sampling from pure noise...")
        print("-" * 60)
        print("Note: Sampling uses progressive denoising over multiple steps")
        print(f"Starting from pure Gaussian noise and denoising to clean data")
        print()

        n_samples = args.get('n_samples', 10000)
        sampling_steps = args.get('sampling_steps', num_timesteps)
        sampling_eta = args.get('sampling_eta', 0.0)
        sampling_start_t = args.get('sampling_start_t', num_timesteps - 1)
        if isinstance(sampling_start_t, float):
            sampling_start_t = int(round(sampling_start_t))
        sampling_start_t = int(sampling_start_t)
        sampling_start_t = max(0, min(num_timesteps - 1, sampling_start_t))

        with torch.no_grad():
            # Start from user-specified noise level
            alpha_start = alphas_cumprod[sampling_start_t]
            sqrt_one_minus_alpha_start = torch.sqrt(torch.clamp(1.0 - alpha_start, min=1e-8))
            sqrt_alpha_start = torch.sqrt(alpha_start)
            # Start from pure noise
            z = torch.randn(n_samples, seq_len, embed_dim, device=device) * sqrt_one_minus_alpha_start
            embedding_matrix = embedding()

            # Load test quiz data for guided sampling
            test_quiz = None
            if dataset_type == 'sudoku':
                test_quiz_path = args.get('test_quiz_path', 'data/sudoku_test.csv')
                if os.path.exists(test_quiz_path):
                    test_quiz, _ = load_sudoku_dataset(test_quiz_path)
                    test_quiz = test_quiz[:n_samples].to(device)
                    print(f"Loaded test quiz data from {test_quiz_path} for guided sampling")
                    print(f"Quiz data shape: {test_quiz.shape}")

            schedule = torch.linspace(float(sampling_start_t), 0.0, sampling_steps, device=device)
            schedule = torch.round(schedule).to(torch.long)
            schedule = torch.unique_consecutive(schedule)

            print(f"Denoising over {sampling_steps} steps from t={sampling_start_t} to t=0...")

            # Create reverse timestep schedule
            for step_idx, t_discrete in enumerate(schedule.tolist()):
                t_tensor = torch.full((n_samples,), t_discrete, dtype=torch.long, device=device)
                t_continuous = t_tensor.float() / denom

                logits = model(z, t_continuous)
                probs = F.softmax(logits, dim=-1)
                x_reconst = probs @ embedding_matrix
                pred_tokens = logits.argmax(dim=-1)

                ### no clamping
                x_embed_disc = x_reconst
                ### clamping
                # x_embed_disc = embedding_matrix[pred_tokens]

                # Inject ground truth quiz values at known positions
                # if test_quiz is not None:
                #     # Create mask for known positions (where quiz != 0)
                #     quiz_mask = (test_quiz != 0).unsqueeze(-1)  # shape: [n_samples, seq_len, 1]
                #     # Get ground truth embeddings for quiz values
                #     quiz_embeddings = embedding_matrix[test_quiz]  # shape: [n_samples, seq_len, embed_dim]
                #     # Replace denoised embeddings with ground truth at known positions
                #     x_embed_disc = torch.where(quiz_mask, quiz_embeddings, x_embed_disc)

                alpha_t = alphas_cumprod[t_tensor].view(n_samples, 1, 1)
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
                eps_pred = (z - sqrt_alpha_t * x_embed_disc) / (sqrt_one_minus_alpha_t + 1e-8)

                if t_discrete > 0:
                    next_index = schedule[min(step_idx + 1, schedule.numel() - 1)].item()
                    alpha_next = alphas_cumprod[next_index].view(1, 1, 1).to(device=device, dtype=dtype)
                    alpha_next = alpha_next.expand(n_samples, 1, 1)
                    sqrt_alpha_next = torch.sqrt(alpha_next)
                    sqrt_one_minus_alpha_next = torch.sqrt(torch.clamp(1.0 - alpha_next, min=1e-8))

                    if sampling_eta > 0:
                        noise = torch.randn_like(z)
                        eps_mix = (1.0 - sampling_eta) * eps_pred + sampling_eta * noise
                    else:
                        eps_mix = eps_pred

                    z = sqrt_alpha_next * x_embed_disc + sqrt_one_minus_alpha_next * eps_mix
                else:
                    z = x_embed_disc

                if step_idx in [0, max(1, schedule.numel() // 4), max(1, schedule.numel() // 2),
                                max(1, 3 * schedule.numel() // 4), schedule.numel() - 1]:
                    print(f"  Step {step_idx:3d} (t={t_discrete:4d}): Sample 1 = {pred_tokens[0].tolist()}, test_quiz = {test_quiz[0].tolist() if test_quiz is not None else 'N/A'}")

            # Final prediction at t=0
            t_zero = torch.zeros(n_samples, dtype=torch.long, device=device)
            t_zero_continuous = t_zero.float() / denom
            final_logits = model(z, t_zero_continuous)
            final_preds = final_logits.argmax(dim=-1)

            print("\nFinal generated sequences (after all denoising steps):")
            if dataset_type == 'sudoku':
                for i in range(min(10, n_samples)):
                    print(f"  Sample {i+1}:")
                    print(final_preds[i].reshape(9, 9))
                    print()

            elif dataset_type == 'sudoku_simple':
                for i in range(min(20, n_samples)):
                    print(f"  Sample {i+1} (row): {final_preds[i].tolist()}")

            elif dataset_type == 'sudoku_tiny':
                for i in range(min(30, n_samples)):
                    print(f"  Sample {i+1}: {final_preds[i].tolist()}")

            else:
                for i in range(min(50, n_samples)):
                    print(f"  Sample {i+1}: {final_preds[i].tolist()}")

            # Pattern analysis
            print("\nPattern analysis:")
            if dataset_type == 'sudoku':
                # For Sudoku, check if generated sequences are valid
                def is_valid_sudoku(grid):
                    """Check if a 9x9 Sudoku grid is valid (ignoring zeros)."""
                    grid = grid.reshape(9, 9)
                    score = 0
                    
                    valid = True

                    # Check rows
                    for i in range(9):
                        row = grid[i][grid[i] != 0]
                        if len(row) != len(set(row.tolist())):
                            valid = False
                        else:
                            score += 1

                    # Check columns
                    for j in range(9):
                        col = grid[:, j][grid[:, j] != 0]
                        if len(col) != len(set(col.tolist())):
                            valid = False
                        else:
                            score += 1

                    # Check 3x3 boxes
                    for box_i in range(3):
                        for box_j in range(3):
                            box = grid[box_i*3:(box_i+1)*3, box_j*3:(box_j+1)*3].flatten()
                            box = box[box != 0]
                            if len(box) != len(set(box.tolist())):
                                valid = False
                            else:
                                score += 1
                    score = score / 27.0  # average per unit

                    return valid, score

                valid_patterns = []
                score_list = []
                for i in range(n_samples):
                    seq = final_preds[i]
                    is_valid, score = is_valid_sudoku(seq.cpu())
                    valid_patterns.append(is_valid)
                    score_list.append(score)

                valid_count = sum(valid_patterns)
                print(f"\nValid Sudoku grids: {valid_count}/{n_samples}")
                accuracy_pct = 100.0 * valid_count / n_samples if n_samples > 0 else 0.0
                print(f"Sampling accuracy: {accuracy_pct:.2f}%")
                print(f"Sampling score: ", np.mean(score_list))

            elif dataset_type == 'sudoku_simple':
                # Valid Sudoku row: digits 1..9 exactly once
                valid = []
                target_set = set(range(1, 10))
                for i in range(n_samples):
                    row = final_preds[i].tolist()
                    is_valid = (len(row) == 9) and (set(row) == target_set)
                    valid.append(is_valid)
                valid_count = sum(valid)
                print(f"\nValid Sudoku first rows: {valid_count}/{n_samples}")
                accuracy_pct = 100.0 * valid_count / n_samples if n_samples > 0 else 0.0
                print(f"Sampling accuracy (row uniqueness 1..9): {accuracy_pct:.2f}%")

            elif dataset_type == 'sudoku_tiny':
                # Valid row = permutation of {0,1,2,3}
                target = set([0, 1, 2, 3])
                valid = []
                for i in range(n_samples):
                    row = final_preds[i].tolist()
                    is_perm = (len(row) == 4) and (set(row) == target) and (len(set(row)) == 4)
                    valid.append(is_perm)
                valid_count = sum(valid)
                print(f"\nValid permutations: {valid_count}/{n_samples}")
                accuracy_pct = 100.0 * valid_count / n_samples if n_samples > 0 else 0.0
                print(f"Sampling accuracy (perm of 0..3): {accuracy_pct:.2f}%")
            elif dataset_type == 'randompair':
                valid_patterns = []
                for i in range(n_samples):
                    seq = final_preds[i].tolist()
                    is_sequential = (seq[1] == (seq[0] + 1) % 10) and (seq[3] == (seq[2] + 1) % 10)
                    valid_patterns.append(is_sequential)

                valid_count = sum(valid_patterns)
                print(f"\nValid sequential patterns: {valid_count}/{n_samples}")
                accuracy_pct = 100.0 * valid_count / n_samples if n_samples > 0 else 0.0
                print(f"Sampling accuracy: {accuracy_pct:.2f}%")
            elif dataset_type == 'sequential':
                # Original sequential pattern check
                valid_patterns = []
                for i in range(n_samples):
                    seq = final_preds[i].tolist()
                    is_sequential = all(seq[j] == (seq[0] + j) % 10 for j in range(len(seq)))
                    valid_patterns.append(is_sequential)

                valid_count = sum(valid_patterns)
                print(f"\nValid sequential patterns: {valid_count}/{n_samples}")
                accuracy_pct = 100.0 * valid_count / n_samples if n_samples > 0 else 0.0
                print(f"Sampling accuracy: {accuracy_pct:.2f}%")

    # Close output file
    out_f.close()
    print(f"\nAll outputs saved to: {exp_dir}")


if __name__ == '__main__':
    fire.Fire(main)
"""
Diffusion Language Models for Discrete Data

Datasets supported:
- sequential:    sequences length 4, [i, i+1, i+2, i+3] [~100%]
- sudoku:        full Sudoku solutions (9x9 grids)

Model types supported:
- continuous:    Continuous diffusion in embedding space (default)
- masked:        Masked diffusion model (MDM) - simpler discrete approach
- combined:      Combination of continuous + masked diffusion (CADD)
- ccdd:          Continuous-Categorical Dual Diffusion with separate latents

Noise schedules (for continuous diffusion):
- ddpm:          DDPM-style linear beta schedule (default)
- sqrt_linear:   alpha_t = sqrt(1-t) schedule (matches discrete diffusion SNR)

Usage examples:
  # Continuous diffusion on sequential data
  python main.py --dataset sequential --model_type continuous

  # Continuous diffusion with sqrt(1-t) schedule
  python main.py --dataset sequential --model_type continuous --noise_schedule sqrt_linear

  # Masked diffusion on sequential data
  python main.py --dataset sequential --model_type masked --steps 10000

  # Masked diffusion on sudoku
  python main.py --dataset sudoku --model_type masked --batch_size 256 --steps 50000
"""

import builtins
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
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
from lib import ops as lib_ops
from ipdb import iex

# Import models and datasets from separate files
from model import (
    EmbeddingMatrix, OneHotEmbedding, UnitSphereEmbedding,
    SimpleDiffusionModel, MaskedPredictor, CCDDModel, llada_mask,
    add_gumbel_noise, get_num_transfer_tokens
)
from dataset import create_simple_dataset
# from dataset import load_sudoku_dataset_npy as load_sudoku_dataset
from dataset import load_sudoku_dataset_csv as load_sudoku_dataset
# from dataset import load_sudoku_dataset


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

    # Find and backup all .py files recursively (excluding experiments/ and common skips)
    skip_tokens = ["venv", "env", "__pycache__", ".git", "site-packages", "experiments"]
    py_files: List[str] = []
    for root, dirs, files in os.walk("."):
        # Prune directories we never want to enter
        dirs[:] = [
            d for d in dirs
            if not any(skip in os.path.join(root, d) for skip in skip_tokens)
        ]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            full_path = os.path.join(root, fname)
            # Skip any path containing excluded tokens
            if any(skip in full_path for skip in skip_tokens):
                continue
            rel_path = os.path.relpath(full_path, ".")
            py_files.append(rel_path)

    for py_file in py_files:
        # Skip files in virtual environments, __pycache__, etc.
        if any(skip in py_file for skip in ["venv", "env", "__pycache__", ".git", "site-packages"]):
            continue

        # Create subdirectories in backup if needed
        dest_path = os.path.join(backup_dir, py_file)
        dest_dir = os.path.dirname(dest_path)
        

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


def is_valid_sudoku(grid):
    """
    Validate a Sudoku grid and compute a validation score.

    Args:
        grid: numpy array or torch tensor of shape (81,) or (9, 9) containing digits 0-9

    Returns:
        valid (bool): True if the grid is valid (no duplicates in rows/cols/boxes)
        score (float): Fraction of valid checks (0-1), where 27 checks total (9 rows + 9 cols + 9 boxes)
    """
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

    score = score / 27.0
    return valid, score


def evaluate_sudoku_samples(final_preds, n_samples, mode_str="Sampling"):
    """
    Evaluate Sudoku samples and compute validation metrics.

    Args:
        final_preds: Tensor of predicted Sudoku grids (n_samples, 81)
        n_samples: Number of samples to evaluate
        mode_str: String prefix for print statements (e.g., "Generation", "Completion")

    Returns:
        dict with keys: valid_count, accuracy_pct, avg_score, valid_patterns, score_list
    """
    valid_patterns = []
    score_list = []

    for i in range(n_samples):
        is_valid, score = is_valid_sudoku(final_preds[i].cpu())
        valid_patterns.append(is_valid)
        score_list.append(score)

    valid_count = sum(valid_patterns)
    accuracy_pct = 100.0 * valid_count / n_samples if n_samples > 0 else 0.0
    avg_score = np.mean(score_list)

    print(f"\n{mode_str} - Valid Sudoku grids: {valid_count}/{n_samples}")
    print(f"{mode_str} - Accuracy: {accuracy_pct:.2f}%")
    print(f"{mode_str} - Score: {avg_score:.4f}")

    return {
        'valid_count': valid_count,
        'accuracy_pct': accuracy_pct,
        'avg_score': avg_score,
        'valid_patterns': valid_patterns,
        'score_list': score_list
    }


def evaluate_sequential_samples(final_preds, n_samples, mode_str="Sampling"):
    """
    Evaluate sequential pattern samples (checks if sequence is [i, i+1, i+2, i+3]).

    Args:
        final_preds: Tensor of predicted sequences (n_samples, seq_len)
        n_samples: Number of samples to evaluate
        mode_str: String prefix for print statements

    Returns:
        dict with keys: valid_count, accuracy_pct, valid_patterns
    """
    valid_patterns = []

    for i in range(n_samples):
        seq = final_preds[i].tolist()
        is_sequential = all(seq[j] == (seq[0] + j) % 10 for j in range(len(seq)))
        valid_patterns.append(is_sequential)

    valid_count = sum(valid_patterns)
    accuracy_pct = 100.0 * valid_count / n_samples if n_samples > 0 else 0.0

    print(f"\n{mode_str} - Valid sequential patterns: {valid_count}/{n_samples}")
    print(f"{mode_str} - Accuracy: {accuracy_pct:.2f}%")

    return {
        'valid_count': valid_count,
        'accuracy_pct': accuracy_pct,
        'valid_patterns': valid_patterns
    }


def display_sudoku_samples(final_preds, n_samples, quiz_data=None, max_display=5):
    """
    Display Sudoku samples (for generation or completion).

    Args:
        final_preds: Tensor of predicted Sudoku grids (n_samples, 81)
        n_samples: Total number of samples available
        quiz_data: Optional quiz data for completion mode (n_samples, 81)
        max_display: Maximum number of samples to display
    """
    display_count = min(max_display, n_samples)

    if quiz_data is not None:
        # Completion mode: show quiz and prediction side by side
        for i in range(display_count):
            quiz_grid = quiz_data[i].reshape(9, 9).cpu().numpy()
            pred_grid = final_preds[i].reshape(9, 9).cpu().numpy()
            print(f"\nSample {i+1}:")
            print("Quiz:                Prediction:")
            for row_idx in range(9):
                quiz_row = ' '.join([str(int(v)) if v != 0 else '.' for v in quiz_grid[row_idx]])
                pred_row = ' '.join([str(int(v)) for v in pred_grid[row_idx]])
                print(f"{quiz_row}    {pred_row}")
            print()
    else:
        # Generation mode: show prediction only
        for i in range(display_count):
            print(f"\nSample {i+1}:")
            print(final_preds[i].reshape(9, 9))
            print()


def display_sequential_samples(final_preds, n_samples, max_display=50):
    """
    Display sequential pattern samples.

    Args:
        final_preds: Tensor of predicted sequences (n_samples, seq_len)
        n_samples: Total number of samples available
        max_display: Maximum number of samples to display
    """
    display_count = min(max_display, n_samples)
    for i in range(display_count):
        print(f"Sample {i+1}: {final_preds[i].tolist()}")


def evaluate_and_display_sudoku(final_preds, n_samples, mode_str, writer=None, step=None,
                                prefix: Optional[str] = None, quiz_data=None, max_display: int = 5):
    """
    Display Sudoku samples, compute metrics, and optionally log them to TensorBoard.
    """
    display_sudoku_samples(final_preds, n_samples, quiz_data=quiz_data, max_display=max_display)
    results = evaluate_sudoku_samples(final_preds, n_samples, mode_str=mode_str)

    if writer is not None and step is not None:
        metric_prefix = prefix or mode_str.lower()
        writer.add_scalar(f'Sampling/{metric_prefix}_accuracy', results['accuracy_pct'], step)
        writer.add_scalar(f'Sampling/{metric_prefix}_score', results['avg_score'], step)
        writer.add_scalar(f'Sampling/{metric_prefix}_valid_count', results['valid_count'], step)

    return results


def evaluate_and_display_sequential(final_preds, n_samples, mode_str, writer=None, step=None,
                                    prefix: Optional[str] = None, max_display: int = 50):
    """
    Display sequential samples, compute metrics, and optionally log them to TensorBoard.
    """
    display_sequential_samples(final_preds, n_samples, max_display=max_display)
    results = evaluate_sequential_samples(final_preds, n_samples, mode_str=mode_str)

    if writer is not None and step is not None:
        metric_prefix = prefix or mode_str.lower()
        writer.add_scalar(f'Sampling/{metric_prefix}_accuracy', results['accuracy_pct'], step)
        writer.add_scalar(f'Sampling/{metric_prefix}_valid_count', results['valid_count'], step)

    return results


@iex
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
    sudoku_train_path = args.get('sudoku_train_path', 'data_vmd/sudoku_train.csv')
    sudoku_test_path = args.get('sudoku_test_path', 'data_vmd/sudoku_test.csv')

    # Model selection
    model_type = str(args.get('model_type', 'continuous')).lower()  # 'continuous' or 'masked'

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
    transformer_block_type = str(args.get('transformer_block_type', 'simple')).lower()  # 'simple' or 'gpt2'
    repae = _coerce_bool(args.get('repae', False), False)  # REPAE option
    combined_coef = float(args.get('combined_coef', 1.0))  # Coefficient for z_t in combined model (0=pure discrete, 1=full CADD)
    ccdd_continuous_coef = float(args.get('ccdd_continuous_coef', 1.0))  # Coefficient for z_t in CCDD model (0=discrete-only, 1=full continuous)
    dropout_rate = float(args.get('dropout_rate', 0.0))  # Dropout rate for continuous latents (0.0=no dropout, 0.1=10% dropout)
    cfg_scale = float(args.get('cfg_scale', 0.0))  # Classifier-Free Guidance scale (0.0=no CFG, >0=apply guidance)
    combine_method = str(args.get('combine_method', 'add')).lower()  # How to combine z_disc and z_t: 'add' or 'concat'
    if combine_method not in ['add', 'concat']:
        raise ValueError(f"combine_method must be 'add' or 'concat', got '{combine_method}'")
    sampling_only = args.get('sampling_only', False)
    resume = args.get('resume', False)

    if load_checkpoint_path is not None:
        resume = True
        
    if resume is True and load_checkpoint_path is None:
        load_checkpoint_path = checkpoint_path

    # Noise schedule parameters
    num_timesteps = args.get('num_timesteps', 1000)
    noise_schedule = args.get('noise_schedule', 'ddpm').lower()  # 'ddpm' or 'sqrt_linear'
    beta_start = args.get('beta_start', 0.0001)
    beta_end = args.get('beta_end', 0.02)

    # Precompute noise schedule
    if noise_schedule == 'sqrt_linear':
        # alpha_t = sqrt(1-t) schedule (matches discrete diffusion SNR)
        t = torch.linspace(0, 1, num_timesteps, device='cpu')
        alphas_cumprod = 1.0 - t
        
    elif noise_schedule == 'ddpm':
        # DDPM-style linear beta schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device='cpu')
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
    else:
        raise ValueError(f"Unknown noise_schedule: {noise_schedule}. Use 'ddpm' or 'sqrt_linear'.")

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

    elif dataset_type == 'sequential':
        # original 4-token toy sequence
        vocab_size = 10
        seq_len = 4
        data = None
        test_data = None
        if not sampling_only:
            data = create_simple_dataset()

    else:
        raise Exception

    # Open output file for scores and mirror future prints to it
    out_f = open(score_output_file, 'w')
    builtin_print = builtins.print

    def tee_print(*args, **kwargs):
        """Write messages to stdout and the experiment output file."""
        builtin_print(*args, **kwargs)
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        message = sep.join(str(a) for a in args) + end
        try:
            out_f.write(message)
            out_f.flush()
        except Exception as e:
            # Fallback to stdout only if writing fails
            builtin_print(f"[warning] Failed to write to output file: {e}")

    builtins.print = tee_print

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
    print(f"noise_schedule: {noise_schedule}")
    if noise_schedule == 'ddpm':
        print(f"beta_start: {beta_start}, beta_end: {beta_end}")
    elif noise_schedule == 'sqrt_linear':
        print(f"alpha_t = sqrt(1-t) schedule (SNR matches discrete diffusion)")
    print(f"embedding_type: {embedding_type}")
    print(f"positional_encoding: {positional_encoding}")
    print(f"transformer_block_type: {transformer_block_type}")
    print(f"repae: {repae}")
    if model_type == 'combined':
        print(f"combined_coef: {combined_coef} (0=pure discrete, 1=full CADD)")
        print(f"dropout_rate: {dropout_rate} (dropout for continuous latents)")
        print(f"cfg_scale: {cfg_scale} (classifier-free guidance scale)")
        print(f"combine_method: {combine_method} (add or concat)")
    if model_type == 'ccdd':
        print(f"ccdd_continuous_coef: {ccdd_continuous_coef} (0=discrete-only, 1=full continuous)")
    print("="*60)
    print()

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
        cf.write(f"  transformer_block_type: {transformer_block_type}\n")
        cf.write(f"  repae: {repae}\n")
        if model_type == 'combined':
            cf.write(f"  combined_coef: {combined_coef}\n")
            cf.write(f"  dropout_rate: {dropout_rate}\n")
            cf.write(f"  combine_method: {combine_method}\n")
        if model_type == 'ccdd':
            cf.write(f"  ccdd_continuous_coef: {ccdd_continuous_coef}\n")
        cf.write(f"\nDiffusion Configuration:\n")
        cf.write(f"  num_timesteps: {num_timesteps}\n")
        cf.write(f"  noise_schedule: {noise_schedule}\n")
        if noise_schedule == 'ddpm':
            cf.write(f"  beta_start: {beta_start}\n")
            cf.write(f"  beta_end: {beta_end}\n")
        elif noise_schedule == 'sqrt_linear':
            cf.write(f"  alpha_t = sqrt(1-t) schedule\n")
        cf.write(f"\nFile Paths:\n")
        cf.write(f"  checkpoint: {checkpoint_path}\n")
        cf.write(f"  tensorboard_logs: {tensorboard_log_dir}\n")
        cf.write(f"  embed_plot: {embed_plot_path}\n")
        cf.write(f"  loss_plot: {loss_plot_path}\n")
        cf.write("="*60 + "\n")

    ### 2. Setup model based on model_type
    if model_type == 'masked':
        # Masked Diffusion Model doesn't use separate embedding
        embedding = None
        model = MaskedPredictor(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_blocks,
            positional_encoding=positional_encoding,
            dataset_type=dataset_type,
            transformer_block_type=transformer_block_type
        ).to(device)
        print(f"Using Masked Diffusion Model (MDM)")
        print(f"  Architecture: SAME as Continuous Model")
        print(f"  vocab_size: {vocab_size}, seq_len: {seq_len}")
        print(f"  embed_dim: {embed_dim}, hidden_dim: {hidden_dim}")
        print(f"  n_heads: {n_heads}, n_layers: {n_blocks}")
        print(f"  positional_encoding: {positional_encoding}")
        print(f"  transformer_block_type: {transformer_block_type}")
    elif model_type == 'continuous':
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

        # Setup continuous diffusion model
        model = SimpleDiffusionModel(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            n_heads=n_heads,
            vocab_size=vocab_size,
            seq_len=seq_len,
            positional_encoding=positional_encoding,
            dataset_type=dataset_type,
            transformer_block_type=transformer_block_type,  # Pass block type to model
            enable_repae=repae  # Enable REPAE hooks if requested
        ).to(device)
        print(f"Using Continuous Diffusion Model")
    elif model_type == 'combined':
        embedding = None

        model = MaskedPredictor(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_blocks,
            positional_encoding=positional_encoding,
            dataset_type=dataset_type,
            transformer_block_type=transformer_block_type,
            combine_method=combine_method
        ).to(device)
        print(f"Using Combined Diffusion Model (CADD)")
        print(f"  vocab_size: {vocab_size}, seq_len: {seq_len}")
        print(f"  embed_dim: {embed_dim}, hidden_dim: {hidden_dim}")
        input_dim = 2 * embed_dim if combine_method == 'concat' else embed_dim
        print(f"  input_dim: {input_dim} (combine_method={combine_method})")
        print(f"  n_heads: {n_heads}, n_layers: {n_blocks}")
        print(f"  positional_encoding: {positional_encoding}")
        print(f"  transformer_block_type: {transformer_block_type}")
        # raise NotImplementedError("Combined model type is not yet implemented.")
    elif model_type == 'ccdd':
        # CCDD Model: Continuous-Categorical Dual Diffusion
        # Uses separate latent space (not derived from token embeddings)
        embedding = None

        # Get latent dimension (default to embed_dim for Option 1)
        latent_dim = args.get('latent_dim', embed_dim)
        if latent_dim != embed_dim:
            print(f"WARNING: latent_dim={latent_dim} != embed_dim={embed_dim}")
            print(f"         Setting latent_dim=embed_dim for proper z_0 encoding")
            latent_dim = embed_dim

        model = CCDDModel(
            vocab_size=vocab_size,
            seq_len=seq_len,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_blocks,
            positional_encoding=positional_encoding,
            dataset_type=dataset_type,
            transformer_block_type=transformer_block_type
        ).to(device)
        print(f"Using CCDD Model (Continuous-Categorical Dual Diffusion)")
        print(f"  vocab_size: {vocab_size}, seq_len: {seq_len}")
        print(f"  latent_dim: {latent_dim} (= embed_dim for z_0 encoding)")
        print(f"  embed_dim: {embed_dim}, hidden_dim: {hidden_dim}")
        print(f"  n_heads: {n_heads}, n_layers: {n_blocks}")
        print(f"  positional_encoding: {positional_encoding}")
        print(f"  transformer_block_type: {transformer_block_type}")
    # Print REPAE status
    if repae:
        print("\n" + "="*60)
        print("="*60)
        print(f"Will capture activations at {model.get_num_layers() + 1} positions:")
        print(f"  - Layer 0: After input projection + positional + time embedding")
        for i in range(model.get_num_layers()):
            print(f"  - Layer {i+1}: After transformer block {i}")
        print("="*60 + "\n")


    # Helper function to get embedding matrix
    def get_embedding_matrix():
        """Get the full embedding matrix"""
        if embedding is not None:
            return embedding()
        else:
            # For masked model, return the embedding from the model
            return model.embed.weight[:-1, :]  # Exclude mask token

    # Count parameters
    if embedding is not None:
        total_params = sum(p.numel() for p in embedding.parameters()) + \
                       sum(p.numel() for p in model.parameters())
    else:
        total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    if resume:
        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        checkpoint_config = checkpoint.get('config', {})

        # For continuous model, check config compatibility
        if model_type == 'continuous':
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
            saved_block_type = str(checkpoint_config.get('transformer_block_type', transformer_block_type)).lower()
            if saved_block_type != transformer_block_type:
                raise ValueError(
                    "Checkpoint transformer_block_type='" + saved_block_type + "' does not match requested transformer_block_type='" + transformer_block_type + "'."
                )

            embedding.load_state_dict(checkpoint['embedding_state_dict'])
            model.load_state_dict(checkpoint['model_state_dict'])
        elif model_type == 'masked' or model_type == 'combined' or model_type == 'ccdd':
            # For masked/combined/ccdd model, validate config if available
            if model_type == 'combined' and 'config' in checkpoint:
                saved_combine_method = str(checkpoint_config.get('combine_method', 'add')).lower()
                if saved_combine_method != combine_method:
                    raise ValueError(
                        f"Checkpoint combine_method='{saved_combine_method}' does not match requested combine_method='{combine_method}'. "
                        f"Model architecture differs between add and concat modes."
                    )
            if model_type == 'ccdd' and 'config' in checkpoint:
                saved_latent_dim = checkpoint_config.get('latent_dim', embed_dim)
                if saved_latent_dim != model.latent_dim:
                    raise ValueError(
                        f"Checkpoint latent_dim={saved_latent_dim} does not match model latent_dim={model.latent_dim}. "
                        f"CCDD model architecture requires matching latent dimensions."
                    )
            # Load model state
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        else:
            raise Exception("Unsupported model type for loading checkpoint.")

    ### 3. Optimizers - setup based on model type
    if model_type == 'masked' or model_type == 'combined' or model_type == 'ccdd':
        # Masked/Combined/CCDD model: single optimizer for all parameters
        optimizer_model = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )
        optimizer_embedding = None  # No separate embedding optimizer
    elif model_type == "continuous":
        # Continuous model: separate optimizers for model and embedding
        optimizer_model = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )
        optimizer_embedding = optim.AdamW(
            embedding.parameters(),
            lr=lr,
            weight_decay=1e-5
        )
    else:
        raise Exception("Unsupported model type for optimizer setup.")

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

    # Create schedulers
    scheduler_model = optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda)
    if optimizer_embedding is not None:
        scheduler_embedding = optim.lr_scheduler.LambdaLR(optimizer_embedding, lr_lambda)
    else:
        scheduler_embedding = None

    # Load optimizer and scheduler states if resuming
    if resume:
        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        # Load optimizer states if they exist in checkpoint
        if 'optimizer_model_state_dict' in checkpoint:
            optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
            print("Loaded model optimizer state from checkpoint")
        if 'optimizer_embedding_state_dict' in checkpoint and optimizer_embedding is not None:
            optimizer_embedding.load_state_dict(checkpoint['optimizer_embedding_state_dict'])
            print("Loaded embedding optimizer state from checkpoint")
        # Load scheduler states if they exist in checkpoint
        if 'scheduler_model_state_dict' in checkpoint:
            scheduler_model.load_state_dict(checkpoint['scheduler_model_state_dict'])
            print("Loaded model scheduler state from checkpoint")
        if 'scheduler_embedding_state_dict' in checkpoint and scheduler_embedding is not None:
            scheduler_embedding.load_state_dict(checkpoint['scheduler_embedding_state_dict'])
            print("Loaded embedding scheduler state from checkpoint")

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

            # Training differs based on model type
            if model_type == 'masked':
                # ===== MASKED DIFFUSION MODEL TRAINING =====
                # Sample masking probability: higher values = more masking
                t = torch.rand((x.shape[0],), device=device)  # Range [0.0, 1.0]

                # Create masked input using llada_mask
                xt = llada_mask(x, t=t, mask_index=model.mask_index)

                # Forward pass - returns loss directly
                loss = model(xt, x, t)

                # Backward and optimize
                optimizer_model.zero_grad()
                loss.backward()
                optimizer_model.step()
                scheduler_model.step()

                # Track losses
                total_losses.append(loss.item())
                recon_losses.append(0.0)  # Not applicable for masked model
                diffusion_losses.append(loss.item())
                prior_losses.append(0.0)  # Not applicable for masked model

                # Validation: compute accuracy on masked positions
                if step % print_freq == 0 or step == steps - 1:
                    with torch.no_grad():
                        # Get predictions for the current masked input
                        logits = model._forward_without_loss(xt, t)
                        preds = logits.argmax(dim=-1)

                        # Compute accuracy only on masked positions
                        mask = (xt == model.mask_index)
                        if mask.any():
                            masked_preds = preds[mask]
                            masked_targets = x[mask]
                            acc = (masked_preds == masked_targets).float().mean().item()
                        else:
                            acc = 0.0

                        # Also compute overall accuracy (including unmasked positions)
                        overall_acc = (preds == x).float().mean().item()

                    # Print progress with validation metrics
                    avg_mask_ratio = mask.float().mean().item()
                    print(f"[Step {step+1:>6}/{steps}] loss={loss.item():.4f} | "
                          f"mask_acc={acc:.4f} overall_acc={overall_acc:.4f} | "
                          f"mask_ratio={avg_mask_ratio:.2f} lr={scheduler_model.get_last_lr()[0]:.2e}")

                    # Log to TensorBoard
                    writer.add_scalar('Loss/total', loss.item(), step)
                    writer.add_scalar('Loss/masked', loss.item(), step)
                    writer.add_scalar('Metrics/masked_accuracy', acc, step)
                    writer.add_scalar('Metrics/overall_accuracy', overall_acc, step)
                    writer.add_scalar('Metrics/mask_ratio', avg_mask_ratio, step)
                    writer.add_scalar('Learning_Rate/model', scheduler_model.get_last_lr()[0], step)

                    # Log model parameter histograms periodically
                    if step % (print_freq * 10) == 0:
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                writer.add_histogram(f'Model/{name}', param.data, step)
                                if param.grad is not None:
                                    writer.add_histogram(f'Model/{name}.grad', param.grad, step)

                    # Log embedding visualization periodically
                    if step % (print_freq * 10) == 0:
                        emb_matrix = model.embed.weight[:-1, :].detach().cpu()  # Exclude mask token
                        writer.add_embedding(
                            emb_matrix,
                            metadata=[str(i) for i in range(vocab_size)],
                            global_step=step,
                            tag='token_embeddings'
                        )

            elif model_type == 'continuous':
                # ===== CONTINUOUS DIFFUSION MODEL TRAINING =====
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
                embedding_matrix = get_embedding_matrix()
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

                ### Option #1: directly on embeddings (COMMENTED OUT)
                dispersive_loss = get_dispersion_loss(get_embedding_matrix().repeat(batch_size, 1)) * 1e1

                ### Option #3: dispersive loss on layer activations, gradient only affects embeddings
                # Computed separately - will be handled by separate embedding optimizer
                if repae:
                    repae_layers = [0, 1, 2, 3, 4]
                    repae_dispersive_loss = 0
                    for layer_idx in repae_layers:
                        repae_dispersive_loss += get_dispersion_loss(rearrange(model.layer_activations[layer_idx], 'b l d -> (b l) d'))

                    repae_dispersive_loss = repae_dispersive_loss / len(repae_layers)
                    dispersive_loss = dispersive_loss + repae_dispersive_loss * 1e1

                # else:
                #     dispersive_loss = torch.tensor(0.0, device=device)

                # Backward pass with separate optimizers
                # Use torch.autograd.grad() to selectively compute gradients
                optimizer_model.zero_grad()
                optimizer_embedding.zero_grad()

                if repae:
                    # Compute gradients for model parameters from main loss only
                    model_params = list(model.parameters())
                    model_grads = torch.autograd.grad(
                        loss,
                        model_params,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=False
                    )

                    # Assign model gradients
                    for param, grad in zip(model_params, model_grads):
                        param.grad = grad

                    # Compute gradients for embedding parameters from dispersive loss only
                    embedding_params = list(embedding.parameters())
                    embedding_grads = torch.autograd.grad(
                        dispersive_loss,
                        embedding_params,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False
                    )

                    # Assign embedding gradients
                    for param, grad in zip(embedding_params, embedding_grads):
                        param.grad = grad
                else:
                    # No REPAE: normal backward for all parameters
                    loss.backward()

                # Step both optimizers
                optimizer_model.step()
                scheduler_model.step()

                if repae:
                    optimizer_embedding.step()
                    scheduler_embedding.step()

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
                    writer.add_scalar('Hyperparameters/learning_rate_model', scheduler_model.get_last_lr()[0], step)
                    writer.add_scalar('Hyperparameters/learning_rate_embedding', scheduler_embedding.get_last_lr()[0], step)

                    # Log embeddings periodically
                    if step % (print_freq * 10) == 0:
                        emb_matrix = get_embedding_matrix().detach().cpu()
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

                    print(get_embedding_matrix())
                    print(f"{step:>6} | recon={reconst_val:.4f} diff_tail={diff_tail_val:.4f} prior={prior_loss.item():.4f} "
                          f"loss={loss.item():.4f} (diff_mean={total_diffusion:.4f}) disp_loss={dispersive_loss:.4f} acc={acc:.4f}")

            elif model_type == 'combined':
                # ===== COMBINED MODEL TRAINING =====

                # Sample independent times for discrete and continuous processes
                t_disc = torch.rand((x.shape[0],), device=device)  # For discrete masking [0.0, 1.0]
                t_cont = torch.rand((x.shape[0],), device=device)  # For continuous noise [0.0, 1.0]

                # add mask (use t_disc for discrete masking)
                xt = llada_mask(x, t=t_disc, mask_index=model.mask_index) # [B, seq_len]

                # Get discrete embeddings from masked input (z_disc)
                z_disc = model.embed(xt)  # [B, seq_len, embed_dim]

                # Get clean embeddings from original tokens x_0
                x_clean_embed = model.embed(x)  # [B, seq_len, embed_dim]

                # Convert continuous t_cont to discrete timesteps for indexing noise schedule
                t_cont_discrete = (t_cont * num_timesteps).long()
                t_cont_discrete = torch.clamp(t_cont_discrete, 0, num_timesteps - 1)

                # Get noise schedule values for these timesteps
                sqrt_alpha_t = sqrt_alphas_cumprod[t_cont_discrete][:, None, None]  # [B, 1, 1]
                sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t_cont_discrete][:, None, None]  # [B, 1, 1]

                # Create boolean mask for masked positions
                mask = (xt == model.mask_index).unsqueeze(-1)  # [B, seq_len, 1]

                # Generate Gaussian noise
                noise = torch.randn_like(x_clean_embed)  # [B, seq_len, embed_dim]

                # Create z_t: behavior depends on combine_method
                if combine_method == 'add':
                    # ADD mode: noisy for masked, zero for unmasked (original CADD)
                    z_t = torch.where(
                        mask,
                        sqrt_alpha_t * x_clean_embed + sqrt_one_minus_alpha_t * noise,  # Noisy for masked positions
                        torch.zeros_like(x_clean_embed)  # Zero for unmasked positions
                    )
                else:  # concat
                    # CONCAT mode: noisy for masked, clean for unmasked
                    z_t = torch.where(
                        mask,
                        sqrt_alpha_t * x_clean_embed + sqrt_one_minus_alpha_t * noise,  # Noisy for masked positions
                        x_clean_embed  # Clean embeddings for unmasked positions
                    )

                # Apply dropout to z_t: randomly drop continuous latents during training
                if dropout_rate > 0.0:
                    # Create dropout mask: 1 to keep, 0 to drop
                    dropout_mask = (torch.rand(x.shape[0], device=device) > dropout_rate).float()
                    dropout_mask = dropout_mask[:, None, None]  # [B, 1, 1]
                    z_t = z_t * dropout_mask

                # CADD: Combine discrete and continuous embeddings
                if combine_method == 'add':
                    # ADD: z_combined = z_disc + coef * z_t, shape: [B, seq_len, embed_dim]
                    z_combined = z_disc + combined_coef * z_t
                else:  # concat
                    # CONCAT: z_combined = concat(z_disc, coef * z_t), shape: [B, seq_len, 2*embed_dim]
                    z_combined = torch.cat([z_disc, combined_coef * z_t], dim=-1)

                loss = model.forward_emb2loss(z_combined, xt, x, t_disc, t_cont)
                optimizer_model.zero_grad()
                loss.backward()
                optimizer_model.step()
                scheduler_model.step()
                
                # Track losses
                total_losses.append(loss.item())
                recon_losses.append(0.0)  # Not applicable for masked model
                diffusion_losses.append(loss.item())
                prior_losses.append(0.0)  # Not applicable for masked model

                # Validation: compute accuracy on masked positions
                if step % print_freq == 0 or step == steps - 1:
                    with torch.no_grad():
                        # Get predictions for the current masked input
                        logits = model.forward_emb2logits(z_combined, t_disc, t_cont)
                        preds = logits.argmax(dim=-1)

                        # Compute accuracy only on masked positions
                        mask = (xt == model.mask_index)
                        if mask.any():
                            masked_preds = preds[mask]
                            masked_targets = x[mask]
                            acc = (masked_preds == masked_targets).float().mean().item()
                        else:
                            acc = 0.0

                        # Also compute overall accuracy (including unmasked positions)
                        overall_acc = (preds == x).float().mean().item()

                    # Print progress with validation metrics
                    avg_mask_ratio = mask.float().mean().item()
                    print(f"[Step {step+1:>6}/{steps}] loss={loss.item():.4f} | "
                          f"mask_acc={acc:.4f} overall_acc={overall_acc:.4f} | "
                          f"mask_ratio={avg_mask_ratio:.2f} lr={scheduler_model.get_last_lr()[0]:.2e}")

                    # Log to TensorBoard
                    writer.add_scalar('Loss/total', loss.item(), step)
                    writer.add_scalar('Loss/masked', loss.item(), step)
                    writer.add_scalar('Metrics/masked_accuracy', acc, step)
                    writer.add_scalar('Metrics/overall_accuracy', overall_acc, step)
                    writer.add_scalar('Metrics/mask_ratio', avg_mask_ratio, step)
                    writer.add_scalar('Learning_Rate/model', scheduler_model.get_last_lr()[0], step)

                    # Log model parameter histograms periodically
                    if step % (print_freq * 10) == 0:
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                writer.add_histogram(f'Model/{name}', param.data, step)
                                if param.grad is not None:
                                    writer.add_histogram(f'Model/{name}.grad', param.grad, step)

                    # Log embedding visualization periodically
                    if step % (print_freq * 10) == 0:
                        emb_matrix = model.embed.weight[:-1, :].detach().cpu()  # Exclude mask token
                        writer.add_embedding(
                            emb_matrix,
                            metadata=[str(i) for i in range(vocab_size)],
                            global_step=step,
                            tag='token_embeddings'
                        )

            elif model_type == 'ccdd':
                # ===== CCDD MODEL TRAINING =====
                # Sample time for both discrete and continuous processes
                t = torch.rand((x.shape[0],), device=device)  # Range [0.0, 1.0]

                # 1. Discrete corruption: categorical noise with masking
                # Using uniform noise distribution (can be modified to use other distributions)
                eta_t = 1.0 - t  # Forward schedule: eta_t = 1 - t (decreases from 1 to 0)
                # Create corrupted discrete tokens
                mask_probs = 1.0 - eta_t  # Probability of masking increases with t
                x_t = llada_mask(x, t=mask_probs, mask_index=model.mask_index)

                # 2. Continuous corruption: Gaussian noise (VP schedule)
                # Algorithm: z_0 = E(x_0), where E is the encoder
                # Using token_embed as encoder: z_0 = token_embed(x_0)
                with torch.no_grad():
                    z_0 = model.token_embed(x)  # [B, L, latent_dim] where latent_dim = embed_dim

                # Convert t to discrete timesteps for noise schedule indexing
                t_discrete = (t * num_timesteps).long()
                t_discrete = torch.clamp(t_discrete, 0, num_timesteps - 1)

                # Get noise schedule values
                sqrt_alpha_t = sqrt_alphas_cumprod[t_discrete][:, None, None]  # [B, 1, 1]
                sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t_discrete][:, None, None]

                # Add Gaussian noise to continuous latent: z_t = α_t * z_0 + σ_t * ε
                epsilon = torch.randn_like(z_0)
                z_t = sqrt_alpha_t * z_0 + sqrt_one_minus_alpha_t * epsilon

                # 3. Model prediction: both heads
                # Scale continuous latent by ccdd_continuous_coef
                epsilon_pred, logits_pred = model(x_t, ccdd_continuous_coef * z_t, t)

                # 4. Compute losses
                # Continuous loss: MSE on noise prediction (ε-prediction objective)
                loss_cont = F.mse_loss(epsilon_pred, epsilon)

                # Discrete loss: Cross-entropy on ALL tokens (not just masked)
                # Algorithm: L_disc = -(1/B) Σ log softmax(ℓ_θ)[x_0]
                loss_disc = F.cross_entropy(
                    logits_pred.reshape(-1, vocab_size),  # [B*L, V]
                    x.reshape(-1)  # [B*L]
                )

                # Dispersive loss: directly on token embeddings (exclude mask token)
                dispersive_loss = get_dispersion_loss(
                    model.token_embed.weight[:-1, :].repeat(batch_size, 1)
                ) * 1e1

                # Total loss: weighted combination
                gamma_cont = args.get('gamma_cont', 1.0)  # Weight for continuous loss
                gamma_disc = args.get('gamma_disc', 1.0)  # Weight for discrete loss
                loss = gamma_cont * loss_cont + gamma_disc * loss_disc + dispersive_loss

                # Backward and optimize
                optimizer_model.zero_grad()
                loss.backward()
                optimizer_model.step()
                scheduler_model.step()

                # Track losses
                total_losses.append(loss.item())
                recon_losses.append(loss_disc.item())
                diffusion_losses.append(loss_cont.item())
                prior_losses.append(0.0)

                # Validation: compute accuracy metrics
                if step % print_freq == 0 or step == steps - 1:
                    with torch.no_grad():
                        # Get predictions
                        preds = logits_pred.argmax(dim=-1)

                        # Overall accuracy (on all tokens)
                        overall_acc = (preds == x).float().mean().item()

                        # Accuracy on masked positions only (for monitoring)
                        mask_positions = (x_t == model.mask_index)
                        if mask_positions.any():
                            masked_preds = preds[mask_positions]
                            masked_targets = x[mask_positions]
                            masked_acc = (masked_preds == masked_targets).float().mean().item()
                        else:
                            masked_acc = 0.0

                    # Print progress
                    avg_mask_ratio = mask_positions.float().mean().item()
                    print(f"[Step {step+1:>6}/{steps}] loss={loss.item():.4f} | "
                          f"loss_cont={loss_cont.item():.4f} loss_disc={loss_disc.item():.4f} disp={dispersive_loss.item():.4f} | "
                          f"acc={overall_acc:.4f} masked_acc={masked_acc:.4f} | "
                          f"mask_ratio={avg_mask_ratio:.2f} lr={scheduler_model.get_last_lr()[0]:.2e}")

                    # Log to TensorBoard
                    writer.add_scalar('Loss/total', loss.item(), step)
                    writer.add_scalar('Loss/continuous', loss_cont.item(), step)
                    writer.add_scalar('Loss/discrete', loss_disc.item(), step)
                    writer.add_scalar('Loss/dispersive', dispersive_loss.item(), step)
                    writer.add_scalar('Metrics/overall_accuracy', overall_acc, step)
                    writer.add_scalar('Metrics/masked_accuracy', masked_acc, step)
                    writer.add_scalar('Metrics/mask_ratio', avg_mask_ratio, step)
                    writer.add_scalar('Learning_Rate/model', scheduler_model.get_last_lr()[0], step)

                    # Log model parameter histograms periodically
                    if step % (print_freq * 10) == 0:
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                writer.add_histogram(f'Model/{name}', param.data, step)
                                if param.grad is not None:
                                    writer.add_histogram(f'Model/{name}.grad', param.grad, step)


            # Save checkpoint every 10000 iterations
            if checkpoint_path and (step + 1) % 10000 == 0:
                # Create checkpoint filename with iteration number
                checkpoint_dir = os.path.dirname(checkpoint_path)
                checkpoint_name = f"checkpoint_{step + 1}.pt"
                iter_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)

                # Save checkpoint based on model type
                if model_type == 'masked' or model_type == 'combined' or model_type == 'ccdd':
                    # Masked/Combined/CCDD model: only save model state
                    checkpoint_config = {
                        'model_type': model_type,
                        'vocab_size': vocab_size,
                        'seq_len': seq_len,
                        'embed_dim': embed_dim,
                        'n_blocks': n_blocks,
                        'n_heads': n_heads,
                    }
                    if model_type == 'combined':
                        checkpoint_config['combined_coef'] = combined_coef
                        checkpoint_config['dropout_rate'] = dropout_rate
                        checkpoint_config['combine_method'] = combine_method
                    elif model_type == 'ccdd':
                        checkpoint_config['latent_dim'] = model.latent_dim
                        checkpoint_config['ccdd_continuous_coef'] = ccdd_continuous_coef

                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'optimizer_model_state_dict': optimizer_model.state_dict(),
                            'scheduler_model_state_dict': scheduler_model.state_dict(),
                            'config': checkpoint_config,
                        },
                        iter_checkpoint_path
                    )
                else:
                    # Continuous model: save model and embedding states
                    embedding_state = embedding.state_dict()
                    model_state = model.state_dict()

                    torch.save(
                        {
                            'embedding_state_dict': embedding_state,
                            'model_state_dict': model_state,
                            'optimizer_model_state_dict': optimizer_model.state_dict(),
                            'optimizer_embedding_state_dict': optimizer_embedding.state_dict(),
                            'scheduler_model_state_dict': scheduler_model.state_dict(),
                            'scheduler_embedding_state_dict': scheduler_embedding.state_dict(),
                            'config': {
                                'model_type': 'continuous',
                                'embed_dim': embed_dim,
                                'hidden_dim': hidden_dim,
                                'n_blocks': n_blocks,
                                'n_heads': n_heads,
                                'vocab_size': vocab_size,
                                'seq_len': seq_len,
                                'positional_encoding': positional_encoding,
                                'embedding_type': embedding_type,
                                'transformer_block_type': transformer_block_type,
                                'enable_repae': repae,
                            },
                        },
                        iter_checkpoint_path
                    )
                print(f"Saved checkpoint to {iter_checkpoint_path}")

            if step % 1000 == 0 and model_type == 'continuous':
                # Sampling for continuous diffusion model
                n_samples = args.get('n_samples', 100)
                sampling_steps = args.get('sampling_steps', num_timesteps)
                sampling_eta = args.get('sampling_eta', 0.0)
                sampling_start_t = args.get('sampling_start_t', num_timesteps - 1)
                if isinstance(sampling_start_t, float):
                    sampling_start_t = int(round(sampling_start_t))
                sampling_start_t = int(sampling_start_t)
                sampling_start_t = max(0, min(num_timesteps - 1, sampling_start_t))

                with torch.no_grad():
                    embedding_matrix = get_embedding_matrix()

                    # Load test quiz data for completion evaluation
                    test_quiz = None
                    if dataset_type == 'sudoku':
                        test_quiz_path = args.get('test_quiz_path', 'data_vmd/sudoku_test.csv')
                        if os.path.exists(test_quiz_path):
                            test_quiz, _ = load_sudoku_dataset(test_quiz_path)
                            test_quiz = test_quiz[:n_samples].to(device)
                            print(f"Loaded test quiz data from {test_quiz_path} for completion evaluation")
                            print(f"Quiz data shape: {test_quiz.shape}")

                    # Prepare common sampling parameters
                    alpha_start = alphas_cumprod[sampling_start_t]
                    sqrt_one_minus_alpha_start = torch.sqrt(torch.clamp(1.0 - alpha_start, min=1e-8))
                    sqrt_alpha_start = torch.sqrt(alpha_start)
                    schedule = torch.linspace(float(sampling_start_t), 0.0, sampling_steps, device=device)
                    schedule = torch.round(schedule).to(torch.long)
                    schedule = torch.unique_consecutive(schedule)

                    # ===== 1. GENERATION FROM PURE NOISE =====
                    print(f"\n{'='*60}")
                    print("1. GENERATION FROM PURE NOISE")
                    print(f"{'='*60}")
                    print(f"Denoising over {sampling_steps} steps from t={sampling_start_t} to t=0...")

                    z_gen = torch.randn(n_samples, seq_len, embed_dim, device=device) * sqrt_one_minus_alpha_start

                    # Denoising loop for generation
                    for step_idx, t_discrete in enumerate(schedule.tolist()):
                        t_tensor = torch.full((n_samples,), t_discrete, dtype=torch.long, device=device)
                        t_continuous = t_tensor.float() / denom

                        logits = model(z_gen, t_continuous)
                        probs = F.softmax(logits, dim=-1)
                        x_reconst = probs @ embedding_matrix
                        pred_tokens = logits.argmax(dim=-1)

                        x_embed_disc = x_reconst
                        ### clamping
                        # x_embed_disc = embedding_matrix[pred_tokens]

                        alpha_t = alphas_cumprod[t_tensor].view(n_samples, 1, 1)
                        sqrt_alpha_t = torch.sqrt(alpha_t)
                        sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
                        eps_pred = (z_gen - sqrt_alpha_t * x_embed_disc) / (sqrt_one_minus_alpha_t + 1e-8)

                        if t_discrete > 0:
                            next_index = schedule[min(step_idx + 1, schedule.numel() - 1)].item()
                            alpha_next = alphas_cumprod[next_index].view(1, 1, 1).to(device=device, dtype=dtype)
                            alpha_next = alpha_next.expand(n_samples, 1, 1)
                            sqrt_alpha_next = torch.sqrt(alpha_next)
                            sqrt_one_minus_alpha_next = torch.sqrt(torch.clamp(1.0 - alpha_next, min=1e-8))

                            if sampling_eta > 0:
                                noise = torch.randn_like(z_gen)
                                eps_mix = (1.0 - sampling_eta) * eps_pred + sampling_eta * noise
                            else:
                                eps_mix = eps_pred

                            z_gen = sqrt_alpha_next * x_embed_disc + sqrt_one_minus_alpha_next * eps_mix
                        else:
                            z_gen = x_embed_disc

                    # Final prediction for generation
                    t_zero = torch.zeros(n_samples, dtype=torch.long, device=device)
                    t_zero_continuous = t_zero.float() / denom
                    final_logits_gen = model(z_gen, t_zero_continuous)
                    final_preds_gen = final_logits_gen.argmax(dim=-1)

                    # Display and evaluate generation results
                    print("\nGeneration samples:")
                    if dataset_type == 'sudoku':
                        evaluate_and_display_sudoku(
                            final_preds_gen,
                            n_samples,
                            mode_str="Generation",
                            writer=writer,
                            step=step,
                            prefix="generation",
                            max_display=5
                        )

                    # ===== 2. COMPLETION FROM PARTIAL QUIZ =====
                    if test_quiz is not None and dataset_type == 'sudoku':
                        print(f"\n{'='*60}")
                        print("2. COMPLETION FROM PARTIAL QUIZ")
                        print(f"{'='*60}")

                        z_comp = torch.randn(n_samples, seq_len, embed_dim, device=device) * sqrt_one_minus_alpha_start

                        # Denoising loop for completion
                        for step_idx, t_discrete in enumerate(schedule.tolist()):
                            t_tensor = torch.full((n_samples,), t_discrete, dtype=torch.long, device=device)
                            t_continuous = t_tensor.float() / denom

                            logits = model(z_comp, t_continuous)
                            probs = F.softmax(logits, dim=-1)
                            x_reconst = probs @ embedding_matrix
                            pred_tokens = logits.argmax(dim=-1)

                            # x_embed_disc = x_reconst
                            ### clamping
                            x_embed_disc = embedding_matrix[pred_tokens]

                            # Inject ground truth quiz values at known positions
                            quiz_mask = (test_quiz != 0).unsqueeze(-1)
                            quiz_embeddings = embedding_matrix[test_quiz]
                            x_embed_disc = torch.where(quiz_mask, quiz_embeddings, x_embed_disc)

                            alpha_t = alphas_cumprod[t_tensor].view(n_samples, 1, 1)
                            sqrt_alpha_t = torch.sqrt(alpha_t)
                            sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
                            eps_pred = (z_comp - sqrt_alpha_t * x_embed_disc) / (sqrt_one_minus_alpha_t + 1e-8)

                            if t_discrete > 0:
                                next_index = schedule[min(step_idx + 1, schedule.numel() - 1)].item()
                                alpha_next = alphas_cumprod[next_index].view(1, 1, 1).to(device=device, dtype=dtype)
                                alpha_next = alpha_next.expand(n_samples, 1, 1)
                                sqrt_alpha_next = torch.sqrt(alpha_next)
                                sqrt_one_minus_alpha_next = torch.sqrt(torch.clamp(1.0 - alpha_next, min=1e-8))

                                if sampling_eta > 0:
                                    noise = torch.randn_like(z_comp)
                                    eps_mix = (1.0 - sampling_eta) * eps_pred + sampling_eta * noise
                                else:
                                    eps_mix = eps_pred

                                z_comp = sqrt_alpha_next * x_embed_disc + sqrt_one_minus_alpha_next * eps_mix
                            else:
                                z_comp = x_embed_disc

                        # Final prediction for completion
                        final_logits_comp = model(z_comp, t_zero_continuous)
                        final_preds_comp = final_logits_comp.argmax(dim=-1)

                        # Inject quiz values into final predictions to preserve known positions
                        quiz_mask_1d = (test_quiz != 0)
                        final_preds_comp = torch.where(quiz_mask_1d, test_quiz, final_preds_comp)


                        # Display completion results side-by-side with quiz
                        print("\nCompletion samples (Quiz | Prediction):")
                        evaluate_and_display_sudoku(
                            final_preds_comp,
                            n_samples,
                            mode_str="Completion",
                            writer=writer,
                            step=step,
                            prefix="completion",
                            quiz_data=test_quiz,
                            max_display=5
                        )

                    elif dataset_type == 'sequential':
                        evaluate_and_display_sequential(
                            final_preds_gen,
                            n_samples,
                            mode_str="Generation",
                            writer=writer,
                            step=step,
                            prefix="generation",
                            max_display=0
                        )

            # ===== MASKED MODEL SAMPLING (every 1000 steps) =====
            if step % 1000 == 0 and model_type == 'masked':
                n_samples = args.get('n_samples', 100)
                mdm_sampling_steps = args.get('mdm_sampling_steps', 10)  # Number of unmasking steps
                mdm_temperature = args.get('mdm_temperature', 0.0)

                with torch.no_grad():
                    # Load test quiz data for completion evaluation
                    test_quiz = None
                    if dataset_type == 'sudoku':
                        test_quiz_path = args.get('test_quiz_path', 'data_vmd/sudoku_test.csv')
                        if os.path.exists(test_quiz_path):
                            test_quiz, _ = load_sudoku_dataset(test_quiz_path)
                            test_quiz = test_quiz[:n_samples].to(device)
                            print(f"Loaded test quiz data from {test_quiz_path} for completion evaluation")
                            print(f"Quiz data shape: {test_quiz.shape}")

                    # Generation: start from fully masked
                    print(f"\nGenerating {n_samples} samples with {mdm_sampling_steps} unmasking steps (generation)...")
                    gen_xt = torch.full((n_samples, seq_len), model.mask_index, dtype=torch.long, device=device)
                    gen_preds = model.generate(gen_xt, steps=mdm_sampling_steps, temperature=mdm_temperature)

                    if dataset_type == 'sudoku':
                        evaluate_and_display_sudoku(
                            gen_preds,
                            n_samples,
                            mode_str="Generation",
                            writer=writer,
                            step=step,
                            prefix="generation",
                            max_display=5
                        )
                    else:
                        evaluate_and_display_sequential(
                            gen_preds,
                            n_samples,
                            mode_str="Generation",
                            writer=writer,
                            step=step,
                            prefix="generation",
                            max_display=50
                        )

                    # Completion: if quiz data available, run a second pass
                    if test_quiz is not None:
                        comp_samples = test_quiz.shape[0]
                        print(f"\nCompleting {comp_samples} quizzes with {mdm_sampling_steps} unmasking steps (completion)...")
                        xt = torch.where(test_quiz != 0, test_quiz, torch.full_like(test_quiz, model.mask_index))
                        num_known = (test_quiz != 0).sum().item()
                        print(f"Starting from partial quiz with {num_known}/{comp_samples * seq_len} known values")

                        comp_preds = model.generate(xt, steps=mdm_sampling_steps, temperature=mdm_temperature)

                        evaluate_and_display_sudoku(
                            comp_preds,
                            comp_samples,
                            mode_str="Completion",
                            writer=writer,
                            step=step,
                            prefix="completion",
                            quiz_data=test_quiz,
                            max_display=5
                        )

            if step % 1000 == 0 and model_type == 'combined':
                # ===== COMBINED MODEL SAMPLING =====
                n_samples = args.get('n_samples', 100)
                combined_sampling_steps = args.get('mdm_sampling_steps', 20)
                combined_temperature = args.get('mdm_temperature', 0.0)

                with torch.no_grad():
                    # Load test quiz data for completion evaluation
                    test_quiz = None
                    if dataset_type == 'sudoku':
                        test_quiz_path = args.get('test_quiz_path', 'data_vmd/sudoku_test.csv')
                        if os.path.exists(test_quiz_path):
                            test_quiz, _ = load_sudoku_dataset(test_quiz_path)
                            test_quiz = test_quiz[:n_samples].to(device)
                            print(f"Loaded test quiz data from {test_quiz_path} for completion evaluation")
                            print(f"Quiz data shape: {test_quiz.shape}")

                    # Prepare common parameters
                    start_noise_level = 1.0
                    t_start_discrete = int(start_noise_level * num_timesteps)
                    t_start_discrete = min(max(0, t_start_discrete), num_timesteps - 1)
                    sqrt_alpha_start = sqrt_alphas_cumprod[t_start_discrete]
                    sqrt_one_minus_alpha_start = sqrt_one_minus_alphas_cumprod[t_start_discrete]

                    # ===== 1. GENERATION FROM FULLY MASKED =====
                    print(f"\n{'='*60}")
                    print("1. GENERATION FROM FULLY MASKED")
                    print(f"{'='*60}")

                    # Start from fully masked
                    xt_gen = torch.full((n_samples, seq_len), model.mask_index, dtype=torch.long, device=device)

                    # Initialize with random tokens + noise
                    random_tokens_gen = torch.randint(0, vocab_size, (n_samples, seq_len), device=device)
                    x_clean_base_gen = model.embed(random_tokens_gen)
                    noise = torch.randn_like(x_clean_base_gen)
                    xt_embed_gen = sqrt_alpha_start * x_clean_base_gen + sqrt_one_minus_alpha_start * noise

                    # Compute transfer schedule
                    initial_mask_gen = xt_gen == model.mask_index
                    num_transfer_tokens_gen = get_num_transfer_tokens(initial_mask_gen, combined_sampling_steps)

                    print(f"Generating {n_samples} samples with {combined_sampling_steps} steps...")

                    # Generation loop
                    for i in range(combined_sampling_steps):
                        t_val = 1.0 - (i / max(combined_sampling_steps - 1, 1)) * 1.0
                        t_continuous = torch.full((n_samples,), t_val, device=device)

                        # CADD: Combine discrete and continuous embeddings
                        z_disc_gen = model.embed(xt_gen)  # Discrete embeddings from current state
                        mask_gen = (xt_gen == model.mask_index).unsqueeze(-1)

                        if combine_method == 'add':
                            # ADD mode: zero out unmasked positions
                            z_t_gen = torch.where(mask_gen, xt_embed_gen, torch.zeros_like(xt_embed_gen))
                        else:  # concat
                            # CONCAT mode: use current embeddings for unmasked (from z_disc)
                            unmasked_embed = model.embed(xt_gen)
                            z_t_gen = torch.where(mask_gen, xt_embed_gen, unmasked_embed)

                        # Reweight z_t during inference to compensate for dropout during training
                        if dropout_rate > 0.0:
                            z_t_gen = z_t_gen / (1.0 - dropout_rate)

                        if combine_method == 'add':
                            z_combined_gen = z_disc_gen + combined_coef * z_t_gen
                        else:  # concat
                            z_combined_gen = torch.cat([z_disc_gen, combined_coef * z_t_gen], dim=-1)

                        # Classifier-Free Guidance: conditional and unconditional predictions
                        if cfg_scale > 0.0:
                            # Conditional prediction (with continuous guidance)
                            logits_cond = model.forward_emb2logits(z_combined_gen, t_continuous, t_continuous)

                            # Unconditional prediction (discrete only, no continuous guidance)
                            if combine_method == 'add':
                                z_uncond = z_disc_gen  # No z_t component
                            else:  # concat
                                z_uncond = torch.cat([z_disc_gen, torch.zeros_like(combined_coef * z_t_gen)], dim=-1)
                            logits_uncond = model.forward_emb2logits(z_uncond, t_continuous, t_continuous)

                            # Interpolate: logits = uncond + scale * (cond - uncond)
                            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                        else:
                            # No CFG, use conditional prediction only
                            logits = model.forward_emb2logits(z_combined_gen, t_continuous, t_continuous)
                        logits_with_noise = add_gumbel_noise(logits, temperature=combined_temperature)
                        x0_pred = torch.argmax(logits_with_noise, dim=-1)

                        p = F.softmax(logits, dim=-1)
                        x0_p = torch.gather(p, dim=-1, index=x0_pred.unsqueeze(-1)).squeeze(-1)

                        mask_positions = xt_gen == model.mask_index
                        neg_inf = torch.tensor(float("-inf"), device=device)
                        confidence = torch.where(mask_positions, x0_p, neg_inf)
                        x0_pred = torch.where(mask_positions, x0_pred, xt_gen)

                        transfer_index = torch.zeros_like(x0_pred, dtype=torch.bool, device=device)
                        for j in range(confidence.size(0)):
                            if num_transfer_tokens_gen[j, i] > 0:
                                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens_gen[j, i])
                                transfer_index[j, select_index] = True

                        xt_gen[transfer_index] = x0_pred[transfer_index]

                        if i < combined_sampling_steps - 1:
                            x_clean_pred = model.embed(x0_pred)
                            mask_still = (xt_gen == model.mask_index).unsqueeze(-1)

                            if mask_still.any():
                                t_next = 1.0 - ((i + 1) / max(combined_sampling_steps - 1, 1)) * 1.0
                                t_next_discrete = int(t_next * num_timesteps)
                                t_next_discrete = min(max(0, t_next_discrete), num_timesteps - 1)
                                t_curr_discrete = int(t_val * num_timesteps)
                                t_curr_discrete = min(max(0, t_curr_discrete), num_timesteps - 1)

                                alpha_curr = alphas_cumprod[t_curr_discrete]
                                alpha_next = alphas_cumprod[t_next_discrete]

                                sqrt_alpha_curr = torch.sqrt(alpha_curr).view(1, 1, 1)
                                sqrt_alpha_next = torch.sqrt(alpha_next).view(1, 1, 1)
                                sqrt_one_minus_alpha_curr = torch.sqrt(1.0 - alpha_curr).view(1, 1, 1)
                                sqrt_one_minus_alpha_next = torch.sqrt(1.0 - alpha_next).view(1, 1, 1)

                                # Denoise using z_t (not z_combined)
                                z_t_curr = torch.where(mask_still, xt_embed_gen, torch.zeros_like(xt_embed_gen))
                                eps_pred = (z_t_curr - sqrt_alpha_curr * x_clean_pred) / (sqrt_one_minus_alpha_curr + 1e-8)
                                xt_embed_denoised = sqrt_alpha_next * x_clean_pred + sqrt_one_minus_alpha_next * eps_pred
                                xt_embed_gen = torch.where(mask_still, xt_embed_denoised, torch.zeros_like(xt_embed_gen))
                            else:
                                xt_embed_gen = torch.zeros_like(xt_embed_gen)

                    # Final generation predictions
                    final_preds_gen = xt_gen

                    # Display and evaluate generation results
                    print("\nGeneration samples:")
                    if dataset_type == 'sudoku':
                        evaluate_and_display_sudoku(
                            final_preds_gen,
                            n_samples,
                            mode_str="Generation",
                            writer=writer,
                            step=step,
                            prefix="generation",
                            max_display=5
                        )
                    elif dataset_type == 'sequential':
                        evaluate_and_display_sequential(
                            final_preds_gen,
                            n_samples,
                            mode_str="Generation",
                            writer=writer,
                            step=step,
                            prefix="generation",
                            max_display=0
                        )

                    # ===== 2. COMPLETION FROM PARTIAL QUIZ =====
                    if test_quiz is not None and dataset_type == 'sudoku':
                        print(f"\n{'='*60}")
                        print("2. COMPLETION FROM PARTIAL QUIZ")
                        print(f"{'='*60}")

                        # Start from partially masked
                        xt_comp = torch.where(test_quiz != 0, test_quiz, torch.full_like(test_quiz, model.mask_index))
                        num_known = (test_quiz != 0).sum().item()
                        print(f"Starting from partial quiz with {num_known}/{n_samples * seq_len} known values")

                        # Initialize embeddings (z_t component)
                        mask_only = (xt_comp == model.mask_index).unsqueeze(-1)
                        # Initialize with random noise for masked positions
                        xt_embed_comp = torch.randn(n_samples, seq_len, embed_dim, device=device)

                        if combine_method == 'add':
                            # ADD mode: z_t should be random noise for masked, zero for known
                            xt_embed_comp = torch.where(mask_only, xt_embed_comp, torch.zeros_like(xt_embed_comp))
                        else:  # concat
                            # CONCAT mode: z_t should be random noise for masked, clean embeddings for known
                            known_embed = model.embed(xt_comp)
                            xt_embed_comp = torch.where(mask_only, xt_embed_comp, known_embed)

                        # Compute transfer schedule
                        initial_mask_comp = xt_comp == model.mask_index
                        num_transfer_tokens_comp = get_num_transfer_tokens(initial_mask_comp, combined_sampling_steps)

                        # Completion loop
                        for i in range(combined_sampling_steps):
                            # Continuous timestep: based on step progression (1.0 → 0.0)
                            t_cont_val = 1.0 - (i / max(combined_sampling_steps - 1, 1)) * 1.0
                            t_cont = torch.full((n_samples,), t_cont_val, device=device)

                            # Discrete timestep: based on current mask ratio (correlated to unmasked tokens)
                            mask_ratio = (xt_comp == model.mask_index).float().mean(dim=1)  # [B]
                            t_disc = mask_ratio  # Higher when more masked, lower when fewer masked

                            # CADD: Combine discrete and continuous embeddings
                            z_disc_comp = model.embed(xt_comp)  # Discrete embeddings from current state
                            mask_comp = (xt_comp == model.mask_index).unsqueeze(-1)
                            
                            
                            # print(i, t_cont[0], t_disc[0], xt_comp[0].tolist())

                            if combine_method == 'add':
                                # ADD mode: zero out unmasked positions
                                z_t_comp = torch.where(mask_comp, xt_embed_comp, torch.zeros_like(xt_embed_comp))
                            else:  # concat
                                # CONCAT mode: use current embeddings for unmasked (from z_disc)
                                unmasked_embed = model.embed(xt_comp)
                                z_t_comp = torch.where(mask_comp, xt_embed_comp, unmasked_embed)

                            # Reweight z_t during inference to compensate for dropout during training
                            if dropout_rate > 0.0:
                                z_t_comp = z_t_comp / (1.0 - dropout_rate)

                            if combine_method == 'add':
                                z_combined_comp = z_disc_comp + combined_coef * z_t_comp
                            else:  # concat
                                z_combined_comp = torch.cat([z_disc_comp, combined_coef * z_t_comp], dim=-1)

                            # Classifier-Free Guidance: conditional and unconditional predictions
                            if cfg_scale > 0.0:
                                # Conditional prediction (with continuous guidance)
                                logits_cond = model.forward_emb2logits(z_combined_comp, t_disc, t_cont)

                                # Unconditional prediction (discrete only, no continuous guidance)
                                if combine_method == 'add':
                                    z_uncond = z_disc_comp  # No z_t component
                                else:  # concat
                                    z_uncond = torch.cat([z_disc_comp, torch.zeros_like(combined_coef * z_t_comp)], dim=-1)
                                logits_uncond = model.forward_emb2logits(z_uncond, t_disc, t_cont)

                                # Interpolate: logits = uncond + scale * (cond - uncond)
                                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                            else:
                                # No CFG, use conditional prediction only
                                logits = model.forward_emb2logits(z_combined_comp, t_disc, t_cont)
                            logits_with_noise = add_gumbel_noise(logits, temperature=combined_temperature)
                            x0_pred = torch.argmax(logits_with_noise, dim=-1)

                            p = F.softmax(logits, dim=-1)
                            x0_p = torch.gather(p, dim=-1, index=x0_pred.unsqueeze(-1)).squeeze(-1)

                            mask_positions = xt_comp == model.mask_index
                            neg_inf = torch.tensor(float("-inf"), device=device)
                            confidence = torch.where(mask_positions, x0_p, neg_inf)
                            x0_pred = torch.where(mask_positions, x0_pred, xt_comp)

                            transfer_index = torch.zeros_like(x0_pred, dtype=torch.bool, device=device)
                            for j in range(confidence.size(0)):
                                if num_transfer_tokens_comp[j, i] > 0:
                                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens_comp[j, i])
                                    transfer_index[j, select_index] = True

                            xt_comp[transfer_index] = x0_pred[transfer_index]

                            if i < combined_sampling_steps - 1:
                                x_clean_pred = model.embed(x0_pred)
                                mask_still = (xt_comp == model.mask_index).unsqueeze(-1)

                                if mask_still.any():
                                    # Use t_cont for continuous denoising (not t_disc)
                                    t_cont_next = 1.0 - ((i + 1) / max(combined_sampling_steps - 1, 1)) * 1.0
                                    t_next_discrete = int(t_cont_next * num_timesteps)
                                    t_next_discrete = min(max(0, t_next_discrete), num_timesteps - 1)
                                    t_curr_discrete = int(t_cont_val * num_timesteps)
                                    t_curr_discrete = min(max(0, t_curr_discrete), num_timesteps - 1)

                                    alpha_curr = alphas_cumprod[t_curr_discrete]
                                    alpha_next = alphas_cumprod[t_next_discrete]

                                    sqrt_alpha_curr = torch.sqrt(alpha_curr).view(1, 1, 1)
                                    sqrt_alpha_next = torch.sqrt(alpha_next).view(1, 1, 1)
                                    sqrt_one_minus_alpha_curr = torch.sqrt(1.0 - alpha_curr).view(1, 1, 1)
                                    sqrt_one_minus_alpha_next = torch.sqrt(1.0 - alpha_next).view(1, 1, 1)

                                    # Denoise using z_t (not z_combined)
                                    z_t_curr = torch.where(mask_still, xt_embed_comp, torch.zeros_like(xt_embed_comp))
                                    eps_pred = (z_t_curr - sqrt_alpha_curr * x_clean_pred) / (sqrt_one_minus_alpha_curr + 1e-8)
                                    xt_embed_denoised = sqrt_alpha_next * x_clean_pred + sqrt_one_minus_alpha_next * eps_pred
                                    # Update: masked positions get denoised, unmasked get zero, known get zero (will be handled by discrete embedding)
                                    xt_embed_comp = torch.where(mask_still, xt_embed_denoised, torch.zeros_like(xt_embed_comp))
                                else:
                                    # All positions unmasked, set z_t to zero
                                    xt_embed_comp = torch.zeros_like(xt_embed_comp)

                        # Final completion predictions
                        final_preds_comp = xt_comp

                        # Display completion results side-by-side
                        print("\nCompletion samples (Quiz | Prediction):")
                        evaluate_and_display_sudoku(
                            final_preds_comp,
                            n_samples,
                            mode_str="Completion",
                            writer=writer,
                            step=step,
                            prefix="completion",
                            quiz_data=test_quiz,
                            max_display=5
                        )

            # ===== CCDD MODEL SAMPLING (every 1000 steps) =====
            if step % 1000 == 0 and model_type == 'ccdd':
                n_samples = args.get('n_samples', 100)
                ccdd_sampling_steps = args.get('ccdd_sampling_steps', 20)  # Number of denoising steps
                ccdd_eta_ddpm = args.get('ccdd_eta_ddpm', 0.0)  # 0=DDIM, 1=DDPM

                with torch.no_grad():
                    # Load test quiz data for completion evaluation
                    test_quiz = None
                    if dataset_type == 'sudoku':
                        test_quiz_path = args.get('test_quiz_path', 'data_vmd/sudoku_test.csv')
                        if os.path.exists(test_quiz_path):
                            test_quiz, _ = load_sudoku_dataset(test_quiz_path)
                            test_quiz = test_quiz[:n_samples].to(device)
                            print(f"Loaded test quiz data from {test_quiz_path} for completion evaluation")
                            print(f"Quiz data shape: {test_quiz.shape}")

                    # Create time schedule for denoising: t_0=1.0 -> t_K=0.0
                    time_schedule = torch.linspace(1.0, 0.0, ccdd_sampling_steps + 1, device=device)

                    # ===== 1. GENERATION FROM PURE NOISE =====
                    print(f"\n{'='*60}")
                    print("1. GENERATION FROM PURE NOISE (CCDD)")
                    print(f"{'='*60}")
                    print(f"Denoising over {ccdd_sampling_steps} steps...")

                    # Initialize both latents from noise
                    # Discrete: all [MASK] tokens (fully masked)
                    x_t = torch.full((n_samples, seq_len), model.mask_index, dtype=torch.long, device=device)
                    # Continuous: Gaussian noise
                    z_t = torch.randn(n_samples, seq_len, model.latent_dim, device=device)

                    # Compute unmasking schedule (MDM-style)
                    initial_mask = (x_t == model.mask_index)  # All True initially
                    num_transfer_tokens = get_num_transfer_tokens(initial_mask, ccdd_sampling_steps)

                    # Denoising loop
                    for k in range(ccdd_sampling_steps):
                        t_k = time_schedule[k]
                        t_next = time_schedule[k + 1]

                        # Time tensor for model input
                        t_tensor = torch.full((n_samples,), t_k.item(), device=device)

                        # Model prediction: both heads
                        # Scale continuous latent by ccdd_continuous_coef
                        epsilon_pred, logits_pred = model(x_t, ccdd_continuous_coef * z_t, t_tensor)

                        # ===== (A) Discrete reverse: MDM-style unmasking =====
                        # Deterministic argmax (no temperature)
                        x0_pred = torch.argmax(logits_pred, dim=-1)  # [B, L]

                        # Compute confidence (softmax probability of predicted token)
                        probs = F.softmax(logits_pred, dim=-1)  # [B, L, V]
                        confidence = torch.gather(probs, dim=-1, index=x0_pred.unsqueeze(-1)).squeeze(-1)  # [B, L]

                        # Mask out already-unmasked positions (set to -inf so topk ignores them)
                        mask_positions = (x_t == model.mask_index)
                        neg_inf = torch.tensor(float("-inf"), device=device)
                        confidence = torch.where(mask_positions, confidence, neg_inf)

                        # Keep unmasked tokens unchanged
                        x0_pred = torch.where(mask_positions, x0_pred, x_t)

                        # Select top-k confident tokens to unmask (per batch element)
                        transfer_index = torch.zeros_like(x0_pred, dtype=torch.bool, device=device)
                        for j in range(n_samples):
                            if num_transfer_tokens[j, k] > 0:
                                _, select_indices = torch.topk(confidence[j], k=num_transfer_tokens[j, k])
                                transfer_index[j, select_indices] = True

                        # Unmask selected tokens
                        x_t[transfer_index] = x0_pred[transfer_index]

                        # ===== (B) Continuous reverse step (DDIM/DDPM) =====
                        # Convert continuous time to discrete timestep for noise schedule
                        t_k_discrete = int(t_k.item() * num_timesteps)
                        t_k_discrete = min(max(0, t_k_discrete), num_timesteps - 1)
                        t_next_discrete = int(t_next.item() * num_timesteps)
                        t_next_discrete = min(max(0, t_next_discrete), num_timesteps - 1)

                        # Get noise schedule values
                        alpha_k = alphas_cumprod[t_k_discrete]
                        sigma_k = torch.sqrt(1.0 - alpha_k)

                        # Predict x_0 from z_t and epsilon_pred
                        sqrt_alpha_k = torch.sqrt(alpha_k)
                        z_0_pred = (z_t - sigma_k * epsilon_pred) / sqrt_alpha_k

                        if k < ccdd_sampling_steps - 1:
                            # Not the final step: compute next z_t
                            alpha_next = alphas_cumprod[t_next_discrete]
                            sigma_next = torch.sqrt(1.0 - alpha_next)
                            sqrt_alpha_next = torch.sqrt(alpha_next)

                            # DDIM mean
                            z_t_mean = sqrt_alpha_next * z_0_pred + sigma_next * epsilon_pred

                            # Optional stochasticity (DDPM)
                            if ccdd_eta_ddpm > 0.0 and k < ccdd_sampling_steps - 1:
                                # Compute posterior variance
                                sigma_k_next = torch.sqrt((1 - alpha_k / alpha_next) * (1 - alpha_next) / (1 - alpha_k))
                                noise = torch.randn_like(z_t)
                                z_t = z_t_mean + ccdd_eta_ddpm * sigma_k_next * noise
                            else:
                                # DDIM (deterministic)
                                z_t = z_t_mean
                        else:
                            # Final step: z_t = z_0_pred
                            z_t = z_0_pred

                    # Final predictions
                    final_preds_gen = x_t

                    # Display and evaluate generation results
                    print("\nGeneration samples:")
                    if dataset_type == 'sudoku':
                        evaluate_and_display_sudoku(
                            final_preds_gen,
                            n_samples,
                            mode_str="Generation",
                            writer=writer,
                            step=step,
                            prefix="generation",
                            max_display=5
                        )
                    elif dataset_type == 'sequential':
                        evaluate_and_display_sequential(
                            final_preds_gen,
                            n_samples,
                            mode_str="Generation",
                            writer=writer,
                            step=step,
                            prefix="generation",
                            max_display=50
                        )

                    # ===== 2. COMPLETION FROM PARTIAL QUIZ =====
                    if test_quiz is not None and dataset_type == 'sudoku':
                        print(f"\n{'='*60}")
                        print("2. COMPLETION FROM PARTIAL QUIZ (CCDD)")
                        print(f"{'='*60}")

                        # Initialize discrete state from quiz
                        x_t_comp = torch.where(test_quiz != 0, test_quiz, torch.full_like(test_quiz, model.mask_index))
                        num_known = (test_quiz != 0).sum().item()
                        print(f"Starting from partial quiz with {num_known}/{n_samples * seq_len} known values")

                        # Initialize continuous latent from noise
                        z_t_comp = torch.randn(n_samples, seq_len, model.latent_dim, device=device)

                        # Compute unmasking schedule (only for masked positions)
                        initial_mask_comp = (x_t_comp == model.mask_index)
                        num_transfer_tokens_comp = get_num_transfer_tokens(initial_mask_comp, ccdd_sampling_steps)

                        # Denoising loop for completion
                        for k in range(ccdd_sampling_steps):
                            t_k = time_schedule[k]
                            t_next = time_schedule[k + 1]
                            t_tensor = torch.full((n_samples,), t_k.item(), device=device)

                            # Model prediction
                            # Scale continuous latent by ccdd_continuous_coef
                            epsilon_pred, logits_pred = model(x_t_comp, ccdd_continuous_coef * z_t_comp, t_tensor)

                            # ===== Discrete reverse: MDM-style (only unmask unknown positions) =====
                            # Deterministic argmax
                            x0_pred = torch.argmax(logits_pred, dim=-1)

                            # Compute confidence
                            probs = F.softmax(logits_pred, dim=-1)
                            confidence = torch.gather(probs, dim=-1, index=x0_pred.unsqueeze(-1)).squeeze(-1)

                            # Only consider masked positions (not known quiz positions)
                            mask_positions = (x_t_comp == model.mask_index)
                            neg_inf = torch.tensor(float("-inf"), device=device)
                            confidence = torch.where(mask_positions, confidence, neg_inf)

                            # Keep both known quiz positions AND unmasked predictions unchanged
                            x0_pred = torch.where(mask_positions, x0_pred, x_t_comp)

                            # Select top-k confident masked tokens to unmask
                            transfer_index = torch.zeros_like(x0_pred, dtype=torch.bool, device=device)
                            for j in range(n_samples):
                                if num_transfer_tokens_comp[j, k] > 0:
                                    _, select_indices = torch.topk(confidence[j], k=num_transfer_tokens_comp[j, k])
                                    transfer_index[j, select_indices] = True

                            x_t_comp[transfer_index] = x0_pred[transfer_index]

                            # Continuous reverse (same as generation)
                            t_k_discrete = int(t_k.item() * num_timesteps)
                            t_k_discrete = min(max(0, t_k_discrete), num_timesteps - 1)
                            t_next_discrete = int(t_next.item() * num_timesteps)
                            t_next_discrete = min(max(0, t_next_discrete), num_timesteps - 1)

                            alpha_k = alphas_cumprod[t_k_discrete]
                            sigma_k = torch.sqrt(1.0 - alpha_k)
                            sqrt_alpha_k = torch.sqrt(alpha_k)
                            z_0_pred = (z_t_comp - sigma_k * epsilon_pred) / sqrt_alpha_k

                            if k < ccdd_sampling_steps - 1:
                                alpha_next = alphas_cumprod[t_next_discrete]
                                sigma_next = torch.sqrt(1.0 - alpha_next)
                                sqrt_alpha_next = torch.sqrt(alpha_next)
                                z_t_mean = sqrt_alpha_next * z_0_pred + sigma_next * epsilon_pred

                                if ccdd_eta_ddpm > 0.0:
                                    sigma_k_next = torch.sqrt((1 - alpha_k / alpha_next) * (1 - alpha_next) / (1 - alpha_k))
                                    noise = torch.randn_like(z_t_comp)
                                    z_t_comp = z_t_mean + ccdd_eta_ddpm * sigma_k_next * noise
                                else:
                                    z_t_comp = z_t_mean
                            else:
                                z_t_comp = z_0_pred

                        # Final predictions for completion
                        final_preds_comp = x_t_comp

                        # Display completion results
                        print("\nCompletion samples:")
                        evaluate_and_display_sudoku(
                            final_preds_comp,
                            n_samples,
                            mode_str="Completion",
                            writer=writer,
                            step=step,
                            prefix="completion",
                            quiz_data=test_quiz,
                            max_display=5
                        )

        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)

    # Close output file
    print(f"\nAll outputs saved to: {exp_dir}")
    builtins.print = builtin_print
    out_f.close()


if __name__ == '__main__':
    fire.Fire(main)

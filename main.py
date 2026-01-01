"""
Diffusion Language Models for Discrete Data

Datasets supported:
- sequential:    sequences length 4, [i, i+1, i+2, i+3] [~100%]
- sudoku:        full Sudoku solutions (9x9 grids)

Model types supported:
- continuous:    Continuous diffusion in embedding space (default)
- masked:        Masked diffusion model (MDM) - simpler discrete approach
- dva:           Diffusion vs AR masked diffusion with pre-trained GPT2-style model
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

  # DVA (Diffusion vs AR) masked diffusion on sequential data
  python main.py --dataset sequential --model_type dva --steps 10000

  # Masked diffusion on sudoku (solution only - default)
  python main.py --dataset sudoku --model_type masked --batch_size 256 --steps 50000

  # Masked diffusion on sudoku with quiz+solution concatenated as input
  python main.py --dataset sudoku --model_type masked --sudoku_input_type quiz_solution --batch_size 256 --steps 50000
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
    SimpleDiffusionModel, MaskedPredictor, llada_mask,
    add_gumbel_noise, get_num_transfer_tokens
)
from dataset import create_simple_dataset
# from dataset import load_sudoku_dataset_npy as load_sudoku_dataset
from dataset import load_sudoku_dataset_csv as load_sudoku_dataset
from dataset import load_text8_dataset, decode_text8_tokens
# from dataset import load_sudoku_dataset
from model import dva_model, dva_tokenizer

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
        # Extract solution portion if in quiz+solution mode
        if final_preds.shape[1] == 164:
            # Format: quiz(0-80) + SEP(81) + solution(82-162) + EOF(163)
            solution = final_preds[i, 82:163]  # Extract solution portion (81 tokens)
        else:
            solution = final_preds[i]  # Already just the solution

        is_valid, score = is_valid_sudoku(solution.cpu())
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


def evaluate_text8_samples(final_preds, n_samples, mode_str="Sampling"):
    """
    Evaluate text8 samples - compute space ratio and letter diversity.

    Args:
        final_preds: Tensor of predicted sequences (n_samples, seq_len)
        n_samples: Number of samples to evaluate
        mode_str: String prefix for print statements

    Returns:
        dict with keys: avg_space_ratio, letter_diversity, unique_chars
    """
    space_ratios = []
    all_chars = []

    for i in range(n_samples):
        tokens = final_preds[i].cpu().tolist()
        space_count = sum(1 for t in tokens if t == 26)
        space_ratios.append(space_count / len(tokens))
        all_chars.extend(tokens)

    avg_space_ratio = sum(space_ratios) / len(space_ratios) if space_ratios else 0.0
    unique_chars = len(set(all_chars))
    letter_diversity = unique_chars / 27.0

    print(f"\n{mode_str} - Space ratio: {avg_space_ratio:.3f}")
    print(f"{mode_str} - Letter diversity: {letter_diversity:.3f} ({unique_chars}/27)")

    return {
        'avg_space_ratio': avg_space_ratio,
        'letter_diversity': letter_diversity,
        'unique_chars': unique_chars
    }


def display_sudoku_samples(final_preds, n_samples, quiz_data=None, max_display=5):
    """
    Display Sudoku samples (for generation or completion).

    Args:
        final_preds: Tensor of predicted Sudoku grids
            - Shape (n_samples, 81) for solution-only mode
            - Shape (n_samples, 162) for quiz+solution mode
        n_samples: Total number of samples available
        quiz_data: Optional quiz data for completion mode (n_samples, 81)
        max_display: Maximum number of samples to display
    """
    display_count = min(max_display, n_samples)

    # Check if predictions are in quiz+solution format (164 tokens with SEP/EOF)
    if final_preds.shape[1] == 164:
        # Quiz+solution mode: extract quiz and solution parts
        # Format: quiz(0-80) + SEP(81) + solution(82-162) + EOF(163)
        for i in range(display_count):
            quiz_part = final_preds[i, :81].reshape(9, 9).cpu().numpy()
            sep_token = final_preds[i, 81].item()
            solution_part = final_preds[i, 82:163].reshape(9, 9).cpu().numpy()
            eof_token = final_preds[i, 163].item()
            print(f"\nSample {i+1}:")
            print("Quiz (input):        Solution (predicted):")
            for row_idx in range(9):
                quiz_row = ' '.join([str(int(v)) if v != 0 else '.' for v in quiz_part[row_idx]])
                sol_row = ' '.join([str(int(v)) for v in solution_part[row_idx]])
                print(f"{quiz_row}    {sol_row}")
            print(f"SEP token: {sep_token}, EOF token: {eof_token}")
            print()
    elif quiz_data is not None:
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


def display_text8_samples(final_preds, n_samples, max_display=10):
    """
    Display text8 samples (decoded text).

    Args:
        final_preds: Tensor of predicted sequences (n_samples, seq_len)
        n_samples: Total number of samples available
        max_display: Maximum number of samples to display
    """
    display_count = min(max_display, n_samples)
    for i in range(display_count):
        decoded = decode_text8_tokens(final_preds[i])
        # Truncate if too long
        if len(decoded) > 200:
            decoded = decoded[:200] + "..."
        print(f"Sample {i+1}: {decoded}")


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


def evaluate_and_display_text8(final_preds, n_samples, mode_str, writer=None, step=None,
                                prefix: Optional[str] = None, max_display: int = 10):
    """
    Display text8 samples, compute metrics, and optionally log them to TensorBoard.
    """
    display_text8_samples(final_preds, n_samples, max_display=max_display)
    results = evaluate_text8_samples(final_preds, n_samples, mode_str=mode_str)

    if writer is not None and step is not None:
        metric_prefix = prefix or mode_str.lower()
        writer.add_scalar(f'Sampling/{metric_prefix}_space_ratio',
                         results['avg_space_ratio'], step)
        writer.add_scalar(f'Sampling/{metric_prefix}_letter_diversity',
                         results['letter_diversity'], step)

    return results


def topk_masking_dva(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    Helper function for DVA topk decoding (matching trainer.py lines 215-232).

    Args:
        scores: [b, n] - confidence scores
        cutoff_len: [b, 1] - number of tokens to keep masked
        stochastic: bool - whether to add Gumbel noise
        temp: float - temperature for Gumbel noise

    Returns:
        mask: [b, n] - True for tokens to keep masked (lowest confidence)
    """
    if stochastic:
        # Add Gumbel noise for stochastic sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores

    # Sort scores and find cutoff threshold
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)

    # Mask positions with scores below cutoff
    masking = _scores < cutoff
    return masking


def topk_decoding_dva(x0, x0_scores, decoding_strategy, init_maskable_mask, t, max_step, noise):
    """
    Helper function for DVA topk decoding (matching trainer.py lines 235-276).

    Args:
        x0: [b, n] - predicted tokens
        x0_scores: [b, n] - confidence scores
        decoding_strategy: str - "<topk_mode>-<schedule>" (e.g., "stochastic0.5-linear")
        init_maskable_mask: [b, n] - mask of initially maskable positions
        t: int - current timestep
        max_step: int - total number of steps
        noise: int or Tensor - mask token id or noise values

    Returns:
        xt: [b, n] - tokens with low-confidence positions masked
    """
    # Parse decoding strategy
    topk_mode, schedule = decoding_strategy.split("-")

    # Compute masking rate (proportion of tokens to keep masked)
    if schedule == "linear":
        rate = t / max_step
    elif schedule == "cosine":
        rate = np.cos((max_step - t) / max_step * np.pi * 0.5)
    else:
        raise NotImplementedError(f"Schedule {schedule} not implemented")

    # Compute cutoff length (number of tokens to keep masked)
    cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()

    # Set scores of non-maskable positions to high value so they won't be selected
    _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)

    # Select lowest-confidence tokens to mask
    if topk_mode.startswith("stochastic"):
        noise_scale = float(topk_mode.replace("stochastic", ""))
        lowest_k_mask = topk_masking_dva(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
    elif topk_mode == "deterministic":
        lowest_k_mask = topk_masking_dva(_scores_for_topk, cutoff_len, stochastic=False)
    else:
        raise NotImplementedError(f"Topk mode {topk_mode} not implemented")

    # Apply masking: recovered tokens can also be remasked based on current scores
    if isinstance(noise, torch.Tensor):
        xt = x0.masked_scatter(lowest_k_mask, noise[lowest_k_mask])
    elif isinstance(noise, (int, float)):
        xt = x0.masked_fill(lowest_k_mask, noise)
    else:
        raise NotImplementedError("noise should be either a tensor or a scalar")

    return xt


def compute_perplexity_text8(model, test_data, model_type, device, batch_size=128, **kwargs):
    """
    Compute perplexity on text8 test dataset using ELBO evaluation.

    For masked models: Samples multiple t values and computes ELBO with 1/t weighting on masked positions
    For continuous models: Samples multiple t values with noise and computes ELBO with SNR weighting (snr_prime)
    For combined/ccdd models: Evaluates at t=0 (clean reconstruction)

    Perplexity = exp(average negative log-likelihood per token)

    Args:
        model: The trained model
        test_data: Test dataset tensor [N, seq_len]
        model_type: 'masked', 'dva', 'continuous', 'combined', or 'ccdd'
        device: torch device
        batch_size: Batch size for evaluation
        **kwargs: Model-specific parameters
            - num_t_samples: Number of t samples for ELBO (masked/dva only, default: 10)
            - tokenizer: Required for dva model (to get mask_token_id)
            - For continuous: embedding, sqrt_alphas_cumprod, num_timesteps
            - For combined: embedding, combined_coef, combine_method
            - For ccdd: ccdd_continuous_coef

    Returns:
        dict with keys: perplexity, avg_nll, avg_nll_bits, accuracy
    """
    model.eval()

    num_test_samples = test_data.shape[0]
    seq_len = test_data.shape[1]
    total_nll = 0.0
    total_tokens = 0
    total_correct = 0

    num_batches = (num_test_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_test_samples)
            x = test_data[start_idx:end_idx].to(device)
            current_batch_size = x.shape[0]

            if model_type == 'masked':
                # Proper ELBO evaluation - sample multiple t values with 1/t importance weighting
                num_t_samples = kwargs.get('num_t_samples', 10)

                batch_nll = 0.0
                for _ in range(num_t_samples):
                    # Sample masking ratio uniformly from [0, 1]
                    t = torch.rand((current_batch_size,), device=device)

                    # Create masked input
                    xt = llada_mask(x, t=t, mask_index=model.mask_index)

                    # Get model predictions
                    logits = model._forward_without_loss(xt, t)

                    # Compute NLL on ALL positions
                    mask = (xt == model.mask_index)

                    if mask.any():
                        from lib.ops import cross_entropy
                        nll_per_token = cross_entropy(logits, x)  # [B, L]

                        # Weight by 1/t (importance sampling correction) and apply only to masked positions
                        t_expanded = t.view(-1, 1).expand_as(nll_per_token)
                        weighted_nll = nll_per_token / (t_expanded + 1e-8)

                        # Sum weighted NLL only over MASKED positions
                        masked_weighted_nll = weighted_nll * mask.float()
                        batch_nll += masked_weighted_nll.sum().item()

                # Average over t samples
                total_nll += batch_nll / num_t_samples
                total_tokens += current_batch_size * seq_len

                # Accuracy: evaluate at t=0.5 for consistency
                t_half = torch.full((current_batch_size,), 0.5, device=device)
                xt_half = llada_mask(x, t=t_half, mask_index=model.mask_index)
                logits_half = model._forward_without_loss(xt_half, t_half)
                preds = logits_half.argmax(dim=-1)
                total_correct += (preds == x).sum().item()

            elif model_type == 'dva':
                # DVA ELBO evaluation - similar to masked but using DVA's diffusion steps
                num_t_samples = kwargs.get('num_t_samples', 10)
                tokenizer = kwargs['tokenizer']
                vocab_size = model.vocab_size
                num_timesteps_dva = model.diffusion_args.diffusion_steps

                batch_nll = 0.0
                for _ in range(num_t_samples):
                    # Sample timestep uniformly from [0, T-1]
                    t = torch.randint(0, num_timesteps_dva, (current_batch_size,), device=device)

                    # Create masked input using DVA's q_sample logic
                    u = torch.rand_like(x, dtype=torch.float)
                    t_mask = (u < ((t + 1) / num_timesteps_dva)[:, None])
                    xt = x.clone()
                    xt[t_mask] = tokenizer.mask_token_id

                    # Get model predictions
                    attention_mask = torch.ones_like(xt)
                    logits = model(xt, t, attention_mask=attention_mask)
                    logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)

                    # Compute NLL on masked positions
                    mask = t_mask
                    if mask.any():
                        from lib.ops import cross_entropy
                        nll_per_token = cross_entropy(logits, x)  # [B, L]

                        # Weight by 1/(t+1) (importance sampling correction)
                        t_expanded = (t + 1).view(-1, 1).expand_as(nll_per_token).float()
                        weighted_nll = nll_per_token / (t_expanded + 1e-8)

                        # Sum weighted NLL only over MASKED positions
                        masked_weighted_nll = weighted_nll * mask.float()
                        batch_nll += masked_weighted_nll.sum().item()

                # Average over t samples
                total_nll += batch_nll / num_t_samples
                total_tokens += current_batch_size * seq_len

                # Accuracy: evaluate at t=10 (mid-range timestep) for consistency
                t_mid = torch.full((current_batch_size,), num_timesteps_dva // 2, device=device)
                u_mid = torch.rand_like(x, dtype=torch.float)
                t_mask_mid = (u_mid < ((t_mid + 1) / num_timesteps_dva)[:, None])
                xt_mid = x.clone()
                xt_mid[t_mask_mid] = tokenizer.mask_token_id
                attention_mask_mid = torch.ones_like(xt_mid)
                logits_mid = model(xt_mid, t_mid, attention_mask=attention_mask_mid)
                logits_mid = torch.cat([logits_mid[:, 0:1], logits_mid[:, :-1]], dim=1)
                preds = logits_mid.argmax(dim=-1)
                total_correct += (preds == x).sum().item()

            elif model_type == 'continuous':
                # ELBO evaluation matching plaid implementation
                # plaid/train.py lines 306-345, 408-413
                embedding = kwargs['embedding']
                sqrt_alphas_cumprod = kwargs['sqrt_alphas_cumprod']
                sqrt_one_minus_alphas_cumprod = kwargs.get('sqrt_one_minus_alphas_cumprod')
                num_timesteps = kwargs.get('num_timesteps', 1000)
                num_t_samples = kwargs.get('num_t_samples', 10)
                gamma_table = kwargs['gamma_table']
                gamma_prime_table = kwargs['gamma_prime_table']

                # Get clean embeddings
                x_embed = embedding(x)

                batch_nll = 0.0
                for _ in range(num_t_samples):
                    # Sample timesteps uniformly from [0, num_timesteps)
                    t = torch.randint(0, num_timesteps, (current_batch_size,), device=device)

                    # Get noise schedule values
                    sqrt_alpha_t = sqrt_alphas_cumprod[t][:, None, None]
                    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t][:, None, None]

                    # Add noise: z_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
                    noise = torch.randn_like(x_embed)
                    z_t = sqrt_alpha_t * x_embed + sqrt_one_minus_alpha_t * noise

                    # Convert to continuous time [0, 1]
                    t_continuous = t.float() / num_timesteps

                    # Get model predictions and reconstructed embeddings
                    logits = model(z_t, t_continuous)

                    # Plaid uses cross_entropy for reconstruction (reconst_loss)
                    # and squared error for diffusion (diffusion_loss)
                    # For perplexity, we use cross_entropy at all timesteps
                    from lib.ops import cross_entropy
                    nll_per_token = cross_entropy(logits, x)  # [B, L]
                    nll_per_sample = nll_per_token.mean(dim=1)  # [B] - match plaid's .mean(dim=1)

                    # Apply SNR weighting (matching plaid line 322)
                    # diffusion_loss = -0.5 * snr_prime * diff_base
                    gamma_t = gamma_table[t]
                    gamma_prime_t = gamma_prime_table[t]
                    snr_prime = -torch.exp(-gamma_t) * gamma_prime_t
                    weighted_nll = -0.5 * snr_prime * nll_per_sample  # [B]

                    # Sum over batch (matching plaid's .sum())
                    batch_nll += weighted_nll.sum().item()

                # Average NLL per sample
                avg_nll_per_sample = batch_nll / (num_t_samples * current_batch_size)

                # Accumulate total NLL (plaid multiplies by X.numel() = batch_size * seq_len)
                total_nll += avg_nll_per_sample * current_batch_size * seq_len
                total_tokens += current_batch_size * seq_len

                # Accuracy: evaluate at t=0 for consistency
                t_zero = torch.zeros(current_batch_size, dtype=torch.long, device=device)
                sqrt_alpha_0 = sqrt_alphas_cumprod[t_zero][:, None, None]
                z_clean = sqrt_alpha_0 * x_embed
                t_zero_continuous = t_zero.float() / num_timesteps
                logits_clean = model(z_clean, t_zero_continuous)
                preds = logits_clean.argmax(dim=-1)
                total_correct += (preds == x).sum().item()

            elif model_type == 'combined':
                # Evaluate at t=0 for discrete only
                embedding = kwargs['embedding']
                combined_coef = kwargs.get('combined_coef', 1.0)

                t_zero = torch.zeros(current_batch_size, device=device)
                xt = x.clone()
                z_disc = model.embed(xt)
                z_t = torch.zeros_like(z_disc)

                combine_method = kwargs.get('combine_method', 'add')
                if combine_method == 'add':
                    z_combined = z_disc + combined_coef * z_t
                else:
                    z_combined = torch.cat([z_disc, combined_coef * z_t], dim=-1)

                logits = model.forward_emb2logits(z_combined, t_zero, t_zero)

                from lib.ops import cross_entropy
                nll_per_token = cross_entropy(logits, x)
                total_nll += nll_per_token.sum().item()
                total_tokens += current_batch_size * seq_len

                preds = logits.argmax(dim=-1)
                total_correct += (preds == x).sum().item()

            elif model_type == 'ccdd':
                # Evaluate discrete likelihood at t=0
                ccdd_continuous_coef = kwargs.get('ccdd_continuous_coef', 1.0)

                x_t = x.clone()
                z_t = torch.zeros(current_batch_size, seq_len, model.latent_dim, device=device)
                t_zero = torch.zeros(current_batch_size, device=device)

                epsilon_pred, logits_pred = model(x_t, ccdd_continuous_coef * z_t, t_zero)

                from lib.ops import cross_entropy
                nll_per_token = cross_entropy(logits_pred, x)
                total_nll += nll_per_token.sum().item()
                total_tokens += current_batch_size * seq_len

                preds = logits_pred.argmax(dim=-1)
                total_correct += (preds == x).sum().item()

    # Compute final metrics
    avg_nll = total_nll / total_tokens
    avg_nll_bits = avg_nll / torch.log(torch.tensor(2.0)).item()
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    accuracy = total_correct / total_tokens

    model.train()

    return {
        'perplexity': perplexity,
        'avg_nll': avg_nll,
        'avg_nll_bits': avg_nll_bits,
        'accuracy': accuracy,
    }


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

    # Sudoku input type: 'solution_only' (default) or 'quiz_solution'
    sudoku_input_type = str(args.get('sudoku_input_type', 'solution_only')).lower()
    if sudoku_input_type not in ['solution_only', 'quiz_solution']:
        raise ValueError(f"sudoku_input_type must be 'solution_only' or 'quiz_solution', got '{sudoku_input_type}'")

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
        # Vocab: 0-9 (digits) + mask token (10) + SEP token (11) + EOF token (12)
        vocab_size = 13

        # Define special token indices
        SEP_TOKEN_ID = 11
        EOF_TOKEN_ID = 12

        # Sequence length depends on input type
        if sudoku_input_type == 'solution_only':
            seq_len = 81  # 9x9 grid (solution only)
        elif sudoku_input_type == 'quiz_solution':
            seq_len = 164  # quiz (81) + SEP (1) + solution (81) + EOF (1)

        data = None
        test_data = None
        test_quiz = None
        if not sampling_only:
            data = load_sudoku_dataset(sudoku_train_path, input_type=sudoku_input_type,
                                      sep_token_id=SEP_TOKEN_ID, eof_token_id=EOF_TOKEN_ID)
            test_quiz, test_data = load_sudoku_dataset(sudoku_test_path, input_type='solution_only')  # Test always returns (quiz, solution) separately

        print(f"[sudoku] Input type: {sudoku_input_type}")
        print(f"[sudoku] Vocabulary size: {vocab_size} (0-9 digits + mask + SEP + EOF)")
        print(f"[sudoku] Training sequence length: {seq_len}")
        if sudoku_input_type == 'quiz_solution':
            print(f"[sudoku] Format: quiz(81) + SEP(1) + solution(81) + EOF(1) = {seq_len} tokens")

    elif dataset_type == 'sequential':
        # original 4-token toy sequence
        vocab_size = 10
        seq_len = 4
        data = None
        test_data = None
        if not sampling_only:
            data = create_simple_dataset()

    elif dataset_type == 'text8':
        # text8 dataset with 27-token vocabulary (a-z + space)
        vocab_size = 27  # a-z (0-25) + space (26)
        seq_len = args.get('seq_len', 64)  # Configurable, default 64
        data = None
        test_data = None

        if not sampling_only:
            text8_stride = args.get('text8_stride', None)  # Default: seq_len // 2

            # Load train and validation splits and combine them
            print("[text8] Loading train and validation splits for training...")
            train_data = load_text8_dataset(split='train', seq_len=seq_len, stride=text8_stride)
            val_data = load_text8_dataset(split='validation', seq_len=seq_len, stride=text8_stride)
            data = torch.cat([train_data, val_data], dim=0)
            print(f"[text8] Combined training data shape: {data.shape}")

            # Load test split for perplexity evaluation (no overlap)
            test8_eval_stride = args.get('text8_eval_stride', seq_len)
            test_data = load_text8_dataset(split='test', seq_len=seq_len, stride=test8_eval_stride)
            print(f"[text8] Test data for perplexity evaluation: {test_data.shape}")

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
        # model = dva_model.to(device)
        print(f"Using Masked Diffusion Model (MDM)")
        print(f"  Architecture: SAME as Continuous Model")
        print(f"  vocab_size: {vocab_size}, seq_len: {seq_len}")
        print(f"  embed_dim: {embed_dim}, hidden_dim: {hidden_dim}")
        print(f"  n_heads: {n_heads}, n_layers: {n_blocks}")
        print(f"  positional_encoding: {positional_encoding}")
        print(f"  transformer_block_type: {transformer_block_type}")
    elif model_type == 'dva':
        # DVA (Diffusion vs AR) Masked Diffusion Model
        embedding = None
        model = dva_model.to(device)
        tokenizer = dva_tokenizer
        vocab_size = model.vocab_size
        print(f"Using DVA Masked Diffusion Model")
        print(f"  Architecture: Pre-trained GPT2-style model with diffusion wrapper")
        print(f"  vocab_size: {vocab_size}")
        print(f"  embed_dim: {model.embed_dim}")
        print(f"  hidden_dim: {model.hidden_dim}")
        print(f"  mask_index: {model.mask_index}")
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
        elif model_type == 'masked' or model_type == 'combined' or model_type == 'ccdd' or model_type == 'dva':
            # For masked/combined/ccdd/dva model, validate config if available
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
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
        else:
            raise Exception("Unsupported model type for loading checkpoint.")

    ### 3. Optimizers - setup based on model type
    if model_type == 'masked' or model_type == 'combined' or model_type == 'ccdd' or model_type == 'dva':
        # Masked/Combined/CCDD/DVA model: single optimizer for all parameters
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
                # t = torch.rand((x.shape[0],), device=device)  # Range [0.0, 1.0]

                t = torch.randint(0, 20, (x.shape[0], ), device=x.device).float() / 20
                
                # Create masked input using llada_mask
                if dataset_type == 'sudoku' and sudoku_input_type == 'quiz_solution':
                    # For quiz+solution mode with SEP/EOF tokens:
                    # Format: quiz(0-80) + SEP(81) + solution(82-162) + EOF(163)
                    # Only mask the solution portion (positions 82-162)
                    quiz_and_sep = x[:, :82]  # quiz (81) + SEP (1) - keep unmasked
                    solution_part = x[:, 82:163]  # solution (81 tokens) - mask this
                    eof_token = x[:, 163:164]  # EOF (1) - keep unmasked
                    

                    # Apply masking only to solution portion
                    solution_masked = llada_mask(solution_part, t=t, mask_index=model.mask_index)

                    # Concatenate: quiz+SEP (unmasked) + solution (masked) + EOF (unmasked)
                    xt = torch.cat([quiz_and_sep, solution_masked, eof_token], dim=1)
                else:
                    # Default: mask the entire sequence
                    xt = llada_mask(x, t=t, mask_index=model.mask_index)

                # Forward pass - returns loss directly
                loss_diff = model(xt, x, t)

                # Dispersive loss: directly on token embeddings (exclude mask token)
                dispersive_loss = get_dispersion_loss(
                    model.embed.weight[:-1, :].repeat(batch_size, 1)
                ) * 0

                # Total loss: diffusion loss + dispersive loss
                loss = loss_diff + dispersive_loss

                # Backward and optimize
                optimizer_model.zero_grad()
                loss.backward()
                optimizer_model.step()
                scheduler_model.step()

                # Track losses
                total_losses.append(loss.item())
                recon_losses.append(0.0)  # Not applicable for masked model
                diffusion_losses.append(loss_diff.item())
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

                        # Compute accuracy based on mode
                        if dataset_type == 'sudoku' and sudoku_input_type == 'quiz_solution':
                            # For quiz+solution mode: only compute accuracy on solution portion (positions 82-162)
                            solution_preds = preds[:, 82:163]
                            solution_targets = x[:, 82:163]
                            overall_acc = (solution_preds == solution_targets).float().mean().item()
                        else:
                            # Default: compute overall accuracy on entire sequence
                            overall_acc = (preds == x).float().mean().item()

                    # Print progress with validation metrics
                    avg_mask_ratio = mask.float().mean().item()
                    print(f"[Step {step+1:>6}/{steps}] loss={loss.item():.4f} (diff={loss_diff.item():.4f} disp={dispersive_loss.item():.4f}) | "
                          f"mask_acc={acc:.4f} overall_acc={overall_acc:.4f} | "
                          f"mask_ratio={avg_mask_ratio:.2f} lr={scheduler_model.get_last_lr()[0]:.2e}")

                    # Display sample predictions as text for text8
                    if dataset_type == 'text8' and step % (print_freq * 5) == 0:
                        print("  Sample predictions (masked positions shown as [?]):")
                        for i in range(min(3, batch_size)):
                            # Decode with mask indicators
                            input_chars = []
                            for j, token in enumerate(xt[i].tolist()):
                                if token == model.mask_index:
                                    input_chars.append('[?]')
                                else:
                                    decoded = decode_text8_tokens([token])
                                    input_chars.append(decoded)
                            input_text = ''.join(input_chars)[:60] + "..."

                            target_text = decode_text8_tokens(x[i])[:60] + "..."
                            pred_text = decode_text8_tokens(preds[i])[:60] + "..."

                            print(f"    Input:  {input_text}")
                            print(f"    Target: {target_text}")
                            print(f"    Pred:   {pred_text}")
                            print()

                    # Log to TensorBoard
                    writer.add_scalar('Loss/total', loss.item(), step)
                    writer.add_scalar('Loss/diffusion', loss_diff.item(), step)
                    writer.add_scalar('Loss/dispersive', dispersive_loss.item(), step)
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

                # PERPLEXITY EVALUATION (every 1000 steps)
                if step % 1000 == 0 and dataset_type == 'text8' and test_data is not None:
                    num_t_samples = args.get('num_t_samples', 10)
                    print(f"\n{'='*60}")
                    print(f"PERPLEXITY EVALUATION (Masked Model) - Step {step}")
                    print(f"Using ELBO with {num_t_samples} t-samples and 1/t weighting on masked positions")
                    print(f"{'='*60}")

                    perplexity_results = compute_perplexity_text8(
                        model=model,
                        test_data=test_data,
                        model_type='masked',
                        device=device,
                        batch_size=args.get('eval_batch_size', 128),
                        num_t_samples=num_t_samples
                    )

                    print(f"Test Perplexity: {perplexity_results['perplexity']:.4f}")
                    print(f"Test NLL (bits/token): {perplexity_results['avg_nll_bits']:.4f}")
                    print(f"Test Accuracy (t=0.5): {perplexity_results['accuracy']:.4f}")
                    print(f"{'='*60}\n")

                    writer.add_scalar('Perplexity/test_perplexity', perplexity_results['perplexity'], step)
                    writer.add_scalar('Perplexity/test_nll_bits', perplexity_results['avg_nll_bits'], step)
                    writer.add_scalar('Perplexity/test_accuracy', perplexity_results['accuracy'], step)

            elif model_type == 'dva':
                # ===== DVA MASKED DIFFUSION MODEL TRAINING =====
                # Based on CustomDiffusionTrainer from diffusion-vs-ar

                # Define src_mask (positions that should NOT be masked)
                # For sudoku quiz_solution mode: quiz+SEP+EOF should not be masked
                if dataset_type == 'sudoku' and sudoku_input_type == 'quiz_solution':
                    # Format: quiz(0-80) + SEP(81) + solution(82-162) + EOF(163)
                    # src_mask = True for positions that should NOT be masked
                    src_mask = torch.zeros_like(x, dtype=torch.bool)
                    src_mask[:, :82] = True   # quiz (81) + SEP (1)
                    src_mask[:, 163] = True   # EOF
                else:
                    # For other datasets, all positions are maskable
                    src_mask = torch.zeros_like(x, dtype=torch.bool)

                # maskable_mask = positions that CAN be masked (inverse of src_mask)
                maskable_mask = ~src_mask

                # Sample random timesteps for the batch
                num_timesteps_dva = model.diffusion_args.diffusion_steps
                t = torch.randint(0, num_timesteps_dva, (x.shape[0],), device=device)

                # Create masked input using q_sample method (matching trainer.py line 55-60)
                # Masking probability: (t+1) / T, but only on maskable positions
                u = torch.rand_like(x, dtype=torch.float)
                t_mask = (u < ((t + 1) / num_timesteps_dva)[:, None]) & maskable_mask

                # Apply mask to create x_t (masked input)
                xt = x.clone()
                xt[t_mask] = tokenizer.mask_token_id

                # Forward pass through model
                attention_mask = torch.ones_like(xt)
                logits = model(xt, t, attention_mask=attention_mask)

                # Shift logits for autoregressive-style loss
                logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)

                # Compute cross-entropy loss
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    x.reshape(-1),
                    reduction="none"
                ).float()

                # Only compute loss on masked positions
                loss = loss.masked_fill(~t_mask.reshape(-1), 0)

                # Token reweighting (focal loss style)
                if model.diffusion_args.token_reweighting:
                    alpha = model.diffusion_args.alpha
                    gamma = model.diffusion_args.gamma
                    loss = alpha * (1 - torch.exp(-loss)) ** gamma * loss

                # Time reweighting
                if model.diffusion_args.time_reweighting == 'original':
                    weight = 1 / (t + 1)[:, None].float()
                elif model.diffusion_args.time_reweighting == 'linear':
                    weight = (num_timesteps_dva - t)[:, None].float()
                else:
                    weight = t.new_ones((x.shape[0], 1)).float()

                weight = weight.expand(t_mask.size())
                loss_diff = (loss * weight.reshape(-1)).sum() / t_mask.sum()

                # Total loss
                loss = loss_diff

                # Backward and optimize
                optimizer_model.zero_grad()
                loss.backward()
                optimizer_model.step()
                scheduler_model.step()

                # Track losses
                total_losses.append(loss.item())
                recon_losses.append(0.0)
                diffusion_losses.append(loss_diff.item())
                prior_losses.append(0.0)

                # Validation: compute accuracy on masked positions
                if step % print_freq == 0 or step == steps - 1:
                    with torch.no_grad():
                        # Get predictions
                        preds = logits.argmax(dim=-1)

                        # Compute accuracy only on masked positions
                        mask = t_mask
                        if mask.any():
                            masked_preds = preds[mask]
                            masked_targets = x[mask]
                            acc = (masked_preds == masked_targets).float().mean().item()
                        else:
                            acc = 0.0

                        # Compute overall accuracy
                        if dataset_type == 'sudoku' and sudoku_input_type == 'quiz_solution':
                            # Only measure solution portion (positions 82-162)
                            pred_solution = preds[:, 82:163]
                            target_solution = x[:, 82:163]
                            overall_acc = (pred_solution == target_solution).float().mean().item()
                        else:
                            overall_acc = (preds == x).float().mean().item()

                    # Print progress
                    avg_mask_ratio = mask.float().mean().item()
                    print(f"[Step {step+1:>6}/{steps}] loss={loss.item():.4f} (diff={loss_diff.item():.4f}) | "
                          f"mask_acc={acc:.4f} overall_acc={overall_acc:.4f} | "
                          f"mask_ratio={avg_mask_ratio:.2f} lr={scheduler_model.get_last_lr()[0]:.2e}")

                    # Log to TensorBoard
                    writer.add_scalar('Loss/total', loss.item(), step)
                    writer.add_scalar('Loss/diffusion', loss_diff.item(), step)
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
                if dataset_type == 'sudoku' and sudoku_input_type == 'quiz_solution':
                    # Split into parts: quiz+SEP (0-81), solution (82-162), EOF (163)
                    quiz_sep_embed = x_embed[:, :82, :]       # [batch, 82, embed_dim]
                    solution_embed = x_embed[:, 82:163, :]    # [batch, 81, embed_dim]
                    eof_embed = x_embed[:, 163:164, :]        # [batch, 1, embed_dim]

                    # Only add noise to solution portion
                    solution_noise = torch.randn_like(solution_embed)
                    solution_z = sqrt_alpha_t * solution_embed + sqrt_one_minus_alpha_t * solution_noise

                    # Concatenate: quiz+SEP (clean) + solution (noisy) + EOF (clean)
                    z = torch.cat([quiz_sep_embed, solution_z, eof_embed], dim=1)
                else:
                    # Original behavior for other modes
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
                    if dataset_type == 'sudoku' and sudoku_input_type == 'quiz_solution':
                        # Only compute loss on solution portion (positions 82-162)
                        logits_solution = logits[:reconst_bs, 82:163, :]  # [reconst_bs, 81, vocab_size]
                        x_solution = x[:reconst_bs, 82:163]               # [reconst_bs, 81]
                        reconst_terms = lib_ops.cross_entropy(logits_solution, x_solution).mean(dim=1)
                        reconst_loss = reconst_terms.mean()
                    else:
                        reconst_terms = lib_ops.cross_entropy(logits[:reconst_bs], x[:reconst_bs]).mean(dim=1)
                        reconst_loss = reconst_terms.mean()
                else:
                    reconst_terms = torch.empty(0, device=device)
                    reconst_loss = torch.tensor(0.0, device=device)

                gamma_t = gamma_table[t]
                gamma_prime_t = gamma_prime_table[t]
                snr_prime = -torch.exp(-gamma_t) * gamma_prime_t

                if dataset_type == 'sudoku' and sudoku_input_type == 'quiz_solution':
                    # Only compute diffusion loss on solution portion (positions 82-162)
                    x_embed_solution = x_embed[:, 82:163, :]
                    x_reconst_solution = x_reconst[:, 82:163, :]
                    diff_base = (x_embed_solution - x_reconst_solution).pow(2).mean(dim=1).sum(dim=1)
                else:
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
                loss = loss + dispersive_loss

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

                # Always step embedding optimizer (not just when repae=True)
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

                        if dataset_type == 'sudoku' and sudoku_input_type == 'quiz_solution':
                            # Only measure solution portion (positions 82-162)
                            pred_solution = preds[:, 82:163]
                            target_solution = x[:, 82:163]
                            acc = (pred_solution == target_solution).float().mean().item()
                        else:
                            acc = (preds == x).float().mean().item()

                    # Display sample predictions as text for text8
                    if dataset_type == 'text8' and step % (print_freq * 5) == 0:
                        print("  Sample predictions (continuous model at t=0):")
                        for i in range(min(3, batch_size)):
                            target_text = decode_text8_tokens(x[i])[:80] + "..."
                            pred_text = decode_text8_tokens(preds[i])[:80] + "..."
                            print(f"    Target: {target_text}")
                            print(f"    Pred:   {pred_text}")
                            print()

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

                    # Print progress (removed debug print of embedding matrix)
                    print(f"[Step {step+1:>6}/{steps}] loss={loss.item():.4f} (recon={reconst_val:.4f} diff={diff_tail_val:.4f} prior={prior_loss.item():.4f} disp={dispersive_loss:.4f}) | "
                          f"acc@t=0={acc:.4f} lr={scheduler_model.get_last_lr()[0]:.2e}")

                # PERPLEXITY EVALUATION (every 1000 steps)
                if step % 1000 == 0 and dataset_type == 'text8' and test_data is not None:
                    num_t_samples = args.get('num_t_samples', 10)
                    print(f"\n{'='*60}")
                    print(f"PERPLEXITY EVALUATION (Continuous Model) - Step {step}")
                    print(f"Using ELBO with {num_t_samples} t-samples and SNR weighting")
                    print(f"{'='*60}")

                    perplexity_results = compute_perplexity_text8(
                        model=model,
                        test_data=test_data,
                        model_type='continuous',
                        device=device,
                        batch_size=args.get('eval_batch_size', 128),
                        embedding=embedding,
                        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                        num_timesteps=num_timesteps,
                        num_t_samples=num_t_samples,
                        gamma_table=gamma_table,
                        gamma_prime_table=gamma_prime_table
                    )

                    print(f"Test Perplexity: {perplexity_results['perplexity']:.4f}")
                    print(f"Test NLL (bits/token): {perplexity_results['avg_nll_bits']:.4f}")
                    print(f"Test Accuracy: {perplexity_results['accuracy']:.4f}")
                    print(f"{'='*60}\n")

                    writer.add_scalar('Perplexity/test_perplexity', perplexity_results['perplexity'], step)
                    writer.add_scalar('Perplexity/test_nll_bits', perplexity_results['avg_nll_bits'], step)
                    writer.add_scalar('Perplexity/test_accuracy', perplexity_results['accuracy'], step)

            elif model_type == 'combined':
                # ===== COMBINED MODEL TRAINING =====

                # Sample independent times for discrete and continuous processes
                t_disc = torch.rand((x.shape[0],), device=device)  # For discrete masking [0.0, 1.0]
                t_cont = torch.rand((x.shape[0],), device=device)  # For continuous noise [0.0, 1.0]

                # add mask (use t_disc for discrete masking)
                if dataset_type == 'sudoku' and sudoku_input_type == 'quiz_solution':
                    # Only mask solution portion (positions 82-162)
                    quiz_and_sep = x[:, :82]        # quiz(81) + SEP(1) - keep unmasked
                    solution_part = x[:, 82:163]    # solution(81) - mask this
                    eof_token = x[:, 163:164]       # EOF(1) - keep unmasked
                    
                    # Apply masking only to solution
                    solution_masked = llada_mask(solution_part, t=t_disc, mask_index=model.mask_index)

                    # Concatenate back
                    xt = torch.cat([quiz_and_sep, solution_masked, eof_token], dim=1)
                else:
                    # Original behavior for other modes
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
                if dataset_type == 'sudoku' and sudoku_input_type == 'quiz_solution':
                    # Only add noise to solution portion (positions 82-162)
                    quiz_sep_embed = x_clean_embed[:, :82, :]       # quiz + SEP (clean)
                    solution_embed = x_clean_embed[:, 82:163, :]    # solution
                    eof_embed = x_clean_embed[:, 163:164, :]        # EOF (clean)

                    # Only noise solution
                    solution_noise = torch.randn_like(solution_embed)
                    solution_epsilon = solution_noise  # Save for loss computation
                    solution_z_t = sqrt_alpha_t * solution_embed + sqrt_one_minus_alpha_t * solution_noise

                    # Create z_t based on combine_method
                    if combine_method == 'add':
                        # ADD mode: quiz+EOF are zero, solution is noisy
                        z_t = torch.cat([
                            torch.zeros_like(quiz_sep_embed),
                            solution_z_t,
                            torch.zeros_like(eof_embed)
                        ], dim=1)
                    else:  # concat
                        # CONCAT mode: quiz+EOF are clean, solution is noisy
                        z_t = torch.cat([
                            quiz_sep_embed,
                            solution_z_t,
                            eof_embed
                        ], dim=1)
                else:
                    # Original behavior for other modes
                    noise = torch.randn_like(x_clean_embed)  # [B, seq_len, embed_dim]
                    epsilon = noise  # Save for loss computation

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

                        # Also compute overall accuracy
                        if dataset_type == 'sudoku' and sudoku_input_type == 'quiz_solution':
                            # Only measure solution portion (positions 82-162)
                            pred_solution = preds[:, 82:163]
                            target_solution = x[:, 82:163]
                            overall_acc = (pred_solution == target_solution).float().mean().item()
                        else:
                            overall_acc = (preds == x).float().mean().item()

                    # Print progress with validation metrics
                    avg_mask_ratio = mask.float().mean().item()
                    print(f"[Step {step+1:>6}/{steps}] loss={loss.item():.4f} | "
                          f"mask_acc={acc:.4f} overall_acc={overall_acc:.4f} | "
                          f"mask_ratio={avg_mask_ratio:.2f} lr={scheduler_model.get_last_lr()[0]:.2e}")

                    # Display sample predictions as text for text8
                    if dataset_type == 'text8' and step % (print_freq * 5) == 0:
                        print("  Sample predictions (masked positions shown as [?]):")
                        for i in range(min(3, batch_size)):
                            # Decode with mask indicators
                            input_chars = []
                            for j, token in enumerate(xt[i].tolist()):
                                if token == model.mask_index:
                                    input_chars.append('[?]')
                                else:
                                    decoded = decode_text8_tokens([token])
                                    input_chars.append(decoded)
                            input_text = ''.join(input_chars)[:60] + "..."

                            target_text = decode_text8_tokens(x[i])[:60] + "..."
                            pred_text = decode_text8_tokens(preds[i])[:60] + "..."

                            print(f"    Input:  {input_text}")
                            print(f"    Target: {target_text}")
                            print(f"    Pred:   {pred_text}")
                            print()

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

                # PERPLEXITY EVALUATION (every 1000 steps)
                if step % 1000 == 0 and dataset_type == 'text8' and test_data is not None:
                    print(f"\n{'='*60}")
                    print(f"PERPLEXITY EVALUATION (Combined Model) - Step {step}")
                    print(f"{'='*60}")

                    perplexity_results = compute_perplexity_text8(
                        model=model,
                        test_data=test_data,
                        model_type='combined',
                        device=device,
                        batch_size=args.get('eval_batch_size', 128),
                        embedding=embedding,
                        combined_coef=combined_coef,
                        combine_method=combine_method
                    )

                    print(f"Test Perplexity: {perplexity_results['perplexity']:.4f}")
                    print(f"Test NLL (bits/token): {perplexity_results['avg_nll_bits']:.4f}")
                    print(f"Test Accuracy: {perplexity_results['accuracy']:.4f}")
                    print(f"{'='*60}\n")

                    writer.add_scalar('Perplexity/test_perplexity', perplexity_results['perplexity'], step)
                    writer.add_scalar('Perplexity/test_nll_bits', perplexity_results['avg_nll_bits'], step)
                    writer.add_scalar('Perplexity/test_accuracy', perplexity_results['accuracy'], step)

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

                # PERPLEXITY EVALUATION (every 1000 steps)
                if step % 1000 == 0 and dataset_type == 'text8' and test_data is not None:
                    print(f"\n{'='*60}")
                    print(f"PERPLEXITY EVALUATION (CCDD Model) - Step {step}")
                    print(f"{'='*60}")

                    perplexity_results = compute_perplexity_text8(
                        model=model,
                        test_data=test_data,
                        model_type='ccdd',
                        device=device,
                        batch_size=args.get('eval_batch_size', 128),
                        ccdd_continuous_coef=ccdd_continuous_coef
                    )

                    print(f"Test Perplexity: {perplexity_results['perplexity']:.4f}")
                    print(f"Test NLL (bits/token): {perplexity_results['avg_nll_bits']:.4f}")
                    print(f"Test Accuracy: {perplexity_results['accuracy']:.4f}")
                    print(f"{'='*60}\n")

                    writer.add_scalar('Perplexity/test_perplexity', perplexity_results['perplexity'], step)
                    writer.add_scalar('Perplexity/test_nll_bits', perplexity_results['avg_nll_bits'], step)
                    writer.add_scalar('Perplexity/test_accuracy', perplexity_results['accuracy'], step)


            # Save checkpoint every 10000 iterations
            if checkpoint_path and (step + 1) % 10000 == 0:
                # Create checkpoint filename with iteration number
                checkpoint_dir = os.path.dirname(checkpoint_path)
                checkpoint_name = f"checkpoint_{step + 1}.pt"
                iter_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)

                # Save checkpoint based on model type
                if model_type == 'masked' or model_type == 'combined' or model_type == 'ccdd' or model_type == 'dva':
                    # Masked/Combined/CCDD/DVA model: only save model state
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
                    elif model_type == 'dva':
                        checkpoint_config['diffusion_steps'] = model.diffusion_args.diffusion_steps
                        checkpoint_config['mask_index'] = model.mask_index

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
                            rand_test_ids = torch.randperm(test_quiz.size(0))[:n_samples]
                            test_quiz = test_quiz[rand_test_ids].to(device)
                            # test_quiz = test_quiz[:n_samples].to(device)
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

                        # Initialize completion based on input type
                        if sudoku_input_type == 'quiz_solution':
                            # test_quiz has full sequence [164] with quiz+SEP+solution+EOF
                            # Embed the full sequence
                            full_embed = torch.randn((test_quiz.shape[0], seq_len, embed_dim), dtype=torch.float32, device=device)
                            full_embed[:, :81, :] = embedding_matrix[test_quiz]  # [batch, 164, embed_dim]
                            full_embed[:, 81, :] = embedding_matrix[SEP_TOKEN_ID]
                            full_embed[:, 163:164, :] = embedding_matrix[EOF_TOKEN_ID]

                            # Split into parts
                            quiz_sep_embed = full_embed[:, :82, :]       # quiz + SEP
                            solution_embed = full_embed[:, 82:163, :]    # solution (ground truth, will be replaced)
                            eof_embed = full_embed[:, 163:164, :]        # EOF

                            # Initialize solution with pure noise
                            solution_noise = torch.randn_like(solution_embed) * sqrt_one_minus_alpha_start

                            # Concatenate: quiz+SEP (given) + solution (noise) + EOF (given)
                            z_comp = torch.cat([quiz_sep_embed, solution_noise, eof_embed], dim=1)
                        else:
                            # solution_only mode: original behavior
                            z_comp = torch.randn(n_samples, seq_len, embed_dim, device=device) * sqrt_one_minus_alpha_start

                        # Denoising loop for completion
                        for step_idx, t_discrete in enumerate(schedule.tolist()):
                            t_tensor = torch.full((n_samples,), t_discrete, dtype=torch.long, device=device)
                            t_continuous = t_tensor.float() / denom
                            logits = model(z_comp, t_continuous)
                            probs = F.softmax(logits, dim=-1)
                            x_reconst = probs @ embedding_matrix
                            pred_tokens = logits.argmax(dim=-1)

                            x_embed_disc = x_reconst
                            ### clamping
                            # x_embed_disc = embedding_matrix[pred_tokens]

                            # Inject ground truth quiz values at known positions
                            if sudoku_input_type == 'quiz_solution':
                                # Protect quiz+SEP+EOF, only update solution
                                quiz_sep_eof_protected = torch.cat([
                                    quiz_sep_embed,      # quiz + SEP (protected)
                                    x_embed_disc[:, 82:163, :],          # solution (denoised)
                                    eof_embed   # EOF (protected)
                                ], dim=1)
                                x_embed_disc = quiz_sep_eof_protected
                            else:
                                # solution_only mode: original behavior
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
                        if sudoku_input_type == 'quiz_solution':
                            # Preserve quiz+SEP+EOF, only keep predicted solution
                            # final_preds_comp = torch.cat([
                            #     test_quiz,              # quiz + SEP (given)
                            #     final_preds_comp[:, 82:163],    # solution (predicted)
                            #     torch.full_like(test_quiz[:, :1], EOF_TOKEN_ID)          # EOF (given)
                            # ], dim=1)
                            final_preds_comp = final_preds_comp[:, 82:163]
                            
                        else:
                            # solution_only mode: original behavior
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
                    elif dataset_type == 'text8':
                        evaluate_and_display_text8(
                            final_preds_gen,
                            n_samples,
                            mode_str="Generation",
                            writer=writer,
                            step=step,
                            prefix="generation",
                            max_display=10
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
                            rand_test_ids = torch.randperm(test_quiz.size(0))[:n_samples]
                            test_quiz = test_quiz[rand_test_ids].to(device)
                            # test_quiz = test_quiz[:n_samples].to(device)
                            print(f"Loaded test quiz data from {test_quiz_path} for completion evaluation")
                            print(f"Quiz data shape: {test_quiz.shape}")
                        else:
                            raise Exception("unexpected")

                    # Generation: start from fully masked
                    print(f"\nGenerating {n_samples} samples with {mdm_sampling_steps} unmasking steps (generation)...")
                    gen_xt = torch.full((n_samples, seq_len), model.mask_index, dtype=torch.long, device=device)

                    # For quiz+solution mode: provide quiz+SEP, mask only solution, add EOF
                    if dataset_type == 'sudoku' and sudoku_input_type == 'quiz_solution' and test_quiz is not None:
                        # Format: quiz(0-80) + SEP(81) + solution(82-162, masked) + EOF(163)
                        gen_xt[:, :81] = test_quiz[:n_samples, :81]  # Quiz portion
                        gen_xt[:, 81] = SEP_TOKEN_ID  # SEP token
                        # Positions 82-162 remain masked (solution to generate)
                        gen_xt[:, 163] = EOF_TOKEN_ID  # EOF token
                        print(f"  Quiz+solution mode: quiz+SEP+EOF provided, generating solution only")

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
                    elif dataset_type == 'text8':
                        evaluate_and_display_text8(
                            gen_preds,
                            n_samples,
                            mode_str="Generation",
                            writer=writer,
                            step=step,
                            prefix="generation",
                            max_display=10
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

                        if sudoku_input_type == 'quiz_solution':
                            # For quiz+solution mode: quiz+SEP+EOF given, only denoise solution
                            # Format: quiz(0-80) + SEP(81) + solution(82-162, masked) + EOF(163)
                            xt = torch.full((comp_samples, seq_len), model.mask_index, dtype=torch.long, device=device)
                            xt[:, :81] = test_quiz[:, :81]  # Quiz portion is given
                            xt[:, 81] = SEP_TOKEN_ID  # SEP token
                            # Positions 82-162 remain masked (solution to denoise)
                            xt[:, 163] = EOF_TOKEN_ID  # EOF token
                            num_known = 83 * comp_samples  # 81 (quiz) + 1 (SEP) + 1 (EOF)
                            print(f"Quiz+solution mode: quiz+SEP+EOF given (83 tokens), denoising solution portion only (81 tokens)")
                        else:
                            # Default mode: use quiz as partial input (non-zero values are kept)
                            xt = torch.where(test_quiz != 0, test_quiz, torch.full_like(test_quiz, model.mask_index))
                            num_known = (test_quiz != 0).sum().item()

                        print(f"Starting from partial input with {num_known}/{comp_samples * seq_len} known values")

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

            if step % 1000 == 0 and model_type == 'dva':
                # ===== DVA MODEL SAMPLING =====
                # Based on generate_samples method in trainer.py (lines 154-213)
                n_samples = args.get('n_samples', 100)
                dva_sampling_steps = model.diffusion_args.diffusion_steps

                with torch.no_grad():
                    model.eval()

                    # Define src_mask for the dataset
                    # For sudoku quiz_solution mode: quiz+SEP+EOF are given (not masked)
                    if dataset_type == 'sudoku' and sudoku_input_type == 'quiz_solution':
                        # Load test quiz data if available
                        test_quiz_path = args.get('test_quiz_path', 'data_vmd/sudoku_test.csv')
                        if os.path.exists(test_quiz_path):
                            test_quiz, _ = load_sudoku_dataset(test_quiz_path)
                            rand_test_ids = torch.randperm(test_quiz.size(0))[:n_samples]
                            test_quiz = test_quiz[rand_test_ids].to(device)

                            # Create x with quiz filled in
                            x_gen = torch.full((n_samples, seq_len), tokenizer.mask_token_id, dtype=torch.long, device=device)
                            x_gen[:, :81] = test_quiz[:n_samples, :81]  # Quiz portion
                            x_gen[:, 81] = SEP_TOKEN_ID  # SEP token
                            x_gen[:, 163] = EOF_TOKEN_ID  # EOF token

                            # src_mask: True for positions that should NOT be masked
                            src_mask = torch.zeros_like(x_gen, dtype=torch.bool)
                            src_mask[:, :82] = True   # quiz (81) + SEP (1)
                            src_mask[:, 163] = True   # EOF

                            print(f"\nGenerating {n_samples} samples (quiz+solution mode) with {dva_sampling_steps} denoising steps...")
                        else:
                            # No quiz data, generate from scratch
                            x_gen = torch.full((n_samples, seq_len), tokenizer.mask_token_id, dtype=torch.long, device=device)
                            src_mask = torch.zeros_like(x_gen, dtype=torch.bool)
                            print(f"\nGenerating {n_samples} samples with {dva_sampling_steps} denoising steps...")
                    else:
                        # For other datasets: generate from fully masked
                        x_gen = torch.full((n_samples, seq_len), tokenizer.mask_token_id, dtype=torch.long, device=device)
                        src_mask = torch.zeros_like(x_gen, dtype=torch.bool)
                        print(f"\nGenerating {n_samples} samples with {dva_sampling_steps} denoising steps...")

                    # init_maskable_mask: positions that CAN be masked (inverse of src_mask)
                    init_maskable_mask = ~src_mask
                    maskable_mask = init_maskable_mask.clone()
                    attention_mask = torch.ones_like(x_gen)

                    # Iterative denoising from T-1 to 0 (matching trainer.py lines 170-212)
                    for t_step in range(dva_sampling_steps - 1, -1, -1):
                        t_tensor = torch.full((n_samples,), t_step, device=device)

                        # Forward through model
                        logits = model(x_gen, t_tensor, attention_mask=attention_mask)
                        logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)

                        # Get predictions (matching trainer.py lines 183-185)
                        scores = torch.log_softmax(logits, dim=-1)
                        # Clip scores beyond vocab_size (trainer.py line 184)
                        scores[:, :, tokenizer.vocab_size:] = -1000
                        x0_scores, x0 = scores.max(-1)

                        # Keep non-maskable positions unchanged (trainer.py line 188)
                        x0 = x_gen.masked_scatter(maskable_mask, x0[maskable_mask])

                        if t_step > 0:
                            # Use topk decoding to select which tokens to unmask
                            if model.diffusion_args.topk_decoding:
                                # Use topk_decoding_dva helper function (trainer.py lines 194-202)
                                x_gen = topk_decoding_dva(
                                    x0,
                                    x0_scores,
                                    model.diffusion_args.decoding_strategy,
                                    init_maskable_mask,
                                    t_step,
                                    dva_sampling_steps,
                                    tokenizer.mask_token_id
                                )
                            else:
                                # Random unmasking (D3PM style) (trainer.py lines 204-210)
                                unmask_prob = 1 / (t_step + 1)
                                mask_to_x0 = torch.rand(x_gen.shape, device=device) < unmask_prob
                                # Don't unmask somewhere already unmasked
                                mask_to_x0 = torch.bitwise_and(mask_to_x0, maskable_mask)
                                x_gen[mask_to_x0] = x0[mask_to_x0]
                                maskable_mask.masked_fill_(mask_to_x0, False)
                        else:
                            # Final step: unmask everything (trainer.py line 212)
                            x_gen = x0

                    # Evaluate generated samples
                    gen_preds = x_gen

                    if dataset_type == 'sudoku':
                        evaluate_and_display_sudoku(
                            gen_preds,
                            n_samples,
                            mode_str="DVA Generation",
                            writer=writer,
                            step=step,
                            prefix="dva_generation",
                            max_display=5
                        )
                    elif dataset_type == 'text8':
                        evaluate_and_display_text8(
                            gen_preds,
                            n_samples,
                            mode_str="DVA Generation",
                            writer=writer,
                            step=step,
                            prefix="dva_generation",
                            max_display=10
                        )
                    else:
                        evaluate_and_display_sequential(
                            gen_preds,
                            n_samples,
                            mode_str="DVA Generation",
                            writer=writer,
                            step=step,
                            prefix="dva_generation",
                            max_display=50
                        )

                    model.train()

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
                            rand_test_ids = torch.randperm(test_quiz.size(0))[:n_samples]
                            test_quiz = test_quiz[rand_test_ids].to(device)
                            # test_quiz = test_quiz[:n_samples].to(device)
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
                    elif dataset_type == 'text8':
                        evaluate_and_display_text8(
                            final_preds_gen,
                            n_samples,
                            mode_str="Generation",
                            writer=writer,
                            step=step,
                            prefix="generation",
                            max_display=10
                        )

                    # ===== 2. COMPLETION FROM PARTIAL QUIZ =====
                    if test_quiz is not None and dataset_type == 'sudoku':
                        print(f"\n{'='*60}")
                        print("2. COMPLETION FROM PARTIAL QUIZ")
                        print(f"{'='*60}")

                        
                        # Start from partially masked
                        if sudoku_input_type == 'quiz_solution':
                            # test_quiz contains full sequence [batch, 164]
                            # Initialize xt_comp with quiz known, solution masked
                            x_temp = torch.full((n_samples, seq_len), model.mask_index, dtype=torch.long, device=device)
                            x_temp[:, :81] = test_quiz[:, :81]  # Quiz portion
                            x_temp[:, 81] = SEP_TOKEN_ID  # SEP token
                            x_temp[:, 163] = EOF_TOKEN_ID  # EOF token
                            test_quiz = x_temp
                            
                            xt_comp = test_quiz.clone()
                            xt_comp[:, 82:163] = model.mask_index  # Mask solution portion only
                            num_known = (xt_comp != model.mask_index).sum().item()
                            print(f"Starting from quiz_solution mode with {num_known}/{n_samples * seq_len} known values")

                            # For continuous latent: keep quiz+SEP+EOF clean, noise for solution only
                            full_embed = torch.randn((test_quiz.shape[0], seq_len, embed_dim), dtype=torch.float32, device=device)
                            full_embed[:, :81, :] = model.embed(test_quiz[:, :81])      # Quiz (clean)
                            full_embed[:, 81:82, :] = model.embed(test_quiz[:, 81:82])  # SEP (clean)
                            full_embed[:, 163:164, :] = model.embed(test_quiz[:, 163:164])  # EOF (clean)
                            # Positions 82-162 remain random noise (solution)

                            quiz_sep_embed_comp = full_embed[:, :82, :]
                            solution_noise_comp = full_embed[:, 82:163, :]
                            eof_embed_comp = full_embed[:, 163:164, :]

                            if combine_method == 'add':
                                # ADD mode: quiz+EOF are zero, solution is noise
                                xt_embed_comp = torch.cat([
                                    torch.zeros_like(quiz_sep_embed_comp),
                                    solution_noise_comp * sqrt_one_minus_alpha_start,
                                    torch.zeros_like(eof_embed_comp)
                                ], dim=1)
                            else:  # concat
                                # CONCAT mode: quiz+EOF are clean, solution is noise
                                xt_embed_comp = torch.cat([
                                    quiz_sep_embed_comp,
                                    solution_noise_comp * sqrt_one_minus_alpha_start,
                                    eof_embed_comp
                                ], dim=1)
                        else:
                            # solution_only mode: original behavior
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

                            if sudoku_input_type == 'quiz_solution':
                                # Only consider solution positions for unmasking
                                # Mask out quiz+SEP+EOF positions from confidence
                                non_solution_mask = torch.ones_like(confidence, dtype=torch.bool)
                                non_solution_mask[:, 82:163] = False  # Allow solution positions
                                confidence_filtered = confidence.masked_fill(non_solution_mask, float('-inf'))

                                # Select top-k from solution positions only
                                transfer_index = torch.zeros_like(x0_pred, dtype=torch.bool, device=device)
                                for j in range(confidence_filtered.size(0)):
                                    if num_transfer_tokens_comp[j, i] > 0:
                                        _, select_index = torch.topk(confidence_filtered[j], k=num_transfer_tokens_comp[j, i])
                                        transfer_index[j, select_index] = True

                                xt_comp[transfer_index] = x0_pred[transfer_index]
                            else:
                                # Original behavior
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

                                    if sudoku_input_type == 'quiz_solution':
                                        # Only denoise solution portion (positions 82-162)
                                        x_clean_pred_solution = x_clean_pred[:, 82:163, :]
                                        mask_still_solution = mask_still[:, 82:163, :]
                                        z_t_curr_solution = xt_embed_comp[:, 82:163, :]

                                        if mask_still_solution.any():
                                            # Denoise only solution embeddings
                                            eps_pred_solution = (z_t_curr_solution - sqrt_alpha_curr * x_clean_pred_solution) / (sqrt_one_minus_alpha_curr + 1e-8)
                                            xt_embed_denoised_solution = sqrt_alpha_next * x_clean_pred_solution + sqrt_one_minus_alpha_next * eps_pred_solution

                                            # Update only solution portion
                                            xt_embed_comp[:, 82:163, :] = torch.where(
                                                mask_still_solution,
                                                xt_embed_denoised_solution,
                                                xt_embed_comp[:, 82:163, :]
                                            )
                                    else:
                                        # Original behavior
                                        # Denoise using z_t (not z_combined)
                                        z_t_curr = torch.where(mask_still, xt_embed_comp, torch.zeros_like(xt_embed_comp))
                                        eps_pred = (z_t_curr - sqrt_alpha_curr * x_clean_pred) / (sqrt_one_minus_alpha_curr + 1e-8)
                                        xt_embed_denoised = sqrt_alpha_next * x_clean_pred + sqrt_one_minus_alpha_next * eps_pred
                                        # Update: masked positions get denoised, unmasked get zero, known get zero (will be handled by discrete embedding)
                                        xt_embed_comp = torch.where(mask_still, xt_embed_denoised, torch.zeros_like(xt_embed_comp))
                                else:
                                    # All positions unmasked, set z_t to zero
                                    if sudoku_input_type != 'quiz_solution':
                                        xt_embed_comp = torch.zeros_like(xt_embed_comp)

                        # Final completion predictions
                        if sudoku_input_type == 'quiz_solution':
                            # Ensure quiz+SEP+EOF remain unchanged
                            final_preds_comp = torch.cat([
                                test_quiz[:, :82],           # quiz + SEP (given)
                                xt_comp[:, 82:163],          # solution (predicted)
                                test_quiz[:, 163:164]        # EOF (given)
                            ], dim=1)
                        else:
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
                ccdd_temperature = args.get('ccdd_temperature', 0.0)  # Gumbel noise temperature for sampling

                with torch.no_grad():
                    # Load test quiz data for completion evaluation
                    test_quiz = None
                    if dataset_type == 'sudoku':
                        test_quiz_path = args.get('test_quiz_path', 'data_vmd/sudoku_test.csv')
                        if os.path.exists(test_quiz_path):
                            test_quiz, _ = load_sudoku_dataset(test_quiz_path)
                            rand_test_ids = torch.randperm(test_quiz.size(0))[:n_samples]
                            test_quiz = test_quiz[rand_test_ids].to(device)
                            # test_quiz = test_quiz[:n_samples].to(device)
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
                        # Add Gumbel noise for stochastic sampling
                        logits_with_noise = add_gumbel_noise(logits_pred, temperature=ccdd_temperature)
                        x0_pred = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

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
                    elif dataset_type == 'text8':
                        evaluate_and_display_text8(
                            final_preds_gen,
                            n_samples,
                            mode_str="Generation",
                            writer=writer,
                            step=step,
                            prefix="generation",
                            max_display=10
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
                            # Add Gumbel noise for stochastic sampling
                            logits_with_noise = add_gumbel_noise(logits_pred, temperature=ccdd_temperature)
                            x0_pred = torch.argmax(logits_with_noise, dim=-1)

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

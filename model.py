"""
Model components for diffusion-language experiments.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

try:  # pragma: no cover - optional dependency
    from transformers import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block as TransformersGPT2Block
except ImportError:  # pragma: no cover - optional dependency
    GPT2Config = None
    TransformersGPT2Block = None

__all__ = [
    "llada_mask",
    "add_gumbel_noise",
    "get_num_transfer_tokens",
    "MaskedPredictor",
    "EmbeddingMatrix",
    "OneHotEmbedding",
    "UnitSphereEmbedding",
    "SimpleTransformerBlock",
    "GPT2Block",
    "SimpleDiffusionModel",
]


def llada_mask(x0: torch.Tensor, t: torch.Tensor, mask_index: int):
    """
    LLaDA-style masking: mask tokens with probability t.

    Args:
        x0: Original tokens [B, L]
        t: Masking probability per batch element [B]
        mask_index: Index to use for masked positions

    Returns:
        xt: Masked tokens [B, L]
    """
    B, L = x0.shape
    mask = torch.rand(B, L, device=x0.device) < t.unsqueeze(1)
    xt = x0.clone()
    xt[mask] = mask_index
    return xt


def add_gumbel_noise(logits: torch.Tensor, temperature: float = 1.0):
    """
    Add Gumbel noise to logits for sampling.

    Args:
        logits: Logits [B, L, vocab_size]
        temperature: Temperature for sampling

    Returns:
        Logits with Gumbel noise added
    """
    if temperature == 0:
        return logits
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    return logits + temperature * gumbel_noise


def get_num_transfer_tokens(mask: torch.Tensor, steps: int):
    """
    Compute number of tokens to unmask at each step for each batch element.

    Args:
        mask: Boolean mask [B, L] indicating masked positions
        steps: Number of denoising steps

    Returns:
        num_transfer: Number of tokens to unmask per step [B, steps]
    """
    B, L = mask.shape
    num_masked = mask.sum(dim=1)  # [B]

    # Compute how many tokens to transfer at each step
    num_transfer = torch.zeros(B, steps, dtype=torch.long, device=mask.device)
    for i in range(B):
        total = num_masked[i].item()
        if total == 0:
            continue
        # Linear schedule: unmask tokens evenly across steps
        tokens_per_step = total / steps
        for j in range(steps):
            start_idx = int(j * tokens_per_step)
            end_idx = int((j + 1) * tokens_per_step)
            num_transfer[i, j] = end_idx - start_idx

    return num_transfer


class MaskedPredictor(nn.Module):
    """
    Masked Diffusion Model for discrete sequences.
    Uses the SAME architecture as SimpleDiffusionModel but with its own embedding.
    """

    def __init__(
        self,
        vocab_size,
        seq_len,
        embed_dim=64,
        hidden_dim=None,
        n_heads=2,
        n_layers=4,
        positional_encoding="learned",
        dataset_type="sequential",
        transformer_block_type="simple",
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_index = vocab_size  # Use vocab_size as mask token
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.positional_encoding = positional_encoding.lower()
        self.dataset_type = dataset_type.lower()
        self.transformer_block_type = transformer_block_type.lower()

        # Validate transformer block type
        if self.transformer_block_type not in ["simple", "gpt2"]:
            raise ValueError(
                f"transformer_block_type must be 'simple' or 'gpt2', got '{transformer_block_type}'"
            )

        # Token embedding (+1 for mask token)
        self.embed = nn.Embedding(vocab_size + 1, embed_dim)

        # Project embedding to hidden dimension (same as SimpleDiffusionModel)
        self.input_proj = nn.Linear(embed_dim, self.hidden_dim, bias=False)

        # Positional embeddings (same options as SimpleDiffusionModel)
        if self.dataset_type == "sudoku" and self.positional_encoding == "sinusoidal":
            # Use 2D positional encoding for sudoku (9x9 grid = 81 positions)
            pe = SimpleDiffusionModel._build_2d_sinusoidal_embedding(9, 9, self.hidden_dim)
            self.register_buffer("pos_embedding", pe, persistent=False)
        elif self.positional_encoding == "learned":
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, self.hidden_dim))
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        elif self.positional_encoding == "sinusoidal":
            pe = SimpleDiffusionModel._build_sinusoidal_embedding(seq_len, self.hidden_dim)
            self.register_buffer("pos_embedding", pe, persistent=False)
        else:
            raise ValueError(f"Unknown positional_encoding '{positional_encoding}'")

        # Time embedding (for masking probability t)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Transformer blocks - SAME as SimpleDiffusionModel
        if self.transformer_block_type == "simple":
            self.blocks = nn.ModuleList(
                [SimpleTransformerBlock(self.hidden_dim, n_heads) for _ in range(n_layers)]
            )
        elif self.transformer_block_type == "gpt2":
            self.blocks = nn.ModuleList(
                [
                    GPT2Block(
                        hidden_size=self.hidden_dim,
                        num_attention_heads=n_heads,
                        intermediate_size=4 * self.hidden_dim,
                        layer_norm_epsilon=1e-5,
                        attn_pdrop=0.0,
                        resid_pdrop=0.1,
                    )
                    for _ in range(n_layers)
                ]
            )

        # Output projection - SAME as SimpleDiffusionModel
        self.norm_out = nn.LayerNorm(self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, vocab_size, bias=True)

    def _get_sinusoidal_time_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Build sinusoidal time embedding (same as SimpleDiffusionModel).
        t: [B] continuous values in [0, 1]
        Returns: [B, dim]
        """
        half_dim = dim // 2
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb_scale)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant", value=0)
        return emb

    def forward(self, xt, x0, t):
        """
        Forward pass for training - using SAME architecture as SimpleDiffusionModel.

        Args:
            xt: Masked tokens [B, L]
            x0: Original tokens [B, L]
            t: Masking probability [B]

        Returns:
            loss: Cross-entropy loss on masked positions
        """
        # Token embedding
        x = self.embed(xt)  # [B, L, embed_dim]

        # Project to hidden dimension
        x = self.input_proj(x)  # [B, L, hidden_dim]

        # Add positional encoding
        x = x + self.pos_embedding  # [B, L, hidden_dim]

        # Add time embedding (masking probability)
        t_emb = self._get_sinusoidal_time_embedding(t, self.hidden_dim)  # [B, hidden_dim]
        t_emb = self.time_mlp(t_emb)  # [B, hidden_dim]
        x = x + t_emb.unsqueeze(1)  # [B, L, hidden_dim]

        # Pass through transformer blocks (same as SimpleDiffusionModel)
        for block in self.blocks:
            x = block(x)

        # Output projection
        x = self.norm_out(x)
        logits = self.output_layer(x)  # [B, L, vocab_size]

        # Compute loss only on masked positions
        mask = xt == self.mask_index
        logits_masked = logits[mask]
        targets_masked = x0[mask]

        # Weight by inverse of masking probability
        t_weight = (1.0 / (t + 1e-8)).view(-1, 1).expand_as(xt)[mask]
        loss = (F.cross_entropy(logits_masked, targets_masked, reduction="none") * t_weight).mean()

        return loss

    def _forward_without_loss(self, xt, t=None):
        """
        Forward pass without computing loss (for generation).

        Args:
            xt: Token sequence [B, L]
            t: Optional time value [B] or scalar (defaults to 0.5)

        Returns:
            logits: [B, L, vocab_size]
        """
        B = xt.shape[0]
        if t is None:
            t = torch.full((B,), 0.5, device=xt.device)
        elif isinstance(t, float):
            t = torch.full((B,), t, device=xt.device)

        # Token embedding
        x = self.embed(xt)  # [B, L, embed_dim]

        # Project to hidden dimension
        x = self.input_proj(x)  # [B, L, hidden_dim]

        # Add positional encoding
        x = x + self.pos_embedding  # [B, L, hidden_dim]

        # Add time embedding
        t_emb = self._get_sinusoidal_time_embedding(t, self.hidden_dim)  # [B, hidden_dim]
        t_emb = self.time_mlp(t_emb)  # [B, hidden_dim]
        x = x + t_emb.unsqueeze(1)  # [B, L, hidden_dim]

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        x = self.norm_out(x)
        logits = self.output_layer(x)  # [B, L, vocab_size]

        return logits

    @torch.no_grad()
    def generate(self, xt, steps=1, temperature=1.0, remasking="low_confidence"):
        """
        Generate tokens by iteratively unmasking.

        Args:
            xt: Initial masked tokens [B, L]
            steps: Number of denoising steps
            temperature: Sampling temperature
            remasking: Strategy for selecting which tokens to unmask

        Returns:
            xt: Generated tokens [B, L]
        """
        device = xt.device

        # Compute transfer schedule based on initial mask
        initial_mask = xt == self.mask_index
        num_transfer_tokens = get_num_transfer_tokens(initial_mask, steps)

        for i in range(steps):
            # Get model predictions using new architecture
            # Time value decreases from 0.9 to 0.1 as we unmask
            t_val = 0.9 - (i / max(steps - 1, 1)) * 0.8
            logits = self._forward_without_loss(xt, t=t_val)

            # Sample with Gumbel noise
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Compute confidence
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            # Get current mask
            mask_index = xt == self.mask_index

            # Build confidence scores for masked positions
            if remasking == "low_confidence":
                neg_inf = torch.tensor(float("-inf"), device=device)
                confidence = torch.where(mask_index, x0_p, neg_inf)
            elif remasking == "random":
                rand = torch.rand_like(x0_p)
                neg_inf = torch.tensor(float("-inf"), device=device)
                confidence = torch.where(mask_index, rand, neg_inf)
            else:
                raise NotImplementedError(remasking)

            # Keep unmasked tokens unchanged
            x0 = torch.where(mask_index, x0, xt)

            # Select tokens to unmask based on confidence
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for j in range(confidence.size(0)):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            xt[transfer_index] = x0[transfer_index]

        return xt

    @torch.no_grad()
    def generate_iteratively(self, xt_init):
        """
        Greedy generation: unmask one token at a time.

        Args:
            xt_init: Initial masked tokens [B, L]

        Returns:
            xt: Generated tokens [B, L]
        """
        xt = xt_init.clone()
        B, L = xt.shape

        for step in range(L):
            mask = xt == self.mask_index
            if not mask.any():
                break

            # Get predictions using new architecture
            t_val = 0.9 - (step / max(L - 1, 1)) * 0.8
            logits = self._forward_without_loss(xt, t=t_val)
            preds = logits.argmax(dim=-1)

            # Sample one masked position per batch element
            probs = mask.float()
            probs /= probs.sum(dim=1, keepdim=True) + 1e-8
            choice = torch.multinomial(probs, num_samples=1).squeeze(1)
            xt[torch.arange(B), choice] = preds[torch.arange(B), choice]

        return xt


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
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # [B, H, T, D]
        out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, T, C)  # [B, T, C]
        x = residual + self.attn_out(out)

        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class GPT2Block(nn.Module):
    """Wrapper for transformers GPT2Block with simplified interface for diffusion."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size=None,
        layer_norm_epsilon=1e-5,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
    ):
        super().__init__()

        if TransformersGPT2Block is None or GPT2Config is None:
            raise ImportError(
                "transformers is required for GPT2Block. Install it or use transformer_block_type='simple'."
            )

        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        # Create GPT2Config for the block
        config = GPT2Config(
            n_embd=hidden_size,
            n_head=num_attention_heads,
            n_inner=intermediate_size,
            layer_norm_epsilon=layer_norm_epsilon,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            embd_pdrop=0.0,  # Not used in block
            activation_function="gelu_new",
            scale_attn_weights=True,
            scale_attn_by_inverse_layer_idx=False,
            reorder_and_upcast_attn=False,
        )

        # Set attention implementation to 'eager' (standard PyTorch attention)
        config._attn_implementation = "eager"

        # Use the actual GPT2Block from transformers
        self.block = TransformersGPT2Block(config, layer_idx=0)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            [batch, seq_len, hidden_size]
        """
        # Call transformers GPT2Block
        batch_size, seq_len, _ = x.shape
        attention_mask = torch.ones(
            batch_size,
            1,
            1,
            seq_len,
            dtype=torch.float32,
            device=x.device,
        )
        outputs = self.block(x, attention_mask=attention_mask)

        # GPT2Block returns a tuple where first element is hidden_states
        return outputs[0]


class SimpleDiffusionModel(nn.Module):
    """
    Simplified diffusion model for discrete sequences.
    Takes noisy embeddings and predicts clean embeddings.
    """

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        n_blocks,
        n_heads,
        vocab_size,
        seq_len,
        positional_encoding: str = "learned",
        dataset_type: str = "simple",
        transformer_block_type: str = "simple",
        enable_repae: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.positional_encoding = positional_encoding.lower()
        self.dataset_type = dataset_type.lower()
        self.transformer_block_type = transformer_block_type.lower()
        self.enable_repae = enable_repae

        # Storage for intermediate layer activations (REPAE)
        self.layer_activations = []

        # Validate transformer block type
        if self.transformer_block_type not in ["simple", "gpt2"]:
            raise ValueError(
                f"transformer_block_type must be 'simple' or 'gpt2', got '{transformer_block_type}'"
            )

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
            raise ValueError(
                f"Unknown positional_encoding '{positional_encoding}'. Use 'learned' or 'sinusoidal'."
            )

        # Time/noise level embedding (using sinusoidal encoding)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer blocks - choose between SimpleTransformerBlock or GPT2Block
        if self.transformer_block_type == "simple":
            # Original simple transformer implementation (no bias, no dropout)
            self.blocks = nn.ModuleList(
                [SimpleTransformerBlock(hidden_dim, n_heads) for _ in range(n_blocks)]
            )
        elif self.transformer_block_type == "gpt2":
            # GPT2-style transformer blocks from transformers library
            # NOTE: Dropout set to 0.0 for diffusion models (better for small datasets and determinism)
            self.blocks = nn.ModuleList(
                [
                    GPT2Block(
                        hidden_size=hidden_dim,
                        num_attention_heads=n_heads,
                        intermediate_size=4 * hidden_dim,  # Standard GPT2 MLP expansion
                        layer_norm_epsilon=1e-5,
                        attn_pdrop=0.0,
                        resid_pdrop=0.1,
                    )
                    for _ in range(n_blocks)
                ]
            )

        # Output projection
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=True)

    @staticmethod
    def _build_sinusoidal_embedding(seq_len: int, dim: int) -> torch.Tensor:
        """Build 1D sinusoidal positional embedding."""
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / max(dim, 1))
        )
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
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(9.0) / d_model))

        row_sinusoid = row_pos * div_term  # [height, d_model//2]
        pe[:, :, 0:d_model:2] = torch.sin(row_sinusoid).unsqueeze(1).repeat(1, width, 1)
        pe[:, :, 1:d_model:2] = torch.cos(row_sinusoid).unsqueeze(1).repeat(1, width, 1)

        # Generate column encodings (second half of dimensions)
        col_pos = torch.arange(width, dtype=torch.float32).unsqueeze(1)  # [width, 1]
        col_sinusoid = col_pos * div_term  # [width, d_model//2]
        pe[:, :, d_model::2] = torch.sin(col_sinusoid).unsqueeze(0).repeat(height, 1, 1)
        pe[:, :, d_model + 1 :: 2] = torch.cos(col_sinusoid).unsqueeze(0).repeat(height, 1, 1)

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
        # Clear previous activations if REPAE is enabled
        if self.enable_repae:
            self.layer_activations = []

        x = self.input_proj(z)  # [B, T, hidden_dim]

        # Positional information
        pos_emb = self.pos_embedding[:, : x.size(1), :]
        x = x + pos_emb

        # Time embedding
        time_emb = self.get_time_embedding(gamma, self.hidden_dim)  # [B, hidden_dim]
        time_emb = self.time_mlp(time_emb)  # [B, hidden_dim]
        x = x + time_emb[:, None, :]

        # Store initial embedding if REPAE is enabled
        if self.enable_repae:
            self.layer_activations.append(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            # Store activation after each block if REPAE is enabled
            if self.enable_repae:
                self.layer_activations.append(x)

        # Output projection
        x = self.norm_out(x)
        logits = self.output_proj(x)  # [B, T, vocab_size]
        return logits

    def get_layer_activations(self):
        """
        Get the stored layer activations (REPAE).

        Returns:
            List of tensors, where each tensor is [batch, seq_len, hidden_dim]
            Index 0: after input projection + positional + time embedding
            Index 1 to n_blocks: after each transformer block
        """
        if not self.enable_repae:
            raise RuntimeError("REPAE is not enabled. Set enable_repae=True when creating the model.")
        return self.layer_activations

    def clear_layer_activations(self):
        """Clear the stored layer activations to free memory."""
        self.layer_activations = []

    def get_num_layers(self):
        """Return the number of transformer blocks."""
        return len(self.blocks)

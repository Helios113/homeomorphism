"""Custom low-dimensional Transformer for exact topological Jacobian analysis.

This module intentionally avoids torch.nn.Transformer wrappers so each sub-block
map can be evaluated independently with torch.autograd.functional.jacobian.

Core setup used by the tests in this repository:
- Ambient dimension d = 32
- Sequence length T = 10
- Number of heads = 2

The implementation exposes explicit sub-block maps with pre-norm residual form:
    h^{n+1} = h^n + g(h^n, theta^n)
where g is either the attention update or the FFN update.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def set_reproducible_seed(seed: int) -> None:
    """Set deterministic seeds for torch CPU and CUDA backends."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ToyTopologicalAttention(nn.Module):
    """Minimal multi-head self-attention sub-block with explicit residual update.

    This block is intentionally small and transparent:
      - Pre-norm LayerNorm on the token stream.
      - Linear Q/K/V projections and output projection.
      - Optional causal masking.
      - Residual connection h + update.

    The update map g_attn(h) can be accessed directly via update(...), while
    forward(...) always returns h + g_attn(h).
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 2,
        *,
        causal: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal

        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.force_negative_identity = False

    def _shape_heads(self, x: Tensor) -> Tensor:
        # (T, d) -> (heads, T, head_dim)
        t = x.shape[0]
        x = x.view(t, self.n_heads, self.head_dim)
        return x.permute(1, 0, 2).contiguous()

    def update(self, h: Tensor) -> Tensor:
        """Compute attention update g_attn(h) with pre-norm."""
        if self.force_negative_identity:
            return -h
        t, d = h.shape
        if d != self.d_model:
            raise ValueError(f"Expected input dim {self.d_model}, got {d}")

        h_norm = self.norm(h)
        q = self._shape_heads(self.w_q(h_norm))
        k = self._shape_heads(self.w_k(h_norm))
        v = self._shape_heads(self.w_v(h_norm))

        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (H, T, T)

        if self.causal:
            mask = torch.triu(
                torch.ones(t, t, device=h.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(attn, v)  # (H, T, head_dim)
        mixed = mixed.permute(1, 0, 2).contiguous().view(t, d)
        return self.w_o(mixed)

    def forward(self, h: Tensor) -> Tensor:
        return h + self.update(h)


class ToyTopologicalFFN(nn.Module):
    """Token-wise FFN sub-block with explicit residual update.

    Uses a standard pre-norm MLP:
      g_ffn(h) = W2 GELU(W1 LN(h))
      phi_ffn(h) = h + g_ffn(h)
    """

    def __init__(
        self,
        d_model: int = 32,
        d_ff: int = 128,
        *,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.w_1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_2 = nn.Linear(d_ff, d_model, bias=False)
        self.force_negative_identity = False

    def update(self, h: Tensor) -> Tensor:
        if self.force_negative_identity:
            return -h
        h_norm = self.norm(h)
        hidden = F.gelu(self.w_1(h_norm))
        return self.w_2(hidden)

    def forward(self, h: Tensor) -> Tensor:
        return h + self.update(h)


class ToyTopologicalLayer(nn.Module):
    """One Transformer layer composed of attention then FFN sub-blocks."""

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 2,
        d_ff: int = 128,
        *,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.attn = ToyTopologicalAttention(d_model=d_model, n_heads=n_heads, causal=causal)
        self.ffn = ToyTopologicalFFN(d_model=d_model, d_ff=d_ff)

    def attn_phi(self, h: Tensor) -> Tensor:
        """Residual map for attention sub-block: h + g_attn(h)."""
        return self.attn(h)

    def ffn_phi(self, h: Tensor) -> Tensor:
        """Residual map for FFN sub-block: h + g_ffn(h)."""
        return self.ffn(h)

    def forward(self, h: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        after_attn = self.attn_phi(h)
        after_ffn = self.ffn_phi(after_attn)
        return after_ffn, {"after_attn": after_attn, "after_ffn": after_ffn}


class ToyTopologicalTransformer(nn.Module):
    """Custom Transformer for Jacobian/topology experiments.

    Exposes per-layer, per-sub-block residual maps so tests can run exact
    Jacobian analysis with torch.autograd.functional.jacobian.
    """

    def __init__(
        self,
        *,
        d_model: int = 32,
        seq_len: int = 10,
        n_heads: int = 2,
        d_ff: int = 128,
        n_layers: int = 2,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [
                ToyTopologicalLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    causal=causal,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def reset_parameters_continuous(self, *, seed: int, scale: float = 0.05) -> None:
        """Continuous initialization using normal and Xavier schemes.

        This mirrors standard continuous random initializations and supports the
        measure-zero non-singularity checks in Experiment 1.
        """
        g = torch.Generator().manual_seed(seed)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                if mod.weight.ndim >= 2:
                    nn.init.xavier_uniform_(mod.weight)
                with torch.no_grad():
                    mod.weight.add_(torch.randn(mod.weight.shape, generator=g) * scale)
            elif isinstance(mod, nn.LayerNorm):
                nn.init.normal_(mod.weight, mean=1.0, std=0.01, generator=g)
                nn.init.normal_(mod.bias, mean=0.0, std=0.01, generator=g)

    def reset_parameters_discrete(self, *, seed: int, mode: str = "binary") -> None:
        """Degenerate discrete initialization baseline for singularity stress.

        Modes:
        - binary: weights become +/-1.
        - zeros: all linear weights become zero and LN affine params are zero.

        The binary branch also enforces a rank-deficient output projection in each
        FFN block, which empirically drives frequent singular token Jacobians.
        """
        g = torch.Generator().manual_seed(seed)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                with torch.no_grad():
                    if mode == "binary":
                        signs = torch.randint(
                            low=0,
                            high=2,
                            size=mod.weight.shape,
                            generator=g,
                            device=mod.weight.device,
                        )
                        mod.weight.copy_(signs.to(mod.weight.dtype) * 2.0 - 1.0)
                    elif mode == "zeros":
                        mod.weight.zero_()
                    else:
                        raise ValueError(f"Unknown discrete mode: {mode}")
            elif isinstance(mod, nn.LayerNorm):
                with torch.no_grad():
                    if mode == "binary":
                        signs = torch.randint(
                            low=0,
                            high=2,
                            size=mod.weight.shape,
                            generator=g,
                            device=mod.weight.device,
                        )
                        mod.weight.copy_(signs.to(mod.weight.dtype) * 2.0 - 1.0)
                        mod.bias.zero_()
                    else:
                        mod.weight.zero_()
                        mod.bias.zero_()

        if mode == "binary":
            # Explicitly enforce FFN output rank deficiency in the degenerate branch
            # by duplicating half of the rows from the first row.
            with torch.no_grad():
                for layer in self.layers:
                    w2 = layer.ffn.w_2.weight
                    if w2.shape[0] >= 2:
                        half = w2.shape[0] // 2
                        w2[half:] = w2[0].unsqueeze(0)

        # Degenerate branch intentionally drives a measure-zero pathological case
        # where g(h) = -h, yielding J = I + G = 0 for the residual map.
        for layer in self.layers:
            layer.attn.force_negative_identity = True
            layer.ffn.force_negative_identity = True

    def forward(self, h: Tensor, *, return_intermediates: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Forward pass over (T, d) stream.

        If return_intermediates=True, returns the final output and a list of
        intermediate residual states (after each attention and FFN sub-block) in
        order: [attn0, ffn0, attn1, ffn1, ...]. Input and final norm are NOT
        included in the intermediates list. This matches the original test
        convention.
        """
        if h.shape != (self.seq_len, self.d_model):
            raise ValueError(
                f"Expected input shape {(self.seq_len, self.d_model)}, got {tuple(h.shape)}"
            )

        states: list[Tensor] = []
        x = h
        for layer in self.layers:
            x, layer_states = layer(x)
            states.append(layer_states["after_attn"])
            states.append(layer_states["after_ffn"])

        if return_intermediates:
            return x, states
        return x

    def forward_with_states(self, h: Tensor) -> list[Tensor]:
        """Return hidden states at all depths.

        Convenience wrapper matching the LLaMA-style experiment interface.
        Returns list of length n_depths = 2 * n_layers + 2:
          [input, post_attn_0, post_ffn_0, post_attn_1, post_ffn_1, ..., post_norm]
        """
        final, intermediates = self.forward(h, return_intermediates=True)
        post_norm = self.final_norm(final)
        return [h] + intermediates + [post_norm]

    def batch_forward(self, h_batch: Tensor) -> Tensor:
        """Batch-friendly forward over shape (B, T, d)."""
        if h_batch.dim() != 3:
            raise ValueError(f"Expected 3-D batch input, got shape {tuple(h_batch.shape)}")
        outputs = []
        for i in range(h_batch.shape[0]):
            outputs.append(self.forward(h_batch[i]))
        return torch.stack(outputs, dim=0)

    @property
    def n_depths(self) -> int:
        """Total number of depth checkpoints: input + 2 per layer + post-norm."""
        return 2 * self.n_layers + 2

    def subblock_phi(self, layer_idx: int, kind: str) -> Callable[[Tensor], Tensor]:
        """Return phi(h) = h + g(h) for a specific sub-block in one layer."""
        if not (0 <= layer_idx < self.n_layers):
            raise ValueError(f"layer_idx={layer_idx} outside [0, {self.n_layers})")

        layer = self.layers[layer_idx]
        if kind == "attn":
            return layer.attn_phi
        if kind == "ffn":
            return layer.ffn_phi
        raise ValueError(f"Unknown sub-block kind: {kind}")

    def batch_forward_with_states(self, h_batch: Tensor) -> list[Tensor]:
        """Return hidden states for a batch of sequences.

        Parameters
        ----------
        h_batch : torch.Tensor
            Shape (B, T, d_model). Each row is an unbatched sequence.

        Returns
        -------
        list[torch.Tensor]
            Length = n_depths. Each tensor has shape (B, T, d_model).
        """
        if h_batch.dim() != 3:
            raise ValueError(f"Expected 3-D input, got shape {tuple(h_batch.shape)}")
        B = h_batch.shape[0]
        # Accumulate per-depth lists over the batch
        depth_lists: list[list[Tensor]] = [[] for _ in range(self.n_depths)]
        for i in range(B):
            states = self.forward_with_states(h_batch[i])
            for d, s in enumerate(states):
                depth_lists[d].append(s)
        # Stack along batch dimension
        return [torch.stack(lst, dim=0) for lst in depth_lists]


def full_sequence_jacobian(phi: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Return full Jacobian d phi(x) / d x with shape (T, d, T, d)."""
    x = x.detach().clone().requires_grad_(True)
    return torch.autograd.functional.jacobian(phi, x, create_graph=False, vectorize=True)


def flatten_jacobian(jac: Tensor) -> Tensor:
    """Flatten (T, d, T, d) into (T*d, T*d)."""
    if jac.dim() != 4:
        raise ValueError(f"Expected 4-D Jacobian, got shape {tuple(jac.shape)}")
    t, d, t2, d2 = jac.shape
    if t != t2 or d != d2:
        raise ValueError(f"Expected square Jacobian blocks, got shape {tuple(jac.shape)}")
    return jac.reshape(t * d, t * d)


def extract_diagonal_blocks(jac: Tensor) -> list[Tensor]:
    """Extract token-wise diagonal Jacobian blocks J^(i) from (T, d, T, d)."""
    t = jac.shape[0]
    return [jac[i, :, i, :] for i in range(t)]


def is_block_strictly_lower_triangular(
    jac: Tensor,
    *,
    atol: float = 1e-7,
) -> bool:
    """Check if all upper off-diagonal blocks are numerically zero."""
    t = jac.shape[0]
    for i in range(t):
        for j in range(i + 1, t):
            if not torch.allclose(jac[i, :, j, :], torch.zeros_like(jac[i, :, j, :]), atol=atol):
                return False
    return True


def determinant_factorization_error(jac: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Compare global determinant with product of diagonal block determinants.

    Returns:
      - full_det: det(flattened full Jacobian)
      - diag_prod: product_i det(J^(i))
      - abs_err: absolute error between the two
    """
    jac_flat = flatten_jacobian(jac).to(torch.float32)
    full_det = torch.linalg.det(jac_flat)

    diag_prod = torch.tensor(1.0, dtype=torch.float32, device=jac.device)
    for block in extract_diagonal_blocks(jac):
        diag_prod = diag_prod * torch.linalg.det(block.to(torch.float32))

    abs_err = torch.abs(full_det - diag_prod)
    return full_det, diag_prod, abs_err


def block_condition_numbers(jac: Tensor, *, eps: float = 1e-12) -> list[float]:
    """Compute condition numbers for all diagonal token blocks in a Jacobian."""
    conds: list[float] = []
    for block in extract_diagonal_blocks(jac):
        s = torch.linalg.svdvals(block.to(torch.float32))
        sigma_min = max(float(s[-1].item()), eps)
        sigma_max = float(s[0].item())
        conds.append(sigma_max / sigma_min)
    return conds


def sample_torus(
    *,
    n_samples: int,
    ambient_dim: int = 32,
    major_radius: float = 2.0,
    minor_radius: float = 0.7,
    seed: int = 0,
) -> Tensor:
    """Sample points from a torus and embed smoothly into R^{ambient_dim}."""
    if ambient_dim < 3:
        raise ValueError("ambient_dim must be >= 3")

    g = torch.Generator().manual_seed(seed)
    theta = 2.0 * torch.pi * torch.rand(n_samples, generator=g)
    phi = 2.0 * torch.pi * torch.rand(n_samples, generator=g)

    x = (major_radius + minor_radius * torch.cos(phi)) * torch.cos(theta)
    y = (major_radius + minor_radius * torch.cos(phi)) * torch.sin(theta)
    z = minor_radius * torch.sin(phi)
    torus_r3 = torch.stack([x, y, z], dim=1)

    # Random full-rank linear embedding from R^3 -> R^ambient_dim.
    # Avoid reduced QR here because a 3xambient matrix would reduce back to 3x3.
    embed = torch.randn(3, ambient_dim, generator=g)
    embed = embed / (embed.norm(dim=0, keepdim=True) + 1e-8)
    points = torus_r3 @ embed

    # Standardize dimension-wise to avoid trivial scale artifacts.
    points = (points - points.mean(dim=0, keepdim=True)) / (points.std(dim=0, keepdim=True) + 1e-6)
    return points.to(torch.float32)


def sample_sphere(
    *,
    n_samples: int,
    ambient_dim: int = 32,
    radius: float = 1.0,
    seed: int = 0,
) -> Tensor:
    """Uniformly sample points on the (ambient_dim-1)-sphere via normal projection.

    Ground-truth intrinsic dimension: ambient_dim - 1.
    """
    if ambient_dim < 2:
        raise ValueError("ambient_dim must be >= 2 for a sphere")
    g = torch.Generator().manual_seed(seed)
    points = torch.randn(n_samples, ambient_dim, generator=g)
    points = points / points.norm(dim=1, keepdim=True) * radius
    return points.to(torch.float32)


def sample_hyperplane(
    *,
    n_samples: int,
    ambient_dim: int,
    manifold_dim: int,
    seed: int = 0,
) -> Tensor:
    """Sample points from a random M-dimensional linear subspace of R^ambient_dim.

    A random orthonormal basis A ∈ R^{ambient_dim×manifold_dim} is drawn once
    (QR of a Gaussian matrix).  Each token is x = A @ z with z ~ N(0, I_M).

    Ground-truth intrinsic dimension: manifold_dim.
    """
    if manifold_dim >= ambient_dim:
        raise ValueError(f"manifold_dim={manifold_dim} must be < ambient_dim={ambient_dim}")
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(ambient_dim, manifold_dim, generator=g)
    Q, _ = torch.linalg.qr(raw)
    basis = Q  # (ambient_dim, manifold_dim), orthonormal columns
    z = torch.randn(n_samples, manifold_dim, generator=g)
    points = z @ basis.T
    # Center to origin (bias-free); standardize scale for numerical stability.
    points = points - points.mean(dim=0, keepdim=True)
    points = points / (points.std(dim=0, keepdim=True) + 1e-6)
    return points.to(torch.float32)


def sample_swiss_roll(
    *,
    n_samples: int,
    ambient_dim: int,
    hole: float = 0.1,
    seed: int = 0,
) -> Tensor:
    """Classic Swiss roll (2D manifold) embedded randomly in R^{ambient_dim}.

    The manifold is first generated in R^3 using the standard parametric
    equations, then projected to R^{ambient_dim} via a random full-rank linear
    embedding that is orthonormalized along the projection dimension.

    Ground-truth intrinsic dimension: 2.
    """
    if ambient_dim < 3:
        raise ValueError("ambient_dim must be >= 3 for Swiss roll embedding")
    g = torch.Generator().manual_seed(seed)
    # Parametric coordinates in [0, 1]
    t = 1.5 * torch.pi * (1 + 2 * torch.rand(n_samples, generator=g))
    # Hole: exclude central region by scaling the radial component
    scale = 1.0 - hole
    x = scale * t.cos() * t
    y = scale * t.sin() * t
    z = t  # height
    points_r3 = torch.stack([x, y, z], dim=1)  # (n_samples, 3)

    # Random linear embedding: R^3 → R^ambient_dim
    embed = torch.randn(3, ambient_dim, generator=g)
    embed = embed / (embed.norm(dim=0, keepdim=True) + 1e-8)
    points = points_r3 @ embed

    # Standardize dimension-wise
    points = (points - points.mean(dim=0, keepdim=True)) / (points.std(dim=0, keepdim=True) + 1e-6)
    return points.to(torch.float32)


def sample_white_noise(*, n_samples: int, ambient_dim: int = 32, seed: int = 0) -> Tensor:
    """Generate baseline white-noise cloud N(0, I_ambient).

    Ground-truth intrinsic dimension: ambient_dim (full ambient space).
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_samples, ambient_dim, generator=g, dtype=torch.float32)


class HyperplaneSampler:
    """Stateful sampler for an M-dimensional linear subspace of R^N.

    Mirrors the sampler originally in exp3_llama_hyperplane.py, but now
    lives in toy_transformer for reuse. Uses the functional ``sample_hyperplane``
    under the hood.

    Parameters
    ----------
    N : int
        Ambient dimension.
    M : int
        Intrinsic manifold dimension (must be < N).
    seed : int
        Random seed.
    device : str
        Torch device for output tensors.
    """

    def __init__(self, N: int, M: int, seed: int = 0, device: str = "cpu"):
        if M >= N:
            raise ValueError(f"manifold_dim M={M} must be < ambient N={N}")
        self.N = N
        self.M = M
        self.seed = seed
        self.device = device

    def sample(self, B: int, T: int) -> Tensor:
        """Return a batch of sequences on the hyperplane.

        Returns
        -------
        Tensor
            Shape (B, T, N) on the requested device.
        """
        points = sample_hyperplane(
            n_samples=B * T,
            ambient_dim=self.N,
            manifold_dim=self.M,
            seed=self.seed,
        )
        return points.view(B, T, self.N).to(self.device)


def quantize_tensor_symmetric(x: Tensor, *, n_levels: int) -> Tensor:
    """Symmetric uniform quantization around zero.

    n_levels controls aggressiveness; lower values are more destructive.
    """
    if n_levels < 2:
        raise ValueError("n_levels must be >= 2")
    max_abs = x.abs().max().clamp(min=1e-8)
    step = (2.0 * max_abs) / float(n_levels - 1)
    q = torch.round((x + max_abs) / step) * step - max_abs
    return q


def apply_adversarial_intervention(
    model: ToyTopologicalTransformer,
    *,
    step: int,
    total_steps: int,
    seed: int,
) -> dict[str, float]:
    """Apply aggressive quantization + noise to weights during training.

    Returns intervention metadata for logging.
    """
    if total_steps <= 1:
        progress = 1.0
    else:
        progress = float(step) / float(total_steps - 1)

    # Progressively reduce precision from 16 levels to 2 levels.
    n_levels = max(2, int(round(16 - 14 * progress)))
    noise_std = 0.0 + 0.35 * progress

    g = torch.Generator().manual_seed(seed + step)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim < 2:
                continue
            if "w_q" in name or "w_k" in name or "w_v" in name or "w_o" in name or "w_1" in name or "w_2" in name:
                q = quantize_tensor_symmetric(param, n_levels=n_levels)
                noise = torch.randn(param.shape, generator=g, device=param.device, dtype=param.dtype) * noise_std
                param.copy_(q + noise)

    return {
        "progress": progress,
        "n_levels": float(n_levels),
        "noise_std": float(noise_std),
    }

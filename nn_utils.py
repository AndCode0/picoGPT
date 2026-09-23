import torch
from torch import nn
import einx
import math
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Iterable


class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        """
        In Italian, 'affine' means 'similar to'. This implementation is
        'affine' to nn.Linear, except it's not affine: y = xA^T.

        Args:
            in_features (int): final dimension of the input
            out_features (int): final dimension of the output
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype): Datatype of the parameters
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2.0 / sum(self.weight.shape))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, in_feature: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return in_feature @ self.weight.mT

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
        """
        A lookup table mapping token_ids to a learnable embedding

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype): Datatype of the parameters
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.weight: Float[Tensor, "num_embedding embedding_dim"] = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, "... embedding_dim"]:
        """Gather rows. All the learning lives in the table; this is just fancy indexing."""
        return einx.get_at("[v] d_emb, b t -> b t d_emb", self.weight, token_ids)

class RMSNorm(nn.Module):
    """
    Root mean square layer normalization (Zhang & Sennrich, 2019).
    LayerNorm's minimalist cousin: just divide by the RMS over the trailing
    dimension and apply a learned gain. Turns out the centering wasn't doing much anyway.

    Args:
        d_model (int): Size of the normalized (trailing) dimension.
        eps (float): Small constant inside the root, for numerical stability.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype): Datatype of the parameters.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.eps: float = eps
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # Statistics in float32: a sum of d_model squares is where half-precision
        # dreams go to die.
        in_dtype = x.dtype
        x_fp32 = x.float()
        rmsx = torch.rsqrt(torch.mean(torch.square(x_fp32), dim=-1, keepdim=True) + self.eps)

        return (x_fp32 * rmsx).to(dtype=in_dtype) * self.weight

class DyT(nn.Module):
    """
    Dynamic Tanh (Zhu et al., 2025, "Transformers without Normalization"):
    a drop-in norm replacement, gamma * tanh(alpha * x) + beta. One parameter
    learns the squish, the other two the scale and shift.

    Args:
        d_model (int): Size of the trailing dimension.
        init_a (float): Initial value of the learnable input scale alpha.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype): Datatype of the parameters.
    """
    def __init__(self, d_model: int, init_a: float, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.a = nn.Parameter(torch.ones(1, **factory_kwargs) * init_a)
        self.gamma = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.beta = nn.Parameter(torch.zeros(d_model, **factory_kwargs))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.gamma * torch.tanh(self.a * x) + self.beta

class SiLu(nn.Module):
    """Sigmoid Linear Unit: x * sigmoid(x). Also answers to "swish" (Ramachandran et al., 2017)."""

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return x * torch.sigmoid(x)

class FFNSwiGLU(nn.Module):
    """
    SwiGLU feed-forward network (Shazeer, 2020):
        FFN(x) = (SiLU(xW_gate) * xW_value) W_out
    Three matrices where the vanilla FFN has two, so if `d_ff` is None
    the hidden size shrinks to 8/3 * d_model to hold the parameter count at
    the classic 8 * d_model^2 (rounded to a multiple of 64).
    Gate and value projections are fused into a single GEMM.

    Args:
        in_features (int): Model dimension (d_model).
        d_ff (int|None): hidden dimension FFN
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype): Datatype of the parameters.
    """
    def __init__(self, in_features: int, d_ff:int|None=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        # n. param. value projection = d_{model} /times k_{GLU} d_{model}
        # n. param. gate projection = d_{model} /times k_{GLU} d_{model}
        # n. param. output projection = k_{GLU} d_{model} /times d_{model}
        # In a FFN we'd have: 2 k d_{model}^2
        # to preserve the number of params 2 k = 3 k_{GLU} -> k_{GLU} = 2/3 k
        # if we pick k ~ 4 ==> k_{GLU} ~ 8/3
        if d_ff is None:
            d_ff = (8 * in_features) // 3
            d_ff = max(64, (d_ff // 64) * 64)

        # Fused gate and value projection
        self.w_gate_value = Linear(in_features, 2 * d_ff, **factory_kwargs)
        self.output_projection = Linear(d_ff, in_features, **factory_kwargs)
        self.act = SiLu()

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # Single matmul
        gate_value = self.w_gate_value(x)
        # Split along the feature dimension
        gate, value = gate_value.chunk(2, dim=-1)

        return self.output_projection(self.act(gate) * value)

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (Su et al., 2021).
    Encodes position by rotating each adjacent pair of features by an angle
    proportional to the token position; attention scores then depend only on
    relative offsets, which is all a causal LM really cares about.
    Uses the interleaved (GPT-J style) pairing convention.

    Args:
        theta (float): Base of the geometric frequency schedule.
        d_k (int): Feature dimension to rotate (head_dim). Must be even.
        max_seq_len (int): Sequence length to precompute the tables for.
        device (torch.device | None): Device to store the cosine/sine tables on.
    """
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even to prevent misbehaving with odd head dimensions"
        theta_k = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        t = torch.arange(max_seq_len, device=device).float()
        freqs: Float[Tensor, "max_seq_len d_k/2"] = torch.outer(t, theta_k)
        # repeat-interleave so each angle sits over its ADJACENT pair
        cos: Float[Tensor, "max_seq_len d_k"] = freqs.cos().repeat_interleave(2, dim=-1).float()
        sin = freqs.sin().repeat_interleave(2, dim=-1).float()
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    @staticmethod
    def rotate_interleaved(x):
        x1 = x[..., 0::2]                                   # x0, x2, x4, ...
        x2 = x[..., 1::2]                                   # x1, x3, x5, ...
        return torch.stack((-x2, x1), dim=-1).flatten(-2)  # [-x1, x0, -x3, x2, ...]

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        """
        Args:
            x (Float[Tensor, "... seq_len d_k"]): Tensor to rotate (Q or K).
            token_positions (Int[Tensor, "... seq_len"]): Position index per token;
                usually arange(seq_len), or a single offset when decoding.

        Returns:
            Float[Tensor, "... seq_len d_k"]: Rotated tensor, same shape and dtype.
        """
        # Tables are fp32; match x's dtype so we don't promote the whole graph.
        cos = self.cos_cached[token_positions].to(x.dtype)
        sin = self.sin_cached[token_positions].to(x.dtype)
        return x * cos + self.rotate_interleaved(x) * sin

def softmax(x: Float[Tensor, "... dim_size"], dim: int) -> Float[Tensor, "... dim_size"]:
    """
    Numerically stable softmax along `dim`.

    Args:
        x (Float[Tensor, "... dim_size"]): Input tensor.
        dim (int): Dimension along which to normalize.

    Returns:
        Float[Tensor, "... dim_size"]: Probabilities summing to 1 along `dim`.
    """
    # Subtracting the max so that all terms are (non-strictly) negative.
    # NOTE: no in-place div_ here — exp's backward needs its output alive.
    exp_x = torch.exp(x - x.amax(dim=dim, keepdim=True))

    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """
    softmax(QK^T / sqrt(d_k)) V (Vaswani et al., 2017).
    The scale is applied to the scores, where magnitudes are ~sqrt(d_k):
    a safer neighborhood for low-precision floats. This materializes the
    full [queries, keys] attention matrix

    Args:
        Q (Float[Tensor, "... queries d_k"]): Query tensor.
        K (Float[Tensor, "... keys d_k"]): Key tensor.
        V (Float[Tensor, "... keys d_v"]): Value tensor.
        mask (Bool[Tensor, "... queries keys"] | None): mask[..., i, j] is True if
            query i may attend to key j; False entries become -inf pre-softmax.

    Returns:
        Float[Tensor, "... queries d_v"]: Attention output.
    """
    d_k = Q.shape[-1]
    scores: Float[Tensor, "... queries keys"] = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores * d_k ** -0.5

    if mask is not None:
        # mask[i, j] is True if i attends to j -> set to -inf if False
        scores = scores.masked_fill(~mask, -torch.inf)
    weights = softmax(scores, dim=-1)

    return weights @ V

class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention with a fused QKV projection
    and optional RoPE applied to Q and K.

    Args:
        d_model (int): Model dimension. Must be divisible by num_heads.
        num_heads (int): Number of attention heads.
        rope (RotaryPositionalEmbedding | None): Shared RoPE module; all layers
            reuse the same instance (no parameters, just cached tables).
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype): Datatype of the parameters.
    """
    def __init__(self, d_model, num_heads, rope: "RotaryPositionalEmbedding|None"=None, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_heads = num_heads # nh
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads # H
        self.qkv_proj = Linear(d_model, 3 * d_model, **factory_kwargs)
        self.out_proj = Linear(d_model, d_model, **factory_kwargs)
        self.rope = rope

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        mask: Bool[Tensor, "seq_len seq_len"] | None = None,
        token_positions: Int[Tensor, "seq_len"] | None = None,
    ) -> Float[Tensor, "batch seq_len d_model"]:
        """
        Args:
            x (Float[Tensor, "batch seq_len d_model"]): Input sequence.
            mask (Bool[Tensor, "seq_len seq_len"] | None): Attention mask. If None,
                a causal mask is built on the fly — a fresh LxL allocation per
                layer per step, so pass one in if you have a cache handy.
            token_positions (Int[Tensor, "seq_len"] | None): Positions for RoPE;
                defaults to arange(seq_len).

        Returns:
            Float[Tensor, "batch seq_len d_model"]: Attention output.
        """
        B, L, D = x.size()
        if mask is None:
            mask: Bool[Tensor, "L L"] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        if token_positions is None:
            token_positions = torch.arange(L, device=x.device)

        qkv: Float[Tensor, "B L 3*D"] = self.qkv_proj(x)
        q, k, v = einx.id("B L (three nh H) -> three B nh L H",qkv, three=3, nh=self.num_heads)

        if self.rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        out: Float[Tensor, "B nh L H"] = scaled_dot_product_attention(q, k, v, mask)
        out: Float[Tensor, "B L D"] = einx.id("B nh L H -> B L (nh H)",out)
        
        return self.out_proj(out)

from typing import Optional
from collections.abc import Callable

def cross_entropy_loss(logits: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "... batch_size"]) -> Float[Tensor, "..."]:
    logits = logits.float()
    target_logits = torch.gather(
        logits, dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    # Numerical stability trick: logits = logits - logits.amax(dim=-1, keepdim= True)
    # torch.logsumexp already does the max-subtraction internally
    lse = torch.logsumexp(logits, dim=-1)

    return (lse - target_logits).mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def update_lr(self, lr: float):
        for group in self.param_groups:
            group['lr'] = lr

    @torch.no_grad()
    def step(self, closure: Optional[Callable]=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, wd, eps = group['lr'], group['weight_decay'], group['eps']
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad # gradient of loss with respect to p
                if len(state) == 0:
                    state['t'] = 1
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                m, v, t = state['m'], state['v'], state['t']

                if wd != 0:
                    p.mul_(1 - lr * wd)     # decoupled weight decay

                m.mul_(b1).add_(grad, alpha=1 - b1)              # Update momentum
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)    # Update 2nd moment

                alpha = lr * (math.sqrt(1 - b2**t)) / (1 - b1**t) # bias correction
                # v.sqrt() is out-of-place, so the .add_ isn't corrupting the state
                p.addcdiv_(m, v.sqrt().add_(eps), value=-alpha)  # moment-adjusted update

                state['t'] = t + 1
        return loss

def lr_cosine_scheduler(
        it: int,
        max_lr: float,
        min_lr: float,
        warmup_it: int,
        cos_cycle_it: int) -> float:
    """
        Given the parameters of a cosine learning rate decay schedule (with linear
        warmup) and an iteration number, return the learning rate at the given
        iteration under the specified schedule.

        Args:
            it (int): Iteration number to get learning rate for.
            max_lr (float): alpha_max, the maximum learning rate for
                cosine learning rate schedule (with warmup).
            min_lr (float): alpha_min, the minimum / final learning rate for
                the cosine learning rate schedule (with warmup).
            warmup_it (int): T_w, the number of iterations to linearly warm-up
                the learning rate.
            cos_cycle_it (int): T_c, the number of cosine annealing iterations.

        Returns:
            Learning rate at the given iteration under the specified schedule.
        """
    if it < warmup_it:
        return max_lr * it / warmup_it

    if it <= cos_cycle_it:
        return min_lr + 0.5 * (max_lr - min_lr) \
        * (1 + math.cos(math.pi *(it - warmup_it) / (cos_cycle_it - warmup_it)))

    return min_lr

@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-6 ) -> Tensor:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

        Args:
            parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
            max_l2_norm (float): a positive value containing the maximum l2-norm.
            eps (float): small epsilon to avoid numerical instability.

        The gradients of the parameters (parameter.grad) should be modified in-place.
        """
    grads = [p.grad for p in parameters if p.grad is not None]

    total_norm = torch.sqrt(sum(g.float().pow(2).sum() for g in grads))
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(clip_coef)

    return total_norm
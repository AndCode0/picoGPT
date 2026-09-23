from pathlib import Path
import typing
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from numpy.typing import NDArray
import nn_utils as nu
from nn_utils import RotaryPositionalEmbedding
import copy
from jaxtyping import Float, Bool, Int
from typing import Any, Mapping

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerBlock(nn.Module):
    """
    One pre-norm Transformer block:
        y = x + MHSA(RMSNorm(x));  z = y + FFN(RMSNorm(y))

    Pre-norm (Xiong et al., 2020) keeps the residual stream a clean identity
    path, which is what lets deep stacks train without learning-rate warm-up.

    Args:
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads.
        rope (RotaryPositionalEmbedding): Shared RoPE module; it holds no
            parameters, so all blocks happily reuse one instance.
        d_ff (int|None): Dimensionality of the feed-forward inner layer —  8/3 * d_model itself if None
        eps (float): RMSNorm epsilon.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype): Datatype of the parameters.
    """
    def __init__(self,
                 d_model: int,
                 n_heads:int,
                 rope: "RotaryPositionalEmbedding",
                 d_ff:int|None=None,
                 eps: float = 1e-5,
                 device=None,
                 dtype=None
                 ):

        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ln1 = nu.RMSNorm(d_model, eps, **factory_kwargs)
        self.mhsa = nu.MultiHeadSelfAttention(d_model, n_heads, rope, **factory_kwargs)
        self.ln2 = nu.RMSNorm(d_model, eps, **factory_kwargs)
        self.ffn = nu.FFNSwiGLU(d_model, d_ff, **factory_kwargs)

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        mask: Bool[Tensor, "seq_len seq_len"] | None = None,
        token_positions: Int[Tensor, "seq_len"] | None = None,
    ) -> Float[Tensor, "batch seq_len d_model"]:
        y = x + self.mhsa(self.ln1(x), mask=mask, token_positions=token_positions)
        return y + self.ffn(self.ln2(y))

class TransformerLM(nn.Module):
    """
    A decoder-only Transformer language model: token embeddings, a stack of
    pre-norm blocks sharing a single RoPE instance and a single cached causal
    mask, a final RMSNorm, and a (currently untied) projection back to vocabulary.

    Args:
        vocab_size (int): Size of the vocabulary.
        context_len (int): Maximum sequence length; sizes the RoPE tables and
            the cached causal mask.
        num_layers (int): Number of Transformer blocks.
        theta (float): RoPE base frequency.
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads per block.
        d_ff: Dimensionality of the feed-forward inner layer —  8/3 * d_model itself if None
        eps (float): RMSNorm epsilon.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype): Datatype of the parameters.
    """
    def __init__(
            self,
            vocab_size: int,
            context_len: int,
            num_layers: int,
            rope_theta: float,
            d_model: int,
            n_heads: int,
            d_ff: int|None=None,
            eps: float = 1e-5,
            device=None,
            dtype=None):

        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # Causal mask, cached once at context_len and sliced [:L, :L] per forward.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_len, context_len, dtype=torch.bool, device=device)),
            persistent=False
        )
        self.token_positions = torch.arange(context_len, device=device)
        self.tok_embedding = nu.Embedding(vocab_size, d_model, **factory_kwargs)
        assert d_model % n_heads == 0
        head_dim = d_model // n_heads
        rope = nu.RotaryPositionalEmbedding(rope_theta, head_dim, context_len, device)
        self.trans_stack = clones(TransformerBlock(d_model, n_heads, rope, d_ff, eps, **factory_kwargs), num_layers)
        self.norm = nu.RMSNorm(d_model, eps, **factory_kwargs)
        self.out_proj = nu.Linear(d_model, vocab_size, **factory_kwargs)

    def forward(
            self,
            x: Int[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len vocab_size"]:
        """
        Args:
            x (Int[Tensor, "batch seq_len"]): Token ids.

        Returns:
            Float[Tensor, "batch seq_len vocab_size"]: Raw logits.
        """
        L = x.shape[-1]
        tokens: Float[Tensor, "batch seq_len d_model"] = self.tok_embedding(x)
        mask: Bool[Tensor, "seq_len seq_len"] = self.causal_mask[:L, :L]
        tok_pos: Int[Tensor, "seq_len"] = self.token_positions[:L]
        for block in self.trans_stack:
            tokens = block(tokens, mask, tok_pos)
        tokens = self.norm(tokens)
        out: Float[Tensor, "batch seq_len vocab_size"] = self.out_proj(tokens)

        return out

def data_loading(
        dataset: NDArray,
        batch_size: int,
        context_length: int,
        gen: np.random.Generator|None=None,
        device: torch.device|None=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = context_length + 1
    # high is exclusive so it doesn't need the extra -1
    idx = gen.integers(low=0, high=len(dataset) - context_length, size=batch_size) if gen is not None \
        else np.random.randint(low=0, high=len(dataset) - context_length, size=batch_size)
    data = (dataset[idx[:, None] + np.arange(block_size)]).astype(np.int32)
    inputs = torch.from_numpy(data[:, :-1])
    targets = torch.from_numpy(data[:,1:])
    return inputs.to(device).long(), targets.to(device).long()

def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | Path | typing.BinaryIO | typing.IO[bytes],
        rng_state: Mapping[str, Any]|None=None,
):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "rng_state": rng_state,
    }, out)


def load_checkpoint(
        src: str | Path | typing.BinaryIO | typing.IO[bytes],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        rng: np.random.Generator|None=None,
        device: torch.device|None=None,
    ):
    checkpoint = torch.load(src, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "rng_state" in checkpoint and rng is not None:
        rng.bit_generator.state = checkpoint["rng_state"]

    return checkpoint["iteration"] + 1

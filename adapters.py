from __future__ import annotations
import math
import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
import numpy.typing as npt
import torch
from torch import Tensor
import regex as re
from collections import Counter,defaultdict
from torch import nn
import numpy as np
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    class Linear(nn.Module):
        def __init__(
            self,
            in_features:int,
            out_features:int,
            device:torch.device|None = None,
            dtype:torch.dtype|None = None
        ):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weights = nn.Parameter(
                torch.empty(out_features,in_features,dtype = dtype,device=device)
            )
            sigma = (2.0/(out_features+in_features)) **0.5
            nn.init.trunc_normal_(self.weights,mean=0,std = sigma,a = -3*sigma,b = 3*sigma)


        def forward(
            self,
            x:torch.Tensor
        ) -> torch.Tensor:
            return  x @ self.weights.T
        
    layer = Linear(d_in,d_out)
    layer.load_state_dict({"weights":weights})
    return layer(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    class Embedding(nn.Module):
        def __init__(
            self,
            num_embeddings:int,
            embedding_dim:int,
            device:torch.device|None = None,
            dtype:torch.dtype|None=None,
        ):
            super().__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim

            self.weights = nn.Parameter(
                torch.empty(num_embeddings,embedding_dim,dtype = dtype,device=device)
            )
            nn.init.trunc_normal_(self.weights,a =-3,b=3)

        def forward(
            self,
            token_ids:torch.Tensor
        )->torch.Tensor:
            return self.weights[token_ids] # 索引查找，tokenids是batchsize，seqlen，通过这个方式，对于seqlen的每一个token，变成
                                            # batchsize，seqlen，embeddinglen
    layer = Embedding(vocab_size,d_model)
    layer.load_state_dict({"weights":weights})
    return layer(token_ids)
            
    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight

    silu = in_features @ w1_weight.T
    silu = silu * torch.sigmoid(silu)
    w3x = in_features @ w3_weight.T
    gelu = silu * w3x
    out = gelu @ w2_weight.T
    return out




    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    d_k = torch.tensor(d_k)
    #  scores = Q K^T / sqrt(d_k)
    #    Q @ K^T -> (..., q, k)
    scores = Q @ K.transpose(-1, -2)
    scores = scores / torch.sqrt((d_k))

    if mask is not None:
        mask_bool = mask.to(dtype=torch.bool)
        scores = scores.masked_fill(~mask_bool, float("-inf"))

    attn = torch.softmax(scores, dim=-1)

    # attention probabilities of positions with a mask value of False should be zero.
    if mask is not None:
        all_false = (~mask_bool).all(dim=-1, keepdim=True)
        # 判断这个维度的元素是否都为True，非零视为true
        attn = torch.where(all_false, torch.zeros_like(attn), attn)
        # 如果allfalse成立，取torchzeroslike，否则取attn
    out = attn @ V
    return out


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    *prefix, S, d_in = in_features.shape

    # —— 0) 把权重改成 (H, D, d_in) 的三维，方便 einsum 分头并行
    H = num_heads
    HD = q_proj_weight.shape[0]
    assert HD % H == 0, "Q/K/V 的行数必须能被 num_heads 整除"
    D = HD // H  # head_dim

    Wq = q_proj_weight.reshape(H, D, d_in)      # [H, D, d_in]
    Wk = k_proj_weight.reshape(H, D, d_in)      # [H, D, d_in]
    Wv = v_proj_weight.reshape(H, D, d_in)      # [H, D, d_in]
    Wo = o_proj_weight.reshape(d_model, H, D)   # [d_model, H, D]

    x = in_features

    # —— 1) 一次性做 Q/K/V 投影：[..., S, d_in] × [H, D, d_in] -> [..., S, H, D]
    #    这里把 d_in 对齐在最后一个维度，用 '... s d, h k d -> ... s h k'
    Q = torch.einsum("... s d, h k d -> ... s h k", x, Wq)
    K = torch.einsum("... s d, h k d -> ... s h k", x, Wk)
    V = torch.einsum("... s d, h k d -> ... s h k", x, Wv)

    # —— 2) 注意力打分：[..., H, S, S]
    #    Q: [..., S, H, D], K: [..., S, H, D] → 先把 S 与 H 维移到合适位置，再做内积
    #    '... s h k, ... t h k -> ... h s t'
    scores = torch.einsum("... s h k, ... t h k -> ... h s t", Q, K) / math.sqrt(D)

    # —— 3) 因果 mask：上三角 True 的位置置 -inf
    causal = torch.triu(
        torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1
    )
    scores = scores.masked_fill(causal, float("-inf"))

    attn = scores.softmax(dim=-1)  # [..., H, S, T] 其中 T=S  # attn 是一个seqlen*seqlen的矩阵

    # —— 4) 上下文向量：attn @ V → [..., S, H, D]
    #    '... h s t, ... t h k -> ... s h k'
    ctx = torch.einsum("... h s t, ... t h k -> ... s h k", attn, V)
    #numhead seqlen seqlen , seqlen numhead headdim -> seqlen numhead headdim
    # —— 5) 合并各头并做输出投影：[..., S, H, D] × [d_model, H, D] -> [..., S, d_model]
    #    '... s h k, d h k -> ... s d'
    out = torch.einsum("... s h k, d h k -> ... s d", ctx, Wo) # 最后的输出是seqlen,d_model

    
    assert out.shape[-1] == d_model
    return out



    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    *prefix, S, d_in = in_features.shape

    H = num_heads
    HD = q_proj_weight.shape[0]
    assert HD % H == 0, "Q/K/V 的行数必须能被 num_heads 整除"
    D = HD // H  # head_dim

    Wq = q_proj_weight.reshape(H, D, d_in)      # [H, D, d_in]
    Wk = k_proj_weight.reshape(H, D, d_in)      # [H, D, d_in]
    Wv = v_proj_weight.reshape(H, D, d_in)      # [H, D, d_in]
    Wo = o_proj_weight.reshape(d_model, H, D)   # [d_model, H, D]

    x = in_features

    # 1) Batched Q/K/V projections -> [..., S, H, D]
    Q = torch.einsum("... s d, h k d -> ... s h k", x, Wq)
    K = torch.einsum("... s d, h k d -> ... s h k", x, Wk)
    V = torch.einsum("... s d, h k d -> ... s h k", x, Wv)

    assert D % 2 == 0, "head_dim must be even for RoPE"

    # RoPE on Q and K (with correct d_k = D)
    # ensure cache length covers all positions; use provided max_seq_len or max pos + 1
    rope_max = max(int(max_seq_len), int(token_positions.max().item()) + 1)
    Q = run_rope(D, theta, rope_max, Q, token_positions)
    K = run_rope(D, theta, rope_max, K, token_positions)

    # 2) Attention scores: [..., H, S, S]
    scores = torch.einsum("... s h k, ... t h k -> ... h s t", Q, K) / math.sqrt(D)

    # 3) Causal mask
    causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))

    attn = scores.softmax(dim=-1)  # [..., H, S, S]

    # 4) Context: [..., S, H, D]
    ctx = torch.einsum("... h s t, ... t h k -> ... s h k", attn, V)

    # 5) Merge heads + output proj: [..., S, d_model]
    out = torch.einsum("... s h k, d h k -> ... s d", ctx, Wo)

    assert out.shape[-1] == d_model
    return out

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    class RotaryPositionalEmbedding(nn.Module):
        def __init__(
            self,
            theta:float,
            d_k:int,
            max_seq_len:int,
            device=None,
        ):
            super().__init__()
            assert d_k % 2 == 0
            self.theta = theta
            self.d_k = d_k
            self.max_seq_len = max_seq_len
            k = torch.arange(0, d_k//2, device=device, dtype=torch.float32)
            inv_freq = 1.0 / (theta ** ((2*k) / d_k))
            position = torch.arange(0,max_seq_len,1,device=device,dtype=torch.float32)
            thetaik = torch.outer(position,inv_freq)

            self.register_buffer("cos_cached",torch.cos(thetaik),persistent=True)
            self.register_buffer("sin_cached",torch.sin(thetaik),persistent=True)

        def forward(
            self,
            x:torch.Tensor,
            token_positions:torch.Tensor
        )-> torch.Tensor:
            cos = self.cos_cached[token_positions]
            sin = self.sin_cached[token_positions]
            cos = cos.to(x.dtype).unsqueeze(-2)     # [..., S, 1, D/2] # 当使用多头的时候，需要添加numhead维度来广播 s:seqlen d:headdim
            sin = sin.to(x.dtype).unsqueeze(-2)     # [..., S, 1, D/2]
            # cos = cos.to(x.dtype)
            # sin = sin.to(x.dtype)
            x_even = x[...,::2]
            x_odd = x[...,1::2]

            rot_even = x_even *cos -x_odd*sin
            rot_odd = x_even*sin + x_odd*cos

            out = torch.empty_like(x)
            out[...,::2] = rot_even
            out[...,1::2] = rot_odd

            return out
            
    RoPE = RotaryPositionalEmbedding(
        theta = theta,
        d_k = d_k,
        max_seq_len= max_seq_len,
    )
    return RoPE(in_query_or_key,token_positions)
            
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    x = in_features
    batch,seqlen,d_model = x.shape
    x1 = run_rmsnorm(d_model,1e-5,weights["ln1.weight"],in_features)

    attn = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=x1,
        token_positions=torch.arange(seqlen),
    )

    out = in_features + attn

    x2 = run_rmsnorm(d_model,1e-5,weights["ln2.weight"],out)
    ffn = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=x2,
    )

    out2 = ffn + out
    return out2
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    x = in_indices
    batchsize,seqlen = x.shape
    embedding = run_embedding(vocab_size,d_model,weights=weights["token_embeddings.weight"],token_ids=x)
    for i in range(num_layers):
        rms = run_rmsnorm(d_model,eps=1e-5,weights=weights[f"layers.{i}.ln1.weight"],in_features=embedding)
        attn = run_multihead_self_attention_with_rope(
            d_model,num_heads,max_seq_len=seqlen,theta=rope_theta,q_proj_weight=weights[f"layers.{i}.attn.q_proj.weight"],
            k_proj_weight=weights[f"layers.{i}.attn.k_proj.weight"],v_proj_weight=weights[f"layers.{i}.attn.v_proj.weight"],
            o_proj_weight=weights[f"layers.{i}.attn.output_proj.weight"],in_features=rms,token_positions=torch.arange(seqlen)
        )
        out = embedding + attn

        rms = run_rmsnorm(d_model,eps=1e-5,weights=weights[f"layers.{i}.ln2.weight"],in_features=out)
        ffn = run_swiglu(d_model,d_ff,w1_weight=weights[f"layers.{i}.ffn.w1.weight"],w2_weight=weights[f"layers.{i}.ffn.w2.weight"],
                        w3_weight=weights[f"layers.{i}.ffn.w3.weight"],in_features=rms)
        out = out + ffn
        embedding = out

    out = run_rmsnorm(d_model,eps=1e-5,weights=weights["ln_final.weight"],in_features=out)
    #batchsize,seqlen,dmodel
    out = run_linear(d_model,vocab_size,weights=weights["lm_head.weight"],in_features=out)

    return out
    


    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    class RMSNorm(nn.Module):
        def __init__(
            self,
            d_model:int,
            eps:float = 1e-5,
            device:torch.device|None = None,
            dtype:torch.dtype|None = None
        ):
            super().__init__()
            self.eps = eps
            self.d_model = d_model
            self.weights = nn.Parameter(
                torch.ones(d_model,device=device,dtype=dtype)
            )

        def forward(
            self,
            x:torch.Tensor
        )-> torch.Tensor:
            in_dtype = x.dtype
            x = x.to(torch.float32)
            rms = torch.sqrt((x**2).mean(dim=-1,keepdim=True)+self.eps)
            result = (x/rms) * self.weights
            return result.to(in_dtype)
    
    layer = RMSNorm(d_model,eps,device=weights.device,dtype=weights.dtype)
    layer.load_state_dict({"weights":weights})
    return layer(in_features)
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    """
    兼容 ndarray 和 memmap。对 memmap 更友好：按样本顺序切片，减少随机 IO。
    返回 (inputs, targets)，形状均为 (B, m)，dtype=torch.long。
    """
    n = int(dataset.shape[0])
    m = int(context_length)
    B = int(batch_size)

    if n < m + 1:
        raise ValueError(f"Dataset too short: need at least context_length+1={m+1}, got {n}")

    # 采样起点：合法范围 [0, n-m-1]，np.random.randint 的 high 是开区间
    starts = np.random.randint(0, n - m, size=B, dtype=np.int64)  #得到B个起始点

    # 预分配输出缓冲（避免多次小数组分配；对 memmap 顺序读取友好）
    x_np = np.empty((B, m), dtype=np.int64)
    y_np = np.empty((B, m), dtype=np.int64)

    # 逐样本做连续切片：dataset[s:s+m] 和 dataset[s+1:s+m+1]
    # 这种方式对 memmap 是顺序读，比花式索引更少随机 IO
    for i, s in enumerate(starts):#index value
        x_np[i, :] = dataset[s : s + m]
        y_np[i, :] = dataset[s + 1 : s + m + 1]

    # 转 torch，放到目标设备
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.long)
    y = torch.from_numpy(y_np).to(device=device, dtype=torch.long)
    return x, y

    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max,_ = torch.max(in_features,dim=dim,keepdim=True)
    softmax = torch.exp(in_features-max) / torch.sum(torch.exp(in_features-max),dim=dim,keepdim=True)
    return softmax
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # 数值稳定：先减去每个样本内的最大值
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values               # (..., V)

    # logsumexp 计算 log Σ exp
    lse = torch.logsumexp(shifted, dim=-1)                                   # (...)

    # 取出目标类别对应的 logit
    correct_logit = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (...)
    #gather 负责取出dim维度中的index值
    # unsqueeze 在最后一维中添加一维 target.unsqueeze(-1) ->(batchsize,1)

    # 交叉熵：-log softmax = logsumexp - correct_logit
    loss_per_example = lse - correct_logit                                   # (...)

    return loss_per_example.mean()   
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    params = [p for p in parameters if p.grad is not None]
    
    if not params:
        return
    eps =1e-6
    total_norm = torch.norm(torch.stack([p.grad.detach().float().norm(2) for p in params]),2) # 将所有params中的2阶范数计算出来拼接成一个向量，然后再计算所有的params的二阶范数

    alpha = max_l2_norm / (total_norm+eps)

    if alpha < 1.0:
        return [p.grad.detach().mul_(alpha) for p in params]
    
    raise NotImplementedError


    
def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    class AdamW(torch.optim.Optimizer):
        def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay = 0.0):
            default = dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
            super().__init__(params,default)# 在创建优化器的时候，把要优化的参数和这些参数的默认超参传递给父类，能够帮助管理参数

        @torch.no_grad()
        def step(self,closure=None):
            loss=None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()# 重新计算一次 loss 并反向传播梯度。
            
            for group in self.param_groups:
                lr = group["lr"]
                beta1,beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                for p in group["params"]:
                    if p.grad == None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("sparse")
                    
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p,dtype=torch.float32,memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p,dtype=torch.float32,memory_format=torch.preserve_format)
                    
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    state["step"] += 1
                    t = state["step"]

                    if weight_decay != 0.0:
                        p.add_(p,alpha=-lr*weight_decay)  # 将权重解耦出来， θt+1​=θt​−m^t*​​α/(sqrt(\hatvt)+ϵ)−αλθt​  反正θt+1只有这臆想中更新，我们就先进行权重解耦
                        # 原地操作，进行p <- p + alpha*p 原地改 节省内存，效率更高

                    exp_avg.mul_(beta1).add_(grad,alpha=1-beta1)#mul_表示inplace乘法，  exp_avg = exp_avg * beta1   # 但不会新建张量，而是原地改
                    exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value=1-beta2)       #addcmul(input1, input2, value) = 当前张量 += value × (input1 × input2)

                    bias_correction1 = 1.0-beta1**t
                    bias_correction2 = 1.0-beta2**t
                    stepsize = lr*math.sqrt(bias_correction2)/bias_correction1
                    denom = exp_avg_sq.sqrt().add_(eps)
                    p.addcdiv_(exp_avg,denom,value=-stepsize)
            return loss
    return AdamW

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # 1) warmup: linearly from 0 -> alpha_max
    if warmup_iters > 0 and it < warmup_iters:
        lr = (it / float(warmup_iters)) * max_learning_rate
        return float(lr)

    # 2) cosine: from alpha_max -> alpha_min (smooth)
    # guard against division by zero when cosine_cycle_iters == warmup_iters
    if it <= cosine_cycle_iters and cosine_cycle_iters > warmup_iters:
        progress = (it - warmup_iters) / float(cosine_cycle_iters - warmup_iters)  # in [0,1]
        cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress))                  # in [0,1]
        lr = min_learning_rate + cosine_term * (max_learning_rate - min_learning_rate)
        return float(lr)

    # 3) after cosine: hold at alpha_min
    return float(min_learning_rate)
    

    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "iteration":int(iteration),
    }
    torch.save(checkpoint,out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    try:
        first_param = next(model.parameters())
        device = first_param.device
    except:
        device = torch.device("cpu")

    checkpoint = torch.load(src,map_location=device)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    iteration = int(checkpoint.get('iteration',0))#如果有返回，没有返回0

    return iteration

    raise NotImplementedError

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> "Any":
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>. Merges are ordered by order of creation.
        special_tokens (list[str] | None): Special tokens that must be kept intact.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    from typing import Any, Iterable, Iterator
    import heapq
    import regex as re

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    class Tokenizer:
        def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None,
        ):
            self.id_to_token = vocab
            self.token_to_id = {v: k for k, v in vocab.items()}
            self.merges = merges
            self.special_tokens = special_tokens or []

            self._pretok_re = re.compile(PAT)
            
            if self.special_tokens:
                specials_escaped = sorted(
                    (re.escape(t) for t in self.special_tokens),# 对于特殊的token进行转义成安全的正则表达式字符串
                    key=len,
                    reverse=True, #当出现"foo", "foobar"这样的组合的时候，如果长度顺序是从短到长，那么就会出现长的识别不到的问题
                )
                self._special_split_re = re.compile("(" + "|".join(specials_escaped) + ")")
            else:
                self._special_split_re = None

            # 为推理阶段准备 rank（pair → 次序），贪心按最小 rank 合并
            self.rank: dict[tuple[bytes, bytes], int] = {
                pair: i for i, pair in enumerate(self.merges)
            }

        def encode(self, text: str) -> list[int]:
            """非流式：返回整段文本的 token id 列表（用于兼容原测试用例）"""
            if not text:
                return []

            parts = (
                self._special_split_re.split(text)
                if self._special_split_re is not None
                else [text]
            )

            out_ids: list[int] = []
            for part in parts:
                if not part:
                    continue
                if part in self.special_tokens:
                    out_ids.append(self.token_to_id[part.encode("utf-8")])
                else:
                    for m in self._pretok_re.finditer(part):
                        out_ids.extend(self._encode_pretoken_iter(m.group(0)))
            return out_ids

        def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
            """流式：逐步产出 token id，避免在受限内存下构造大列表"""
            for chunk in iterable:
                if not chunk:
                    continue
                parts = (
                    self._special_split_re.split(chunk)
                    if self._special_split_re is not None
                    else [chunk]
                )
                for part in parts:
                    if not part:
                        continue
                    if part in self.special_tokens:
                        yield self.token_to_id[part.encode("utf-8")]
                    else:
                        for m in self._pretok_re.finditer(part):
                            for tid in self._encode_pretoken_iter(m.group(0)):
                                yield tid

        def decode(self, ids: list[int]) -> str:
            bb = bytearray()
            repl = "�".encode("utf-8")  # U+FFFD
            for i in ids:
                token_bytes = self.id_to_token.get(int(i))
                bb.extend(token_bytes if token_bytes is not None else repl)
            return bb.decode("utf-8", errors="replace")

        # ========== 内部方法 ==========
        def _encode_pretoken_iter(self, s: str) -> Iterator[int]:
            """单个预分词片段的 BPE 编码（贪心：按 merges 的 rank 最小优先合并）"""
            b = s.encode("utf-8")
            # 初始序列：逐字节
            seq = [bytes([x]) for x in b]
            if len(seq) < 2 or not self.rank:
                # 直接逐 token → id
                for tok in seq:
                    tid = self.token_to_id.get(tok)
                    if tid is not None:
                        yield tid
                    else:
                        for byte in tok:
                            yield self.token_to_id[bytes([byte])]
                return

            # 轻量双向链表，便于局部合并
            class Node:
                __slots__ = ("tok", "prev", "next", "alive", "id")
                _auto = 0
                def __init__(self, tok: bytes):
                    self.tok = tok
                    self.prev = None
                    self.next = None
                    self.alive = True
                    self.id = Node._auto
                    Node._auto += 1

            nodes = [Node(t) for t in seq]
            for i in range(len(nodes) - 1):
                nodes[i].next = nodes[i + 1]
                nodes[i + 1].prev = nodes[i]

            # 最小堆：元素为 (rank, ticket, left_node_id)
            heap: list[tuple[int, int, int]] = []
            ticket = 0

            def push_pair(left: "Node | None"):
                nonlocal ticket
                if left is None or left.next is None:
                    return
                if not left.alive or not left.next.alive:
                    return
                pair = (left.tok, left.next.tok)
                r = self.rank.get(pair)
                if r is None:
                    return
                heapq.heappush(heap, (r, ticket, left.id))
                ticket += 1

            # 初始把所有在 rank 里的相邻对入堆
            for i in range(len(nodes) - 1):
                push_pair(nodes[i])

            # id → node，用于检查过期
            id2node = {n.id: n for n in nodes}

            while heap:
                r, tk, left_id = heapq.heappop(heap)
                left = id2node.get(left_id)
                if not left or not left.alive:
                    continue
                right = left.next
                if not right or not right.alive:
                    continue
                # 确认 rank 未改变且仍相邻
                if self.rank.get((left.tok, right.tok)) != r:
                    continue

                # --- 执行合并：left + right ---
                merged_tok = left.tok + right.tok
                left.tok = merged_tok
                nxt = right.next
                right.alive = False
                left.next = nxt
                if nxt:
                    nxt.prev = left

                # 增量更新左右两侧候选对
                push_pair(left.prev)
                push_pair(left)

            # 找到链表头部
            cur = nodes[0]
            while cur and cur.prev:
                cur = cur.prev

            # 遍历存活节点，产出 id
            while cur:
                if cur.alive:
                    tid = self.token_to_id.get(cur.tok)
                    if tid is not None:
                        yield tid
                    else:
                        for byte in cur.tok:
                            yield self.token_to_id[bytes([byte])]
                cur = cur.next

    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # GPT-2

    if isinstance(input_path, os.PathLike):  # os.PathLike -> str
        input_path = os.fspath(input_path)

    base_vocab_size = 256 + len(special_tokens)
    if vocab_size < base_vocab_size:
        raise ValueError(
            f"vocab_size ({vocab_size}) is smaller than base size "
            f"(256 bytes + {len(special_tokens)} specials = {base_vocab_size})."
        )

    def iter_pretokens(text: str):
        rgx = re.compile(PAT)
        for m in rgx.finditer(text):
            yield m.group(0)

    def str_to_bytes_tokens(s: str) -> list[bytes]:
        b = s.encode("utf-8")
        return [bytes([x]) for x in b]  # FIX: bytes([i]) 而不是 bytes(i)

    def count_adjacent_pairs(tokens: list[bytes]) -> Counter[tuple[bytes, bytes]]:
        c: Counter[tuple[bytes, bytes]] = Counter()
        for i in range(len(tokens) - 1):
            c[(tokens[i], tokens[i + 1])] += 1
        return c

    def apply_merge_on_word(
        word_tokens: list[bytes],
        pair: tuple[bytes, bytes],
    ) -> list[bytes]:
        A, B = pair
        merge = A + B
        out: list[bytes] = []
        i = 0
        L = len(word_tokens)
        while i < L:
            if i < L - 1 and word_tokens[i] == A and word_tokens[i + 1] == B:
                out.append(merge)
                i += 2
            else:
                out.append(word_tokens[i])
                i += 1
        return out

    # ---- 读取语料 ----
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # ---- 先在文本层面“剥离” special tokens，作为边界（不参与训练）----  # FIX
    if special_tokens:
        sep = re.compile("|".join(re.escape(tok) for tok in special_tokens))
        chunks = [p for p in sep.split(text) if p]  # 丢弃 special 本体，只保留普通段
    else:
        chunks = [text]

    # ---- 预分词与去重计数 ----
    word_counts: dict[tuple[bytes, ...], int] = defaultdict(int)
    for chunk in chunks:
        for tok in iter_pretokens(chunk):
            seq = tuple(str_to_bytes_tokens(tok))
            if seq:
                word_counts[seq] += 1

    word: list[list[bytes]] = [list(i) for i in word_counts.keys()]
    freq: list[int] = [word_counts[i] for i in word_counts.keys()]

    # ---- 构建初始 vocab（保持你原始顺序：先 256 bytes，后 specials）----
    vocab: dict[int, bytes] = {}
    next_id = 0

    for i in range(256):
        vocab[next_id] = bytes([i])  # FIX: bytes([i])
        next_id += 1

    merges: list[tuple[bytes, bytes]] = []
    max_merge = vocab_size - 256 - len(special_tokens)

    if max_merge == 0 or not word:
        for sp in special_tokens:
            vocab[next_id] = sp.encode("utf-8")
            next_id += 1
        return vocab, merges

    pair_count: Counter[tuple[bytes, bytes]] = Counter()
    pair2word: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for wi, tokens in enumerate(word):
        if len(tokens) < 2:
            continue
        local = count_adjacent_pairs(tokens)
        if not local:
            continue
        f = freq[wi]
        for p, c in local.items():
            pair_count[p] += c * f
            pair2word[p].add(wi)

    for _ in range(max_merge):
        if not pair_count:
            break  # FIX: 没有可合并的 pair 了，直接退出循环

        best_pair, best_count = max(pair_count.items(), key=lambda kv: (kv[1], kv[0]))
        if best_count <= 0:
            break  # FIX: 同理，退出

        merges.append(best_pair)
        vocab[next_id] = best_pair[0] + best_pair[1]  # 新 token 进入词表
        next_id += 1                                  # FIX: 别忘了递增 id

        affected = list(pair2word.get(best_pair, ()))
        pair2word.pop(best_pair, None)

        for wi in affected:
            old_tokens = word[wi]
            if len(old_tokens) < 2:
                continue

            old_pairs = count_adjacent_pairs(old_tokens)
            old_pairs_w = Counter({p: c * freq[wi] for p, c in old_pairs.items()})

            new_tokens = apply_merge_on_word(old_tokens, best_pair)
            word[wi] = new_tokens

            new_pairs = count_adjacent_pairs(new_tokens)
            new_pairs_w = Counter({p: c * freq[wi] for p, c in new_pairs.items()})

            for p, c in old_pairs_w.items():
                pair_count[p] -= c
                if pair_count[p] <= 0:
                    pair_count.pop(p, None)
                    pair2word.pop(p, None)
            for p, c in new_pairs_w.items():
                pair_count[p] += c
                pair2word[p].add(wi)

    for sp in special_tokens:
        vocab[next_id] = sp.encode("utf-8")
        next_id += 1

    # 可选：确保长度不超过 vocab_size（通常不需要）
    if len(vocab) > vocab_size:
        for k in sorted(vocab.keys())[vocab_size:]:
            vocab.pop(k)

    return vocab, merges
                    
    raise NotImplementedError

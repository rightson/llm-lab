"""A teaching decoder: two pre-norm layers, one attention head, learned positions.

Random weights demonstrate mechanics, not language quality. No trained model is
downloaded. Shapes omit the batch dimension to keep the arithmetic visible.
"""
from dataclasses import dataclass
import numpy as np


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x):
    # Fixed gain=1 and bias=0 in this teaching model.
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(
        x.var(axis=-1, keepdims=True) + 1e-5
    )


def distribution(logits, temperature=1.0, top_k=None, top_p=1.0):
    """Temperature -> top-k -> top-p, with threshold-crossing token retained."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 1 or not np.isfinite(logits).all():
        raise ValueError("logits must be a finite vector")
    if temperature < 0 or not np.isfinite(temperature) or not 0 < top_p <= 1:
        raise ValueError("invalid temperature or top_p")
    if top_k is not None and (not isinstance(top_k, int) or not 1 <= top_k <= len(logits)):
        raise ValueError("top_k must be an integer within vocabulary size")
    if temperature == 0:
        p = np.zeros_like(logits)
        p[np.argmax(logits)] = 1
        return p
    p = softmax(logits / temperature)
    order = np.argsort(-p, kind="stable")
    if top_k is not None:
        p[order[top_k:]] = 0
        p /= p.sum()
    cumulative = np.cumsum(p[order])
    # Remove a token only if the mass BEFORE it already reaches the threshold.
    remove = np.concatenate(([False], cumulative[:-1] >= top_p))
    p[order[remove]] = 0
    return p / p.sum()


@dataclass
class TinyDecoder:
    vocab_size: int = 16
    width: int = 8
    layers: int = 2
    max_context: int = 128
    seed: int = 7

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        def weight(a, b):
            return rng.normal(0, 1 / np.sqrt(a), (a, b))
        d = self.width
        self.embedding = weight(self.vocab_size, d)
        self.position = weight(self.max_context, d)
        self.blocks = [dict(q=weight(d, d), k=weight(d, d), v=weight(d, d),
                            o=weight(d, d), up=weight(d, 2*d), down=weight(2*d, d))
                       for _ in range(self.layers)]
        self.head = weight(d, self.vocab_size)

    def forward(self, ids, cache=None):
        """Return logits at every supplied position and the extended per-layer KV.

        With a cache, ids contains ONLY new tokens. Cache positions define offset.
        Supports both one-token decode and a chunk of multiple new tokens.
        """
        ids = np.asarray(ids, dtype=int)
        if ids.ndim != 1 or len(ids) == 0 or np.any(ids < 0) or np.any(ids >= self.vocab_size):
            raise ValueError("ids must be a nonempty vector of valid token IDs")
        if cache is not None and len(cache) != self.layers:
            raise ValueError("one cache entry is required per layer")
        offset = 0 if cache is None else len(cache[0][0])
        n = len(ids)
        if offset + n > self.max_context:
            raise ValueError("context exceeds learned position table")
        x = self.embedding[ids] + self.position[offset:offset+n]
        new_cache = []
        for layer, block in enumerate(self.blocks):
            h = layer_norm(x)
            q, k, v = h @ block['q'], h @ block['k'], h @ block['v']
            if cache is not None:
                old_k, old_v = cache[layer]
                if old_k.shape != (offset, self.width) or old_v.shape != old_k.shape:
                    raise ValueError("cache shapes do not agree")
                k, v = np.concatenate((old_k, k)), np.concatenate((old_v, v))
            scores = q @ k.T / np.sqrt(self.width)
            # New query i may see old positions AND new positions through i.
            allowed = np.arange(offset+n)[None, :] <= (offset + np.arange(n))[:, None]
            scores = np.where(allowed, scores, -np.inf)
            x = x + (softmax(scores) @ v) @ block['o']
            h = layer_norm(x)
            x = x + np.maximum(h @ block['up'], 0) @ block['down']
            new_cache.append((k, v))
        return layer_norm(x) @ self.head, new_cache


def kv_bytes(tokens, layers, kv_heads, head_dim, bytes_per_element=2, batch=1):
    return 2 * batch * tokens * layers * kv_heads * head_dim * bytes_per_element


def online_attention(q, k, v, block_size=3):
    """One query, exact attention through blockwise online softmax (CPU demo)."""
    if block_size < 1 or len(k) == 0 or len(k) != len(v):
        raise ValueError("positive block size and nonempty matching K/V required")
    maximum, denominator = -np.inf, 0.0
    numerator = np.zeros(v.shape[1], dtype=float)
    for start in range(0, len(k), block_size):
        scores = q @ k[start:start+block_size].T / np.sqrt(len(q))
        new_maximum = max(maximum, float(scores.max()))
        correction = np.exp(maximum-new_maximum)
        p = np.exp(scores-new_maximum)
        numerator = correction*numerator + p @ v[start:start+block_size]
        denominator = correction*denominator + p.sum()
        maximum = new_maximum
    return numerator / denominator


def speculative_mass(p, q):
    """Analytic single-step accepted and corrected probability mass.

    This verifies the identity, not a full speculative generation implementation.
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.ndim != 1 or p.shape != q.shape or np.any(p < 0) or np.any(q < 0):
        raise ValueError("matching nonnegative probability vectors required")
    if not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1):
        raise ValueError("probability vectors must sum to one")
    accepted = np.minimum(p, q)
    residual = np.maximum(p-q, 0)
    rejection = residual.sum()
    correction = residual/rejection if rejection > 0 else np.zeros_like(p)
    return accepted, accepted + rejection*correction


def schedule(lengths, slots, continuous):
    """Unit-cost decode-only simulation; all requests arrive at t=0.

    No GPU timing, prefill, queue arrivals, or memory capacity model is implied.
    Returns makespan (rounds), each request's completion, and slot utilization.
    """
    if slots < 1 or not lengths or any(n < 1 for n in lengths):
        raise ValueError("positive lengths and slots required")
    remaining = list(lengths)
    active, cursor, clock = [], 0, 0
    completed = [None] * len(lengths)
    while cursor < len(lengths) or active:
        if continuous or not active:
            while len(active) < slots and cursor < len(lengths):
                active.append(cursor)
                cursor += 1
        clock += 1
        survivors = []
        for i in active:
            remaining[i] -= 1
            if remaining[i] == 0:
                completed[i] = clock
            else:
                survivors.append(i)
        active = survivors
    return clock, completed, sum(lengths) / (clock * slots)

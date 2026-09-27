"""Run with python -m labs.run all (or one experiment name)."""
import argparse
import codecs
import time
import numpy as np
from .core import (TinyDecoder, softmax, distribution, kv_bytes, schedule,
                   online_attention, speculative_mass)


def tokens():
    text = '重力 gravity'
    data = list(text.encode('utf-8'))
    print('text:', text, '| Unicode characters:', len(text), '| byte tokens:', len(data))
    print('byte IDs:', data)
    print('round-trip:', bytes(data).decode('utf-8'))
    # This is a BYTE tokenizer, NOT a production BPE implementation.
    pieces = list('banana')
    for pair in [('a', 'n'), ('b', 'an')]:
        result, i = [], 0
        while i < len(pieces):
            if i + 1 < len(pieces) and tuple(pieces[i:i+2]) == pair:
                result.append(''.join(pair)); i += 2
            else:
                result.append(pieces[i]); i += 1
        pieces = result
        print('illustrative fixed merge', pair, '=>', pieces)


def attention():
    q = np.array([1., 0.])
    k = np.array([[1., 0.], [0., 1.]])
    v = np.array([[10., 0.], [0., 20.]])
    scores = q @ k.T / np.sqrt(2)
    p = softmax(scores)
    print('scores:', scores, '| attention weights:', p, '| weighted values:', p @ v)


def training():
    # A trainable bigram model; it is not the TinyDecoder and has no attention.
    corpus = 'aba ' * 100 + 'aca ' * 25
    vocab = sorted(set(corpus))
    ids = np.array([vocab.index(c) for c in corpus])
    x, y = ids[:-1], ids[1:]
    w = np.zeros((len(vocab), len(vocab)))
    losses = []
    for step in range(201):
        p = softmax(w[x])
        loss = -np.log(p[np.arange(len(y)), y]).mean()
        if step in (0, 50, 200):
            print(f'step={step:3d} cross_entropy={loss:.4f}')
        losses.append(float(loss))
        grad = p.copy()
        grad[np.arange(len(y)), y] -= 1
        dw = np.zeros_like(w)
        np.add.at(dw, x, grad / len(y))
        w -= 2.0 * dw
    print('next character after a:', dict(zip(vocab, softmax(w[vocab.index('a')]).round(3))))
    return losses


def cache():
    model = TinyDecoder()
    ids = [1, 5, 3, 8, 2, 9]
    full, _ = model.forward(ids)
    first, state = model.forward(ids[:3])
    outputs = [first]
    for token in ids[3:]:
        logits, state = model.forward([token], state)
        outputs.append(logits)
    error = np.max(np.abs(full - np.concatenate(outputs)))
    print(f'full vs cached maximum absolute logit error: {error:.3e}')
    print('per-layer K,V shapes:', [(k.shape, v.shape) for k, v in state])
    print('NumPy cache bytes (float64 teaching implementation):', sum(k.nbytes+v.nbytes for k,v in state))
    return error


def sampling():
    logits = np.array([2., 1., 0.])
    for t in (0., .5, 1., 2.):
        print(f'T={t}:', distribution(logits, temperature=t).round(4))
    print('top-p=.8:', distribution(np.log([.6,.3,.1]), top_p=.8))
    print('top-k=2:', distribution(logits, top_k=2).round(4))


def generation():
    model = TinyDecoder()
    ids = [1, 2, 3]
    logits, state = model.forward(ids)
    generated = []
    for i in range(8):
        token = int(np.argmax(logits[-1]))
        generated.append(token)
        # The first output came from prefill. Feed it to obtain the next output.
        if i < 7:
            logits, state = model.forward([token], state)
    print('random-weight model token IDs (NOT meaningful language):', generated)
    decoder = codecs.getincrementaldecoder('utf-8')()
    chunks = [b'\xe9', b'\x87\x8d', b'\xe5\x8a', b'\x9b']
    print('UTF-8 stream chunks:', [decoder.decode(c, final=False) for c in chunks] + [decoder.decode(b'', final=True)])


def memory():
    for heads in (32, 8, 1):
        size = kv_bytes(8192, 32, heads, 128)
        print(f'KV heads={heads:2d}, 8192 tokens, 32 layers: {size/2**30:.3f} GiB per request')
    weight = 8_000_000_000 * 2
    print(f'8B BF16 weights: {weight/1e9:.1f} GB = {weight/2**30:.2f} GiB (weights only)')


def scheduling():
    lengths = [1, 5, 1, 5]
    for continuous in (False, True):
        rounds, ends, utilization = schedule(lengths, 2, continuous)
        print('continuous' if continuous else 'static',
              '| rounds:', rounds, '| completion:', ends, '| slot utilization:', round(utilization, 3))


def quantization():
    weights = np.array([-.9, -.2, .1, .8, 4.])
    for bits in (8, 4):
        limit = 2**(bits-1)-1
        scale = np.max(np.abs(weights)) / limit
        q = np.clip(np.round(weights / scale), -limit, limit)
        restored = q * scale
        print(f'{bits}-bit symmetric values:', q, '| restored:', restored.round(3),
              '| max error:', round(float(np.max(np.abs(weights-restored))), 4))
    print('Nominal arrays above illustrate rounding; they are not packed integer kernels.')
    print('Illustrative bandwidth lower bound: 16GB / 800GB/s = 20ms per step.')


def timing():
    model = TinyDecoder(width=32, max_context=512)
    rng = np.random.default_rng(3)
    model.forward([1,2,3])  # warmup
    for n in (16, 64, 128):
        ids = rng.integers(0, model.vocab_size, n).tolist()
        prefill, decode = [], []
        for _ in range(5):
            begin = time.perf_counter()
            logits, state = model.forward(ids)
            prefill.append((time.perf_counter()-begin)*1000)
            token = int(np.argmax(logits[-1]))
            begin = time.perf_counter()
            model.forward([token], state)
            decode.append((time.perf_counter()-begin)*1000)
        print(f'prompt={n}, median forward prefill={np.median(prefill):.3f}ms, '
              f'one cached forward={np.median(decode):.3f}ms')
    print('CPU toy timings exclude queue/network/sampling; NOT GPU benchmarks or service TTFT.')


def online():
    rng = np.random.default_rng(19)
    q, k, v = rng.normal(size=4), rng.normal(size=(17,4)), rng.normal(size=(17,6))
    dense = softmax(q @ k.T / 2) @ v
    blocked = online_attention(q, k, v, block_size=3)
    print('dense vs blockwise attention maximum error:', np.max(np.abs(dense-blocked)))
    print('CPU one-query mathematical demo, not a FlashAttention GPU implementation.')


def speculation():
    target, draft = np.array([.6,.3,.1]), np.array([.5,.2,.3])
    accepted, recovered = speculative_mass(target,draft)
    print('target p:', target, '| draft q:', draft)
    print('accepted mass:', accepted, '| acceptance probability:', accepted.sum())
    print('corrected output distribution:', recovered)
    print('Analytic one-step verification; no draft model or performance benchmark.')


EXPERIMENTS = dict(tokens=tokens, attention=attention, training=training, cache=cache,
                   sampling=sampling, generation=generation, memory=memory,
                   scheduling=scheduling, quantization=quantization, timing=timing,
                   online=online, speculation=speculation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=['all', *EXPERIMENTS])
    args = parser.parse_args()
    chosen = EXPERIMENTS if args.experiment == 'all' else {args.experiment: EXPERIMENTS[args.experiment]}
    for name, run in chosen.items():
        print(f'\n[{name}]')
        run()


if __name__ == '__main__':
    main()

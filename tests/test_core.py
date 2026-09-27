import unittest
import numpy as np
from labs.core import (TinyDecoder, distribution, kv_bytes, schedule, softmax,
                       online_attention, speculative_mass)
from labs.run import training


class InferenceInvariants(unittest.TestCase):
    def test_causal_prefix_independence(self):
        m = TinyDecoder()
        short, _ = m.forward([1,2,3])
        long, _ = m.forward([1,2,3,8,9])
        np.testing.assert_allclose(short, long[:3], atol=1e-12)

    def test_every_cache_chunking_matches_full(self):
        m = TinyDecoder()
        ids = [1,2,3,4,5,6,7]
        full, _ = m.forward(ids)
        for cut in range(1, len(ids)):
            first, cache = m.forward(ids[:cut])
            rest, cache = m.forward(ids[cut:], cache)
            np.testing.assert_allclose(full, np.concatenate((first, rest)), atol=1e-12)
            self.assertEqual(cache[0][0].shape[0], len(ids))

    def test_context_overflow_rejected(self):
        m = TinyDecoder(max_context=4)
        _, cache = m.forward([1,2,3,4])
        with self.assertRaises(ValueError):
            m.forward([5], cache)

    def test_softmax_stability_and_shift(self):
        np.testing.assert_allclose(softmax([1000,1001]), softmax([0,1]))

    def test_top_p_keeps_crossing_token(self):
        p = distribution(np.log([.6,.3,.1]), top_p=.8)
        np.testing.assert_allclose(p, [2/3, 1/3, 0])

    def test_top_k_and_greedy(self):
        np.testing.assert_allclose(distribution([1,3,2], temperature=0), [0,1,0])
        self.assertEqual(np.count_nonzero(distribution([1,3,2], top_k=2)), 2)
        self.assertAlmostEqual(distribution([1,3,2], top_k=2).sum(), 1)

    def test_kv_budget(self):
        self.assertEqual(kv_bytes(8192,32,8,128), 2**30)
        self.assertEqual(kv_bytes(8192,32,32,128), 4*2**30)

    def test_scheduler_conserves_tokens(self):
        for mode, expected in [(False,10),(True,7)]:
            rounds, ends, u = schedule([1,5,1,5],2,mode)
            self.assertEqual(rounds,expected)
            self.assertEqual(max(ends),rounds)
            self.assertAlmostEqual(u * rounds * 2,12)

    def test_bigram_learns(self):
        loss = training()
        self.assertLess(loss[-1], loss[0] - .3)

    def test_online_attention_matches_dense_across_blocks(self):
        rng = np.random.default_rng(31)
        q, k, v = rng.normal(size=4), rng.normal(size=(17,4)), rng.normal(size=(17,6))
        expected = softmax(q @ k.T / 2) @ v
        for size in [1,3,8,17,32]:
            np.testing.assert_allclose(online_attention(q,k,v,size), expected, atol=1e-12)

    def test_online_attention_large_scores(self):
        q = np.array([1.])
        k = np.array([[1000.],[1001.],[-1000.]])
        v = np.array([[1.,2.],[3.,4.],[5.,6.]])
        np.testing.assert_allclose(online_attention(q,k,v,1), softmax(k[:,0]) @ v)

    def test_speculative_mass_recovers_target_including_zero_support(self):
        for p,q in [([.6,.3,.1],[.5,.2,.3]),([1.,0.],[0.,1.]),([.2,.8],[.2,.8])]:
            accepted,recovered = speculative_mass(p,q)
            np.testing.assert_allclose(recovered,p)
            self.assertLessEqual(accepted.sum(), 1)


if __name__ == '__main__':
    unittest.main()

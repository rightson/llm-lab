import unittest
import numpy as np
from codesign.baseline import signed,requant,PE,ws_mesh


class CodesignContract(unittest.TestCase):
    def test_exhaustive_int8_products(self):
        pe=PE()
        for w in range(-128,128):
            pe.tick(weight=w,load=True)
            for a in range(-128,128):
                out,psum=pe.tick(act=a,psum=17)
                self.assertEqual(out,a)
                self.assertEqual(psum,a*w+17)

    def test_overflow_wrap_and_explicit_enable(self):
        pe=PE()
        pe.tick(weight=1,load=True)
        self.assertEqual(pe.tick(act=1,psum=2**31-1),(1,-2**31))
        self.assertEqual(pe.tick(act=99,psum=33,enable=False),(1,-2**31))
        self.assertEqual(pe.tick(reset=True,enable=False),(0,0))

    def test_requant_signedness_shift_and_saturation(self):
        self.assertEqual(requant(-3,1,1),-2)
        self.assertEqual(requant(200),127)
        self.assertEqual(requant(-200),-128)
        self.assertEqual(requant(1,0xffffffff,0),-1)
        self.assertEqual(requant(2**31),-128)
        with self.assertRaises(ValueError): requant(1,1,32)

    def test_requant_must_follow_reduction(self):
        self.assertNotEqual(requant(1,1,1)+requant(1,1,1),requant(2,1,1))

    def test_mesh_matches_independent_integer_matmul(self):
        rng=np.random.default_rng(9)
        for s in (2,4,8):
            for m in (1,s,2*s+1):
                for k,n in [(s,s),(s-1,1)]:
                    a=rng.integers(-128,128,(m,k),dtype=np.int64)
                    b=rng.integers(-128,128,(k,n),dtype=np.int64)
                    result,stats,trace=ws_mesh(a,b,s)
                    np.testing.assert_array_equal(result,a@b)
                    self.assertEqual(sum(x['useful_macs'] for x in trace),m*k*n)
                    self.assertEqual(sum(len(x['outputs']) for x in trace),m*n)
                    self.assertEqual(stats['total_cycles'],m+3*s-2)

    def test_invalid_shape_or_precision_rejected(self):
        with self.assertRaises(ValueError): ws_mesh([[1.5]],[[2]],2)
        with self.assertRaises(ValueError): ws_mesh([[128]],[[2]],2)
        with self.assertRaises(ValueError): ws_mesh([[1,2]],[[2]],2)
        self.assertEqual(signed(255,8),-1)


if __name__=='__main__': unittest.main()

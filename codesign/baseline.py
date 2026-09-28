"""Explicit integer semantics and a no-stall, weight-stationary mesh model.

The mesh follows the PE transport idea inspected in nanoNPU, but does not model
its top-level CU, SRAM, UART, output deskew circuitry, or exact instruction timing.
All cycles reported here belong to THIS educational schedule only.
"""
from dataclasses import dataclass
import numpy as np


def signed(value, bits):
    value = int(value) & ((1 << bits) - 1)
    return value - (1 << bits) if value >= (1 << (bits-1)) else value


def requant(acc, multiplier=1, shift=0):
    """INT32 * signed INT32 -> INT64 -> arithmetic shift -> INT8 saturation.

    Shift range 0..31 matches the inspected five-bit field. Python integers avoid
    accidental host overflow. Negative right shifts round toward minus infinity.
    """
    if not isinstance(shift, int) or not 0 <= shift <= 31:
        raise ValueError('shift must be an integer in 0..31')
    value = (signed(acc,32) * signed(multiplier,32)) >> shift
    return min(127, max(-128, value))


@dataclass
class PE:
    """Proposed reference PE: nanoNPU-like math with an explicit global enable.

    The added enable is our design contract, not a claim that nanoNPU has it.
    tick returns registered values AFTER the edge. Reset has highest priority.
    """
    weight: int = 0
    act: int = 0
    psum: int = 0

    def tick(self, act=0, psum=0, weight=0, load=False, enable=True, reset=False):
        if reset:
            self.weight = self.act = self.psum = 0
        elif enable:
            if load:
                self.weight = signed(weight,8)
            else:
                self.act = signed(act,8)
                self.psum = signed(signed(psum,32) + self.act*self.weight,32)
        return self.act, self.psum


def _int8_matrix(x):
    x = np.asarray(x)
    if x.ndim != 2 or not np.issubdtype(x.dtype, np.integer):
        raise ValueError('a two-dimensional integer matrix is required')
    if np.any(x < -128) or np.any(x > 127):
        raise ValueError('matrix values must fit signed INT8')
    return x.astype(np.int64)


def ws_mesh(a,b,size=8):
    """Stream M rows through an SxS mesh with K,N <= S; zero pad unused lanes.

    Weight load is charged S cycles, followed by M+2S-2 feed/drain cycles.
    Clock t injects A[t-r,r] at row r. Registered psums move down one row,
    registered activations move right one column. Results are collected at
    t=m+S-1+j. Arbitrary M streaming is a proposal, not the nanoNPU CU protocol.
    """
    a,b = _int8_matrix(a), _int8_matrix(b)
    m,k = a.shape
    kb,n = b.shape
    if size < 1 or k != kb or min(m,k,n) < 1 or max(k,n) > size:
        raise ValueError('require nonempty A[M,K], B[K,N] with K,N <= size')
    weights = np.zeros((size,size),dtype=np.int64)
    weights[:k,:n] = b
    padded = np.zeros((m,size),dtype=np.int64)
    padded[:,:k] = a
    acts = np.zeros_like(weights)
    psums = np.zeros_like(weights)
    output = np.zeros((m,n),dtype=np.int64)
    cycles = m+2*size-2
    trace=[]
    for t in range(cycles):
        incoming_a = np.zeros_like(weights)
        incoming_a[:,1:] = acts[:,:-1]
        for row in range(size):
            index = t-row
            if 0 <= index < m:
                incoming_a[row,0] = padded[index,row]
        incoming_p = np.zeros_like(weights)
        incoming_p[1:,:] = psums[:-1,:]
        # Products and one addition fit int64; wrap to the PE's 32-bit register.
        total = incoming_p + incoming_a*weights
        psums = ((total + 2**31) % 2**32) - 2**31
        acts = incoming_a
        emitted=[]
        for col in range(n):
            row = t-(size-1+col)
            if 0 <= row < m:
                output[row,col]=psums[-1,col]
                emitted.append([row,col,int(output[row,col])])
        useful = sum(0 <= t-r-c < m for r in range(k) for c in range(n))
        trace.append(dict(feed_cycle=t,useful_macs=useful,outputs=emitted))
    stats=dict(array_size=size,shape=[m,k,n],weight_load_cycles=size,
               feed_drain_cycles=cycles,total_cycles=size+cycles,
               useful_macs=m*k*n,
               slot_utilization=(m*k*n)/(size*size*(size+cycles)),
               timing_scope='educational resident-weight schedule, not nanoNPU top-level timing')
    return output,stats,trace


def main():
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=8,choices=[2,4,8,16])
    parser.add_argument('--rows',type=int,default=8)
    parser.add_argument('--trace',action='store_true')
    args=parser.parse_args()
    if args.rows < 1:
        parser.error('--rows must be positive')
    rng=np.random.default_rng(42)
    a=rng.integers(-128,128,(args.rows,args.size),dtype=np.int64)
    b=rng.integers(-128,128,(args.size,args.size),dtype=np.int64)
    result,stats,trace=ws_mesh(a,b,args.size)
    expected=a @ b
    np.testing.assert_array_equal(result,expected)
    print(json.dumps(dict(**stats,integer_reference_equal=True),indent=2))
    if args.trace:
        print(json.dumps(trace,indent=2))
    print('requant(-3, 1, 1) =',requant(-3,1,1),'(arithmetic shift, not truncation toward zero)')
    print('early requant:',requant(1,1,1)+requant(1,1,1),
          '| after reduction:',requant(1+1,1,1))


if __name__=='__main__':
    main()

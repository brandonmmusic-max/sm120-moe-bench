# SM120 MMA Fragment Layout — Critical Finding

## Discovery

On SM120 (RTX PRO 6000 Blackwell), `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
accepts 4 A-fragment registers and 2 B-fragment registers, but **only produces output for
rows 0-7 in the C accumulator**. Rows 8-15 (c[2] and c[3]) are always zero.

This means SM120 effectively implements m16n8k16 as **m8n8k16** — it uses the extra A
fragments (rows 8-15) as additional K-dimension data, not additional M-dimension rows.

## Empirical Evidence

Test: A[m, 0] = m+1, K[n, 0] = n+1, C[m, n] = (m+1)*(n+1)

All-ones test: 32×4 fragments all = 16.0 ← PASSES (but ambiguous)
Structured test: only rows 0-7 appear in output, rows 8-15 = 0

## Verified Fragment Mapping (SM120)

For `mma.sync.m16n8k16` on SM120:

```
c[0] → C[lane/4, (lane%4)*2]       # row 0-7, even col
c[1] → C[lane/4, (lane%4)*2 + 1]   # row 0-7, odd col
c[2] → always 0 (or extra k-dim accumulation?)
c[3] → always 0
```

## Implication for Kernel Design

Must use TWO mma.sync calls to cover a full m16 tile:
1. First MMA: A rows [0..7], produces C rows [0..7]
2. Second MMA: A rows [8..15], produces C rows [8..15]

Or use smaller tiles (BLOCK_M=8 per MMA iteration instead of 16).

## TODO

- Verify if m16n8k32 has the same behavior
- Test if the A[2] and A[3] registers are used for extended K (k=16..31)
- Check PTX ISA for SM120-specific mma documentation

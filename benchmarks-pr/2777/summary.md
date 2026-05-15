| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-5fe86793edfbaadc9ae49aa259adeb8f4c00bcb4.md) | 1,833 |  4,000,051 |  439 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-5fe86793edfbaadc9ae49aa259adeb8f4c00bcb4.md) | 13,996 |  14,365,133 |  2,362 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-5fe86793edfbaadc9ae49aa259adeb8f4c00bcb4.md) | 8,145 |  11,167,961 |  891 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-5fe86793edfbaadc9ae49aa259adeb8f4c00bcb4.md) | 1,562 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-5fe86793edfbaadc9ae49aa259adeb8f4c00bcb4.md) | 604 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-5fe86793edfbaadc9ae49aa259adeb8f4c00bcb4.md) | 740 |  592,827 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-5fe86793edfbaadc9ae49aa259adeb8f4c00bcb4.md) | 1,888 |  1,979,971 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5fe86793edfbaadc9ae49aa259adeb8f4c00bcb4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25927969290)

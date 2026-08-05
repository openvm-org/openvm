| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-4e9099fe3a98eff9f9be0b11c46cb7772be0b0fc.md) | 467 |  4,000,051 |  226 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-4e9099fe3a98eff9f9be0b11c46cb7772be0b0fc.md) | 7,416 |  14,365,133 |  1,516 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-4e9099fe3a98eff9f9be0b11c46cb7772be0b0fc.md) | 4,130 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-4e9099fe3a98eff9f9be0b11c46cb7772be0b0fc.md) | 656 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-4e9099fe3a98eff9f9be0b11c46cb7772be0b0fc.md) | 223 |  112,210 |  193 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-4e9099fe3a98eff9f9be0b11c46cb7772be0b0fc.md) | 233 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-4e9099fe3a98eff9f9be0b11c46cb7772be0b0fc.md) | 2,018 |  1,979,971 |  524 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4e9099fe3a98eff9f9be0b11c46cb7772be0b0fc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31050685175)

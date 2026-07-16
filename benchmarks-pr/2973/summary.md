| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-d0d09a2f6b847128a8d0617960f0c4774167149e.md) | 412 |  4,000,051 |  227 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-d0d09a2f6b847128a8d0617960f0c4774167149e.md) | 8,391 |  14,365,133 |  1,518 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-d0d09a2f6b847128a8d0617960f0c4774167149e.md) | 4,156 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-d0d09a2f6b847128a8d0617960f0c4774167149e.md) | 509 |  4,090,656 |  191 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-d0d09a2f6b847128a8d0617960f0c4774167149e.md) | 223 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-d0d09a2f6b847128a8d0617960f0c4774167149e.md) | 277 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-d0d09a2f6b847128a8d0617960f0c4774167149e.md) | 1,994 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d0d09a2f6b847128a8d0617960f0c4774167149e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29493313636)

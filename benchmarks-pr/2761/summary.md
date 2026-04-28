| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2761/fibonacci-fdf2f080353bc94821cbc8e64355caafa288cc66.md) | 1,910 |  4,000,051 |  546 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2761/keccak-fdf2f080353bc94821cbc8e64355caafa288cc66.md) | 13,628 |  14,365,133 |  2,252 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2761/sha2_bench-fdf2f080353bc94821cbc8e64355caafa288cc66.md) | 9,352 |  11,167,961 |  1,262 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2761/regex-fdf2f080353bc94821cbc8e64355caafa288cc66.md) | 1,582 |  4,090,656 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2761/ecrecover-fdf2f080353bc94821cbc8e64355caafa288cc66.md) | 643 |  112,210 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2761/pairing-fdf2f080353bc94821cbc8e64355caafa288cc66.md) | 762 |  592,827 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2761/kitchen_sink-fdf2f080353bc94821cbc8e64355caafa288cc66.md) | 2,053 |  1,979,971 |  431 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fdf2f080353bc94821cbc8e64355caafa288cc66

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25058490471)

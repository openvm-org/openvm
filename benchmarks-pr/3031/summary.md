| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/fibonacci-7eab43767058df06305f16b8692c2129733bd95e.md) | 410 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/keccak-7eab43767058df06305f16b8692c2129733bd95e.md) | 8,581 |  14,365,133 |  1,517 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/sha2_bench-7eab43767058df06305f16b8692c2129733bd95e.md) | 4,197 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/regex-7eab43767058df06305f16b8692c2129733bd95e.md) | 580 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/ecrecover-7eab43767058df06305f16b8692c2129733bd95e.md) | 217 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/pairing-7eab43767058df06305f16b8692c2129733bd95e.md) | 283 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/kitchen_sink-7eab43767058df06305f16b8692c2129733bd95e.md) | 1,919 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7eab43767058df06305f16b8692c2129733bd95e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29574252564)

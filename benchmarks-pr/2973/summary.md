| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-c9acdfc8756fb9ed314fb6495f44f4df233270d5.md) | 414 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-c9acdfc8756fb9ed314fb6495f44f4df233270d5.md) | 8,750 |  14,365,133 |  1,550 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-c9acdfc8756fb9ed314fb6495f44f4df233270d5.md) | 4,210 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-c9acdfc8756fb9ed314fb6495f44f4df233270d5.md) | 572 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-c9acdfc8756fb9ed314fb6495f44f4df233270d5.md) | 227 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-c9acdfc8756fb9ed314fb6495f44f4df233270d5.md) | 285 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-c9acdfc8756fb9ed314fb6495f44f4df233270d5.md) | 2,040 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c9acdfc8756fb9ed314fb6495f44f4df233270d5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29684794973)

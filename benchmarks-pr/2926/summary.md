| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/fibonacci-f93ba4e24941e820448f8248f6d6da95ab2d8956.md) | 1,044 |  4,000,051 |  399 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/keccak-f93ba4e24941e820448f8248f6d6da95ab2d8956.md) | 16,461 |  14,365,133 |  3,047 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/sha2_bench-f93ba4e24941e820448f8248f6d6da95ab2d8956.md) | 8,250 |  11,167,961 |  1,001 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/regex-f93ba4e24941e820448f8248f6d6da95ab2d8956.md) | 1,221 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/ecrecover-f93ba4e24941e820448f8248f6d6da95ab2d8956.md) | 438 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/pairing-f93ba4e24941e820448f8248f6d6da95ab2d8956.md) | 597 |  592,827 |  292 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/kitchen_sink-f93ba4e24941e820448f8248f6d6da95ab2d8956.md) | 3,880 |  1,979,971 |  863 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f93ba4e24941e820448f8248f6d6da95ab2d8956

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28192999167)

| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/fibonacci-c085e4977313572bd0a0c1e44340179a3670e4f0.md) | 471 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/keccak-c085e4977313572bd0a0c1e44340179a3670e4f0.md) | 7,333 |  14,365,133 |  1,547 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/sha2_bench-c085e4977313572bd0a0c1e44340179a3670e4f0.md) | 4,757 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/regex-c085e4977313572bd0a0c1e44340179a3670e4f0.md) | 669 |  4,090,656 |  221 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/ecrecover-c085e4977313572bd0a0c1e44340179a3670e4f0.md) | 228 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/pairing-c085e4977313572bd0a0c1e44340179a3670e4f0.md) | 307 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/kitchen_sink-c085e4977313572bd0a0c1e44340179a3670e4f0.md) | 2,691 |  1,979,971 |  477 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c085e4977313572bd0a0c1e44340179a3670e4f0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30132951218)

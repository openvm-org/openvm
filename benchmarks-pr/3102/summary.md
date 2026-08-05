| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/fibonacci-aaaf13b3e42922b10e069a720cd47a7d53cbbec6.md) | 479 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/keccak-aaaf13b3e42922b10e069a720cd47a7d53cbbec6.md) | 7,424 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/sha2_bench-aaaf13b3e42922b10e069a720cd47a7d53cbbec6.md) | 4,137 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/regex-aaaf13b3e42922b10e069a720cd47a7d53cbbec6.md) | 657 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/ecrecover-aaaf13b3e42922b10e069a720cd47a7d53cbbec6.md) | 223 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/pairing-aaaf13b3e42922b10e069a720cd47a7d53cbbec6.md) | 236 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/kitchen_sink-aaaf13b3e42922b10e069a720cd47a7d53cbbec6.md) | 2,044 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/aaaf13b3e42922b10e069a720cd47a7d53cbbec6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31025592081)

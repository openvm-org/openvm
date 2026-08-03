| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/fibonacci-19996c416e742eb157beb00f7d71d30b6df64dea.md) | 471 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/keccak-19996c416e742eb157beb00f7d71d30b6df64dea.md) | 7,424 |  14,365,133 |  1,521 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/sha2_bench-19996c416e742eb157beb00f7d71d30b6df64dea.md) | 4,158 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/regex-19996c416e742eb157beb00f7d71d30b6df64dea.md) | 648 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/ecrecover-19996c416e742eb157beb00f7d71d30b6df64dea.md) | 229 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/pairing-19996c416e742eb157beb00f7d71d30b6df64dea.md) | 245 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/kitchen_sink-19996c416e742eb157beb00f7d71d30b6df64dea.md) | 2,053 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/19996c416e742eb157beb00f7d71d30b6df64dea

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30808294537)

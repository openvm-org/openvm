| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/fibonacci-a630dfe19a4ad7dfa168d62bbb789e3d59b59ad9.md) | 415 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/keccak-a630dfe19a4ad7dfa168d62bbb789e3d59b59ad9.md) | 8,729 |  14,365,133 |  1,552 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/sha2_bench-a630dfe19a4ad7dfa168d62bbb789e3d59b59ad9.md) | 4,272 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/regex-a630dfe19a4ad7dfa168d62bbb789e3d59b59ad9.md) | 575 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/ecrecover-a630dfe19a4ad7dfa168d62bbb789e3d59b59ad9.md) | 217 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/pairing-a630dfe19a4ad7dfa168d62bbb789e3d59b59ad9.md) | 285 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/kitchen_sink-a630dfe19a4ad7dfa168d62bbb789e3d59b59ad9.md) | 1,936 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a630dfe19a4ad7dfa168d62bbb789e3d59b59ad9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29649727085)

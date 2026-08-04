| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3097/fibonacci-897321cac2bf1a95acee43454d6bc35238ff9cc8.md) | 469 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3097/keccak-897321cac2bf1a95acee43454d6bc35238ff9cc8.md) | 7,332 |  14,365,133 |  1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3097/sha2_bench-897321cac2bf1a95acee43454d6bc35238ff9cc8.md) | 4,103 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3097/regex-897321cac2bf1a95acee43454d6bc35238ff9cc8.md) | 664 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3097/ecrecover-897321cac2bf1a95acee43454d6bc35238ff9cc8.md) | 225 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3097/pairing-897321cac2bf1a95acee43454d6bc35238ff9cc8.md) | 230 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3097/kitchen_sink-897321cac2bf1a95acee43454d6bc35238ff9cc8.md) | 2,037 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/897321cac2bf1a95acee43454d6bc35238ff9cc8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30922472391)

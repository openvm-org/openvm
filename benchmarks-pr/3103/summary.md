| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-57bfff7eab14e0295b8b6c87538e45e33937e5eb.md) | 473 |  4,000,051 |  236 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-57bfff7eab14e0295b8b6c87538e45e33937e5eb.md) | 7,477 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-57bfff7eab14e0295b8b6c87538e45e33937e5eb.md) | 4,161 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-57bfff7eab14e0295b8b6c87538e45e33937e5eb.md) | 674 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-57bfff7eab14e0295b8b6c87538e45e33937e5eb.md) | 195 |  112,210 |  195 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-57bfff7eab14e0295b8b6c87538e45e33937e5eb.md) | 238 |  592,827 |  198 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-57bfff7eab14e0295b8b6c87538e45e33937e5eb.md) | 2,034 |  1,979,971 |  523 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/57bfff7eab14e0295b8b6c87538e45e33937e5eb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32059649086)

| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/fibonacci-c155ebdbd70b9692687fdde4e62847910c7bd506.md) | 453 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/keccak-c155ebdbd70b9692687fdde4e62847910c7bd506.md) | 7,210 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/sha2_bench-c155ebdbd70b9692687fdde4e62847910c7bd506.md) | 4,687 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/regex-c155ebdbd70b9692687fdde4e62847910c7bd506.md) | 655 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/ecrecover-c155ebdbd70b9692687fdde4e62847910c7bd506.md) | 228 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/pairing-c155ebdbd70b9692687fdde4e62847910c7bd506.md) | 307 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/kitchen_sink-c155ebdbd70b9692687fdde4e62847910c7bd506.md) | 2,639 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c155ebdbd70b9692687fdde4e62847910c7bd506

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30122044800)

| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-1c2783affa71ef335d689c9557fd36b1ffd6e4be.md) | 416 |  4,000,051 |  236 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-1c2783affa71ef335d689c9557fd36b1ffd6e4be.md) | 8,705 |  14,365,133 |  1,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-1c2783affa71ef335d689c9557fd36b1ffd6e4be.md) | 4,200 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-1c2783affa71ef335d689c9557fd36b1ffd6e4be.md) | 575 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-1c2783affa71ef335d689c9557fd36b1ffd6e4be.md) | 223 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-1c2783affa71ef335d689c9557fd36b1ffd6e4be.md) | 292 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-1c2783affa71ef335d689c9557fd36b1ffd6e4be.md) | 1,908 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1c2783affa71ef335d689c9557fd36b1ffd6e4be

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29779457970)

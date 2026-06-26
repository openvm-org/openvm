| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/fibonacci-0a3598a159305c5e50528c774c2efd9ac212f71f.md) | 1,033 |  4,000,051 |  392 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/keccak-0a3598a159305c5e50528c774c2efd9ac212f71f.md) | 15,775 |  14,365,133 |  3,024 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/sha2_bench-0a3598a159305c5e50528c774c2efd9ac212f71f.md) | 8,041 |  11,167,961 |  988 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/regex-0a3598a159305c5e50528c774c2efd9ac212f71f.md) | 1,178 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/ecrecover-0a3598a159305c5e50528c774c2efd9ac212f71f.md) | 438 |  112,210 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/pairing-0a3598a159305c5e50528c774c2efd9ac212f71f.md) | 598 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/kitchen_sink-0a3598a159305c5e50528c774c2efd9ac212f71f.md) | 3,874 |  1,979,971 |  857 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0a3598a159305c5e50528c774c2efd9ac212f71f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28268006589)

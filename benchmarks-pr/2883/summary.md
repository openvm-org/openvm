| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/fibonacci-ad9a0f86662d426dab2a9e7873dc5e976dfd0967.md) | 1,030 |  4,000,051 |  399 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/keccak-ad9a0f86662d426dab2a9e7873dc5e976dfd0967.md) | 16,380 |  14,365,133 |  3,023 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/sha2_bench-ad9a0f86662d426dab2a9e7873dc5e976dfd0967.md) | 8,230 |  11,167,961 |  1,002 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/regex-ad9a0f86662d426dab2a9e7873dc5e976dfd0967.md) | 1,235 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/ecrecover-ad9a0f86662d426dab2a9e7873dc5e976dfd0967.md) | 435 |  112,210 |  279 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/pairing-ad9a0f86662d426dab2a9e7873dc5e976dfd0967.md) | 598 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/kitchen_sink-ad9a0f86662d426dab2a9e7873dc5e976dfd0967.md) | 3,886 |  1,979,971 |  861 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ad9a0f86662d426dab2a9e7873dc5e976dfd0967

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27830044616)

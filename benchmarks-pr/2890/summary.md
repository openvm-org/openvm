| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/fibonacci-99c63411569c9dcf7d99127dd0ac7a38d5ac1997.md) | 1,668 |  4,000,051 |  533 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/keccak-99c63411569c9dcf7d99127dd0ac7a38d5ac1997.md) | 16,322 |  14,365,133 |  3,036 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/sha2_bench-99c63411569c9dcf7d99127dd0ac7a38d5ac1997.md) | 10,503 |  11,167,961 |  1,960 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/regex-99c63411569c9dcf7d99127dd0ac7a38d5ac1997.md) | 1,532 |  4,090,656 |  418 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/ecrecover-99c63411569c9dcf7d99127dd0ac7a38d5ac1997.md) | 477 |  112,210 |  308 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/pairing-99c63411569c9dcf7d99127dd0ac7a38d5ac1997.md) | 623 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/kitchen_sink-99c63411569c9dcf7d99127dd0ac7a38d5ac1997.md) | 3,925 |  1,979,971 |  849 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/99c63411569c9dcf7d99127dd0ac7a38d5ac1997

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27559486372)

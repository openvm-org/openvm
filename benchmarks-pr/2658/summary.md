| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2658/fibonacci-a021cefc8efa3cf38edc2da679acfa09ac1ad1a6.md) | 3,818 |  12,000,265 |  939 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2658/keccak-a021cefc8efa3cf38edc2da679acfa09ac1ad1a6.md) | 15,865 |  1,235,218 |  2,229 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2658/regex-a021cefc8efa3cf38edc2da679acfa09ac1ad1a6.md) | 1,435 |  4,136,694 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2658/ecrecover-a021cefc8efa3cf38edc2da679acfa09ac1ad1a6.md) | 632 |  122,348 |  262 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2658/pairing-a021cefc8efa3cf38edc2da679acfa09ac1ad1a6.md) | 913 |  1,745,757 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2658/kitchen_sink-a021cefc8efa3cf38edc2da679acfa09ac1ad1a6.md) | 2,367 |  154,763 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a021cefc8efa3cf38edc2da679acfa09ac1ad1a6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23955614810)

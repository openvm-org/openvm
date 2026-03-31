| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/fibonacci-cfa133589b1d5eef4a342804c289c31f15b3181b.md) | 3,817 |  12,000,265 |  941 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/keccak-cfa133589b1d5eef4a342804c289c31f15b3181b.md) | 15,776 |  1,235,218 |  2,189 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/regex-cfa133589b1d5eef4a342804c289c31f15b3181b.md) | 1,424 |  4,136,694 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/ecrecover-cfa133589b1d5eef4a342804c289c31f15b3181b.md) | 644 |  122,348 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/pairing-cfa133589b1d5eef4a342804c289c31f15b3181b.md) | 920 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/kitchen_sink-cfa133589b1d5eef4a342804c289c31f15b3181b.md) | 2,355 |  154,763 |  403 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cfa133589b1d5eef4a342804c289c31f15b3181b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23817838562)

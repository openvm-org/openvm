| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/fibonacci-57a96d3aa88952db28680b85f82b197bb230538c.md) | 4,193 |  12,000,265 |  1,367 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/keccak-57a96d3aa88952db28680b85f82b197bb230538c.md) | 19,198 |  1,235,218 |  3,373 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/regex-57a96d3aa88952db28680b85f82b197bb230538c.md) | 1,614 |  4,136,694 |  524 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/ecrecover-57a96d3aa88952db28680b85f82b197bb230538c.md) | 656 |  122,348 |  343 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/pairing-57a96d3aa88952db28680b85f82b197bb230538c.md) | 1,061 |  1,745,757 |  348 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/kitchen_sink-57a96d3aa88952db28680b85f82b197bb230538c.md) | 3,298 |  154,763 |  724 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/57a96d3aa88952db28680b85f82b197bb230538c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23558876204)

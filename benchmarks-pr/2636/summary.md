| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/fibonacci-e5d2d0718edcdd21220a376280573620604d252d.md) | 3,825 |  12,000,265 |  945 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/keccak-e5d2d0718edcdd21220a376280573620604d252d.md) | 15,672 |  1,235,218 |  2,200 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/regex-e5d2d0718edcdd21220a376280573620604d252d.md) | 1,414 |  4,136,694 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/ecrecover-e5d2d0718edcdd21220a376280573620604d252d.md) | 634 |  122,348 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/pairing-e5d2d0718edcdd21220a376280573620604d252d.md) | 922 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/kitchen_sink-e5d2d0718edcdd21220a376280573620604d252d.md) | 2,371 |  154,763 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e5d2d0718edcdd21220a376280573620604d252d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23816956806)

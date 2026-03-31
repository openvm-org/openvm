| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2642/fibonacci-bc53ae279ff18e948d79b63a34bc0786d93f3049.md) | 3,871 |  12,000,265 |  956 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2642/keccak-bc53ae279ff18e948d79b63a34bc0786d93f3049.md) | 15,848 |  1,235,218 |  2,218 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2642/regex-bc53ae279ff18e948d79b63a34bc0786d93f3049.md) | 1,436 |  4,136,694 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2642/ecrecover-bc53ae279ff18e948d79b63a34bc0786d93f3049.md) | 633 |  122,348 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2642/pairing-bc53ae279ff18e948d79b63a34bc0786d93f3049.md) | 912 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2642/kitchen_sink-bc53ae279ff18e948d79b63a34bc0786d93f3049.md) | 2,377 |  154,763 |  416 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bc53ae279ff18e948d79b63a34bc0786d93f3049

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23822912140)

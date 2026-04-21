| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2731/fibonacci-da8130ab15d0ef7dd228159df42db62cceb3629e.md) | 3,811 |  12,000,265 |  945 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2731/keccak-da8130ab15d0ef7dd228159df42db62cceb3629e.md) | 18,562 |  18,655,329 |  3,317 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2731/sha2_bench-da8130ab15d0ef7dd228159df42db62cceb3629e.md) | 8,987 |  14,793,960 |  1,397 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2731/regex-da8130ab15d0ef7dd228159df42db62cceb3629e.md) | 1,420 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2731/ecrecover-da8130ab15d0ef7dd228159df42db62cceb3629e.md) | 641 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2731/pairing-da8130ab15d0ef7dd228159df42db62cceb3629e.md) | 909 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2731/kitchen_sink-da8130ab15d0ef7dd228159df42db62cceb3629e.md) | 2,105 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/da8130ab15d0ef7dd228159df42db62cceb3629e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24732559967)

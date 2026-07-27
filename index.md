| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-742f3bd63e95bce45f3279bac5174a2753575d9a.md) | 1,574 |  12,000,265 |  363 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-742f3bd63e95bce45f3279bac5174a2753575d9a.md) | 9,274 |  18,655,329 |  1,514 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-742f3bd63e95bce45f3279bac5174a2753575d9a.md) | 4,947 |  14,793,960 |  576 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-742f3bd63e95bce45f3279bac5174a2753575d9a.md) | 661 |  4,137,067 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-742f3bd63e95bce45f3279bac5174a2753575d9a.md) | 431 |  123,583 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-742f3bd63e95bce45f3279bac5174a2753575d9a.md) | 569 |  1,745,757 |  190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-742f3bd63e95bce45f3279bac5174a2753575d9a.md) | 2,203 |  2,579,903 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/742f3bd63e95bce45f3279bac5174a2753575d9a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30275457905)

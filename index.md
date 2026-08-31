| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-3195e0a149c23ac70495ba833881586f685fa6d5.md) | 1,667 |  12,000,265 |  368 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-3195e0a149c23ac70495ba833881586f685fa6d5.md) | 9,639 |  18,655,329 |  1,551 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-3195e0a149c23ac70495ba833881586f685fa6d5.md) | 5,276 |  14,793,960 |  596 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-3195e0a149c23ac70495ba833881586f685fa6d5.md) | 686 |  4,137,067 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-3195e0a149c23ac70495ba833881586f685fa6d5.md) | 447 |  123,583 |  192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-3195e0a149c23ac70495ba833881586f685fa6d5.md) | 583 |  1,745,757 |  198 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-3195e0a149c23ac70495ba833881586f685fa6d5.md) | 2,303 |  2,579,903 |  496 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3195e0a149c23ac70495ba833881586f685fa6d5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33410679512)

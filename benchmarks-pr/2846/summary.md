| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-e37560b5493436802ec969ff8577381fa8ed8df7.md) | 3,927 |  12,000,265 | <span style='color: green'>(-3348 [-74.6%])</span> 1,138 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-e37560b5493436802ec969ff8577381fa8ed8df7.md) | 21,779 |  18,655,329 |  4,612 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-e37560b5493436802ec969ff8577381fa8ed8df7.md) | 9,599 |  14,793,960 |  1,849 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-e37560b5493436802ec969ff8577381fa8ed8df7.md) | 1,539 |  4,137,067 | <span style='color: green'>(-11561 [-96.4%])</span> 436 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-e37560b5493436802ec969ff8577381fa8ed8df7.md) | 608 |  123,583 | <span style='color: green'>(-5569 [-95.1%])</span> 287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-e37560b5493436802ec969ff8577381fa8ed8df7.md) | 938 |  1,745,757 | <span style='color: green'>(-6072 [-95.2%])</span> 308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-e37560b5493436802ec969ff8577381fa8ed8df7.md) | 4,138 |  2,579,903 |  883 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-e37560b5493436802ec969ff8577381fa8ed8df7.md) | 1,713 |  12,000,265 |  492 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-e37560b5493436802ec969ff8577381fa8ed8df7.md) | 718 |  4,137,067 |  197 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-e37560b5493436802ec969ff8577381fa8ed8df7.md) | 366 |  123,583 |  143 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-e37560b5493436802ec969ff8577381fa8ed8df7.md) | 505 |  1,745,757 |  147 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-e37560b5493436802ec969ff8577381fa8ed8df7.md) | 2,174 |  2,579,903 |  385 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e37560b5493436802ec969ff8577381fa8ed8df7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27239467086)

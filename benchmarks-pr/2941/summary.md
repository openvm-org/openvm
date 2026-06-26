| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/fibonacci-35775c9791e1b9cace32914551613b6cf9ba4a8e.md) | 3,069 |  12,000,265 | <span style='color: green'>(-3517 [-83.8%])</span> 682 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/keccak-35775c9791e1b9cace32914551613b6cf9ba4a8e.md) | 16,304 |  18,655,329 |  3,030 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/sha2_bench-35775c9791e1b9cace32914551613b6cf9ba4a8e.md) | 9,191 |  14,793,960 |  1,121 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/regex-35775c9791e1b9cace32914551613b6cf9ba4a8e.md) | 1,180 |  4,137,067 | <span style='color: green'>(-11762 [-97.0%])</span> 359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/ecrecover-35775c9791e1b9cace32914551613b6cf9ba4a8e.md) | 624 |  123,583 | <span style='color: green'>(-5981 [-95.4%])</span> 290 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/pairing-35775c9791e1b9cace32914551613b6cf9ba4a8e.md) | 929 |  1,745,757 | <span style='color: green'>(-6349 [-95.4%])</span> 304 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/kitchen_sink-35775c9791e1b9cace32914551613b6cf9ba4a8e.md) | 4,086 |  2,579,903 |  876 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/35775c9791e1b9cace32914551613b6cf9ba4a8e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28269625913)

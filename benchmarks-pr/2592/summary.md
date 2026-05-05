| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-1764881c17158ee9056d70bf7f299972e0c6a516.md) | 3,800 |  12,000,265 |  959 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-1764881c17158ee9056d70bf7f299972e0c6a516.md) | 19,010 |  18,655,329 |  3,336 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-1764881c17158ee9056d70bf7f299972e0c6a516.md) | 9,126 |  14,793,960 |  1,411 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-1764881c17158ee9056d70bf7f299972e0c6a516.md) | 1,432 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-1764881c17158ee9056d70bf7f299972e0c6a516.md) | 638 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-1764881c17158ee9056d70bf7f299972e0c6a516.md) | 918 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-1764881c17158ee9056d70bf7f299972e0c6a516.md) | 2,047 |  2,579,903 |  437 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-1764881c17158ee9056d70bf7f299972e0c6a516.md) | 1,825 |  12,000,265 |  452 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-1764881c17158ee9056d70bf7f299972e0c6a516.md) | 851 |  4,137,067 |  193 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-1764881c17158ee9056d70bf7f299972e0c6a516.md) | 548 |  123,583 |  153 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-1764881c17158ee9056d70bf7f299972e0c6a516.md) | 652 |  1,745,757 |  153 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-1764881c17158ee9056d70bf7f299972e0c6a516.md) | 2,159 |  2,579,903 |  420 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1764881c17158ee9056d70bf7f299972e0c6a516

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25406581763)

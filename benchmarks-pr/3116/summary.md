| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3116/fibonacci-11f5459155d07c0c104c43c89055930e2b60ea00.md) | 1,562 |  12,000,265 |  364 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3116/keccak-11f5459155d07c0c104c43c89055930e2b60ea00.md) | 9,284 |  18,655,329 |  1,521 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3116/sha2_bench-11f5459155d07c0c104c43c89055930e2b60ea00.md) | 4,867 |  14,793,960 |  571 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3116/regex-11f5459155d07c0c104c43c89055930e2b60ea00.md) | 651 |  4,137,067 |  209 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3116/ecrecover-11f5459155d07c0c104c43c89055930e2b60ea00.md) | 424 |  123,583 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3116/pairing-11f5459155d07c0c104c43c89055930e2b60ea00.md) | 599 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3116/kitchen_sink-11f5459155d07c0c104c43c89055930e2b60ea00.md) | 2,181 |  2,579,903 |  475 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3116/fibonacci_e2e-11f5459155d07c0c104c43c89055930e2b60ea00.md) | 1,608 |  12,000,265 |  345 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3116/regex_e2e-11f5459155d07c0c104c43c89055930e2b60ea00.md) | 819 |  4,137,067 |  204 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3116/ecrecover_e2e-11f5459155d07c0c104c43c89055930e2b60ea00.md) | 495 |  123,583 |  174 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3116/pairing_e2e-11f5459155d07c0c104c43c89055930e2b60ea00.md) | 644 |  1,745,757 |  179 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3116/kitchen_sink_e2e-11f5459155d07c0c104c43c89055930e2b60ea00.md) | 2,443 |  2,579,903 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/11f5459155d07c0c104c43c89055930e2b60ea00

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31423493323)

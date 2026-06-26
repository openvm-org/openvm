| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-7f29dc04d744ce6fb032e1e4d8cfe875421552fa.md) | 3,062 |  12,000,265 |  676 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-7f29dc04d744ce6fb032e1e4d8cfe875421552fa.md) | 16,524 |  18,655,329 |  3,067 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-7f29dc04d744ce6fb032e1e4d8cfe875421552fa.md) | 9,131 |  14,793,960 |  1,132 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-7f29dc04d744ce6fb032e1e4d8cfe875421552fa.md) | 1,161 |  4,137,067 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-7f29dc04d744ce6fb032e1e4d8cfe875421552fa.md) | 596 |  123,583 |  278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-7f29dc04d744ce6fb032e1e4d8cfe875421552fa.md) | 945 |  1,745,757 |  309 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-7f29dc04d744ce6fb032e1e4d8cfe875421552fa.md) | 4,111 |  2,579,903 |  883 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7f29dc04d744ce6fb032e1e4d8cfe875421552fa

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28266495522)

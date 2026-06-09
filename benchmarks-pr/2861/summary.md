| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/fibonacci-479516f627771441c838bfa02edf23c6fb1e28ed.md) | 3,976 |  12,000,265 |  1,151 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/keccak-479516f627771441c838bfa02edf23c6fb1e28ed.md) | 21,729 |  18,655,329 |  4,594 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/sha2_bench-479516f627771441c838bfa02edf23c6fb1e28ed.md) | 9,563 |  14,793,960 |  1,841 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/regex-479516f627771441c838bfa02edf23c6fb1e28ed.md) | 1,524 |  4,137,067 |  433 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/ecrecover-479516f627771441c838bfa02edf23c6fb1e28ed.md) | 611 |  123,583 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/pairing-479516f627771441c838bfa02edf23c6fb1e28ed.md) | 939 |  1,745,757 |  303 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/kitchen_sink-479516f627771441c838bfa02edf23c6fb1e28ed.md) | 4,105 |  2,579,903 |  876 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/479516f627771441c838bfa02edf23c6fb1e28ed

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27188725418)

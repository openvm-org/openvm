| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/fibonacci-09b0ea27084edfc53e622803cc8dd44cd1e8088b.md) | 3,795 |  12,000,265 |  953 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/keccak-09b0ea27084edfc53e622803cc8dd44cd1e8088b.md) | 18,963 |  18,655,329 |  3,341 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/sha2_bench-09b0ea27084edfc53e622803cc8dd44cd1e8088b.md) | 9,070 |  14,793,960 |  1,396 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/regex-09b0ea27084edfc53e622803cc8dd44cd1e8088b.md) | 1,447 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/ecrecover-09b0ea27084edfc53e622803cc8dd44cd1e8088b.md) | 642 |  123,583 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/pairing-09b0ea27084edfc53e622803cc8dd44cd1e8088b.md) | 909 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/kitchen_sink-09b0ea27084edfc53e622803cc8dd44cd1e8088b.md) | 2,040 |  2,579,903 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/09b0ea27084edfc53e622803cc8dd44cd1e8088b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25755420615)

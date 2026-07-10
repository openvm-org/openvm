| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2995/fibonacci-a887cda06d438cc694d1152b2187df06d1517593.md) | 3,146 |  12,000,265 |  685 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2995/keccak-a887cda06d438cc694d1152b2187df06d1517593.md) | 16,888 |  18,655,329 |  3,074 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2995/sha2_bench-a887cda06d438cc694d1152b2187df06d1517593.md) | 9,560 |  14,793,960 |  1,130 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2995/regex-a887cda06d438cc694d1152b2187df06d1517593.md) | 1,242 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2995/ecrecover-a887cda06d438cc694d1152b2187df06d1517593.md) | 550 |  123,583 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2995/pairing-a887cda06d438cc694d1152b2187df06d1517593.md) | 885 |  1,745,757 |  310 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2995/kitchen_sink-a887cda06d438cc694d1152b2187df06d1517593.md) | 4,593 |  2,579,903 |  896 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a887cda06d438cc694d1152b2187df06d1517593

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29062638048)

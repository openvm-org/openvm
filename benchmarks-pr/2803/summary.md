| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/fibonacci-489a4d71e4ca291296772e52b81aa6d4a4f2feaa.md) | 3,732 |  12,000,265 |  911 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/keccak-489a4d71e4ca291296772e52b81aa6d4a4f2feaa.md) | 18,603 |  18,655,329 |  3,285 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/sha2_bench-489a4d71e4ca291296772e52b81aa6d4a4f2feaa.md) | 10,135 |  14,793,960 |  1,452 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/regex-489a4d71e4ca291296772e52b81aa6d4a4f2feaa.md) | 1,425 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/ecrecover-489a4d71e4ca291296772e52b81aa6d4a4f2feaa.md) | 598 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/pairing-489a4d71e4ca291296772e52b81aa6d4a4f2feaa.md) | 891 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/kitchen_sink-489a4d71e4ca291296772e52b81aa6d4a4f2feaa.md) | 1,889 |  2,579,903 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/489a4d71e4ca291296772e52b81aa6d4a4f2feaa

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26472939424)

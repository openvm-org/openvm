| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/fibonacci-6d6ae30f239fb3106e50635d53f676eb5fe2d1cd.md) | 3,056 |  12,000,265 |  671 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/keccak-6d6ae30f239fb3106e50635d53f676eb5fe2d1cd.md) | 16,436 |  18,655,329 |  3,044 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/sha2_bench-6d6ae30f239fb3106e50635d53f676eb5fe2d1cd.md) | 9,092 |  14,793,960 |  1,115 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/regex-6d6ae30f239fb3106e50635d53f676eb5fe2d1cd.md) | 1,205 |  4,137,067 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/ecrecover-6d6ae30f239fb3106e50635d53f676eb5fe2d1cd.md) | 610 |  123,583 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/pairing-6d6ae30f239fb3106e50635d53f676eb5fe2d1cd.md) | 938 |  1,745,757 |  308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/kitchen_sink-6d6ae30f239fb3106e50635d53f676eb5fe2d1cd.md) | 4,145 |  2,579,903 |  885 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6d6ae30f239fb3106e50635d53f676eb5fe2d1cd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28545640263)

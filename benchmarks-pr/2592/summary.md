| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-9b5c9d6a3b05792faf71fc9eec5fd23e906870da.md) | 3,801 |  12,000,265 |  957 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-9b5c9d6a3b05792faf71fc9eec5fd23e906870da.md) | 18,447 |  18,655,329 |  3,309 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-9b5c9d6a3b05792faf71fc9eec5fd23e906870da.md) | 1,418 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-9b5c9d6a3b05792faf71fc9eec5fd23e906870da.md) | 646 |  123,583 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-9b5c9d6a3b05792faf71fc9eec5fd23e906870da.md) | 907 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-9b5c9d6a3b05792faf71fc9eec5fd23e906870da.md) | 2,163 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9b5c9d6a3b05792faf71fc9eec5fd23e906870da

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24200840514)

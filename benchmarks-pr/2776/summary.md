| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/fibonacci-6162841c36e8470d8d4218e4d8f074c742b22226.md) | 3,868 |  12,000,265 |  974 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/keccak-6162841c36e8470d8d4218e4d8f074c742b22226.md) | 19,250 |  18,655,329 |  3,377 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/sha2_bench-6162841c36e8470d8d4218e4d8f074c742b22226.md) | 9,165 |  14,793,960 |  1,408 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/regex-6162841c36e8470d8d4218e4d8f074c742b22226.md) | 1,419 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/ecrecover-6162841c36e8470d8d4218e4d8f074c742b22226.md) | 646 |  123,583 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/pairing-6162841c36e8470d8d4218e4d8f074c742b22226.md) | 913 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/kitchen_sink-6162841c36e8470d8d4218e4d8f074c742b22226.md) | 2,027 |  2,579,903 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6162841c36e8470d8d4218e4d8f074c742b22226

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25821930177)

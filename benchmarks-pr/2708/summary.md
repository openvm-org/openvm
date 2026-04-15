| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2708/fibonacci-43596172176f02fa3edf87d8e51c8c846113b431.md) | 3,864 |  12,000,265 |  960 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2708/keccak-43596172176f02fa3edf87d8e51c8c846113b431.md) | 18,498 |  18,655,329 |  3,296 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2708/sha2_bench-43596172176f02fa3edf87d8e51c8c846113b431.md) | 8,899 |  14,793,960 |  1,383 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2708/regex-43596172176f02fa3edf87d8e51c8c846113b431.md) | 1,427 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2708/ecrecover-43596172176f02fa3edf87d8e51c8c846113b431.md) | 650 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2708/pairing-43596172176f02fa3edf87d8e51c8c846113b431.md) | 899 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2708/kitchen_sink-43596172176f02fa3edf87d8e51c8c846113b431.md) | 2,083 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/43596172176f02fa3edf87d8e51c8c846113b431

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24468684867)

| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/fibonacci-e4b759601674b639684617cfe315118074d26154.md) | 3,665 |  12,000,265 |  911 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/keccak-e4b759601674b639684617cfe315118074d26154.md) | 17,993 |  18,655,329 |  3,289 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/sha2_bench-e4b759601674b639684617cfe315118074d26154.md) | 10,098 |  14,793,960 |  1,474 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/regex-e4b759601674b639684617cfe315118074d26154.md) | 1,384 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/ecrecover-e4b759601674b639684617cfe315118074d26154.md) | 603 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/pairing-e4b759601674b639684617cfe315118074d26154.md) | 884 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/kitchen_sink-e4b759601674b639684617cfe315118074d26154.md) | 3,866 |  2,579,903 |  954 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e4b759601674b639684617cfe315118074d26154

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27214695345)

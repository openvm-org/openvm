| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2847/fibonacci-707645e936e53fb492ef4c3e0a2b9aae667613d8.md) | 3,770 |  12,000,265 |  935 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2847/keccak-707645e936e53fb492ef4c3e0a2b9aae667613d8.md) | 18,169 |  18,655,329 |  3,299 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2847/sha2_bench-707645e936e53fb492ef4c3e0a2b9aae667613d8.md) | 9,976 |  14,793,960 |  1,467 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2847/regex-707645e936e53fb492ef4c3e0a2b9aae667613d8.md) | 1,394 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2847/ecrecover-707645e936e53fb492ef4c3e0a2b9aae667613d8.md) | 600 |  123,583 |  247 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2847/pairing-707645e936e53fb492ef4c3e0a2b9aae667613d8.md) | 881 |  1,745,757 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2847/kitchen_sink-707645e936e53fb492ef4c3e0a2b9aae667613d8.md) | 3,829 |  2,579,903 |  945 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/707645e936e53fb492ef4c3e0a2b9aae667613d8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27039374846)

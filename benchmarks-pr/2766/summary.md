| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/fibonacci-572ce4a8690971dbe7c2f516e880e7504df4bf50.md) | 3,822 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/keccak-572ce4a8690971dbe7c2f516e880e7504df4bf50.md) | 18,673 |  18,655,329 |  3,328 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/sha2_bench-572ce4a8690971dbe7c2f516e880e7504df4bf50.md) | 8,912 |  14,793,960 |  1,389 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/regex-572ce4a8690971dbe7c2f516e880e7504df4bf50.md) | 1,438 |  4,137,067 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/ecrecover-572ce4a8690971dbe7c2f516e880e7504df4bf50.md) | 633 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/pairing-572ce4a8690971dbe7c2f516e880e7504df4bf50.md) | 901 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/kitchen_sink-572ce4a8690971dbe7c2f516e880e7504df4bf50.md) | 2,094 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/572ce4a8690971dbe7c2f516e880e7504df4bf50

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25197678271)

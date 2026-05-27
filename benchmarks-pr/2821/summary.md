| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2821/fibonacci-743245e1cdd9985aa621b20be619640b864c14f3.md) | 3,746 |  12,000,265 |  915 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2821/keccak-743245e1cdd9985aa621b20be619640b864c14f3.md) | 18,725 |  18,655,329 |  3,297 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2821/sha2_bench-743245e1cdd9985aa621b20be619640b864c14f3.md) | 10,356 |  14,793,960 |  1,486 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2821/regex-743245e1cdd9985aa621b20be619640b864c14f3.md) | 1,423 |  4,137,067 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2821/ecrecover-743245e1cdd9985aa621b20be619640b864c14f3.md) | 600 |  123,583 |  248 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2821/pairing-743245e1cdd9985aa621b20be619640b864c14f3.md) | 899 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2821/kitchen_sink-743245e1cdd9985aa621b20be619640b864c14f3.md) | 1,894 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/743245e1cdd9985aa621b20be619640b864c14f3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26543899337)

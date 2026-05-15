| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/fibonacci-80ced2b6064bd5ee6572313ef56a0fc463d91acb.md) | 3,745 |  12,000,265 |  917 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/keccak-80ced2b6064bd5ee6572313ef56a0fc463d91acb.md) | 18,726 |  18,655,329 |  3,302 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/sha2_bench-80ced2b6064bd5ee6572313ef56a0fc463d91acb.md) | 10,243 |  14,793,960 |  1,472 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/regex-80ced2b6064bd5ee6572313ef56a0fc463d91acb.md) | 1,404 |  4,137,067 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/ecrecover-80ced2b6064bd5ee6572313ef56a0fc463d91acb.md) | 598 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/pairing-80ced2b6064bd5ee6572313ef56a0fc463d91acb.md) | 900 |  1,745,757 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/kitchen_sink-80ced2b6064bd5ee6572313ef56a0fc463d91acb.md) | 1,889 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/80ced2b6064bd5ee6572313ef56a0fc463d91acb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25935947752)

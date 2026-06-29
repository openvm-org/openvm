| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2948/fibonacci-deaab4585f05e00209981c83fece7894791716cb.md) | 3,055 |  12,000,265 |  671 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2948/keccak-deaab4585f05e00209981c83fece7894791716cb.md) | 16,381 |  18,655,329 |  3,046 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2948/sha2_bench-deaab4585f05e00209981c83fece7894791716cb.md) | 9,152 |  14,793,960 |  1,120 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2948/regex-deaab4585f05e00209981c83fece7894791716cb.md) | 1,161 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2948/ecrecover-deaab4585f05e00209981c83fece7894791716cb.md) | 600 |  123,583 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2948/pairing-deaab4585f05e00209981c83fece7894791716cb.md) | 944 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2948/kitchen_sink-deaab4585f05e00209981c83fece7894791716cb.md) | 4,085 |  2,579,903 |  871 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/deaab4585f05e00209981c83fece7894791716cb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28406224499)

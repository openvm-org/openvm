| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2702/fibonacci-d74d19ea5434e810799e52c83e5a7388d119dbd6.md) | 3,830 |  12,000,265 |  947 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2702/keccak-d74d19ea5434e810799e52c83e5a7388d119dbd6.md) | 18,654 |  18,655,329 |  3,311 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2702/sha2_bench-d74d19ea5434e810799e52c83e5a7388d119dbd6.md) | 9,844 |  14,793,960 |  1,396 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2702/regex-d74d19ea5434e810799e52c83e5a7388d119dbd6.md) | 1,409 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2702/ecrecover-d74d19ea5434e810799e52c83e5a7388d119dbd6.md) | 642 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2702/pairing-d74d19ea5434e810799e52c83e5a7388d119dbd6.md) | 902 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2702/kitchen_sink-d74d19ea5434e810799e52c83e5a7388d119dbd6.md) | 2,151 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d74d19ea5434e810799e52c83e5a7388d119dbd6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24356446089)

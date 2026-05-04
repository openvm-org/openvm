| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2773/fibonacci-2cfa08595597a23cf98e1ed24d6ce0deba25d46f.md) | 3,827 |  12,000,265 |  954 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2773/keccak-2cfa08595597a23cf98e1ed24d6ce0deba25d46f.md) | 18,547 |  18,655,329 |  3,309 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2773/sha2_bench-2cfa08595597a23cf98e1ed24d6ce0deba25d46f.md) | 8,964 |  14,793,960 |  1,391 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2773/regex-2cfa08595597a23cf98e1ed24d6ce0deba25d46f.md) | 1,419 |  4,137,067 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2773/ecrecover-2cfa08595597a23cf98e1ed24d6ce0deba25d46f.md) | 639 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2773/pairing-2cfa08595597a23cf98e1ed24d6ce0deba25d46f.md) | 898 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2773/kitchen_sink-2cfa08595597a23cf98e1ed24d6ce0deba25d46f.md) | 2,085 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2cfa08595597a23cf98e1ed24d6ce0deba25d46f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25339585005)

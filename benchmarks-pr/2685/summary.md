| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/fibonacci-e1a9e02b179a94b75189d0617a383d57b0fb8da9.md) | 3,815 |  12,000,265 |  947 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/keccak-e1a9e02b179a94b75189d0617a383d57b0fb8da9.md) | 18,691 |  18,655,329 |  3,361 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/regex-e1a9e02b179a94b75189d0617a383d57b0fb8da9.md) | 1,417 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/ecrecover-e1a9e02b179a94b75189d0617a383d57b0fb8da9.md) | 644 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/pairing-e1a9e02b179a94b75189d0617a383d57b0fb8da9.md) | 901 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/kitchen_sink-e1a9e02b179a94b75189d0617a383d57b0fb8da9.md) | 2,159 |  2,579,903 |  440 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e1a9e02b179a94b75189d0617a383d57b0fb8da9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24242733158)

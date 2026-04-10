| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/fibonacci-15b5773561d9f23fdc4058c0b9dbb73d9ba4e350.md) | 3,832 |  12,000,265 |  958 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/keccak-15b5773561d9f23fdc4058c0b9dbb73d9ba4e350.md) | 18,585 |  18,655,329 |  3,321 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/regex-15b5773561d9f23fdc4058c0b9dbb73d9ba4e350.md) | 1,424 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/ecrecover-15b5773561d9f23fdc4058c0b9dbb73d9ba4e350.md) | 642 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/pairing-15b5773561d9f23fdc4058c0b9dbb73d9ba4e350.md) | 899 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/kitchen_sink-15b5773561d9f23fdc4058c0b9dbb73d9ba4e350.md) | 2,140 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/15b5773561d9f23fdc4058c0b9dbb73d9ba4e350

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24244701432)

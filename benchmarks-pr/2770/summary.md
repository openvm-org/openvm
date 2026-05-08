| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/fibonacci-d29b49a1b0ae6bb79d4839e876ae443dc774cb16.md) | 3,845 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/keccak-d29b49a1b0ae6bb79d4839e876ae443dc774cb16.md) | 18,493 |  18,655,329 |  3,310 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/sha2_bench-d29b49a1b0ae6bb79d4839e876ae443dc774cb16.md) | 9,025 |  14,793,960 |  1,408 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/regex-d29b49a1b0ae6bb79d4839e876ae443dc774cb16.md) | 1,416 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/ecrecover-d29b49a1b0ae6bb79d4839e876ae443dc774cb16.md) | 638 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/pairing-d29b49a1b0ae6bb79d4839e876ae443dc774cb16.md) | 893 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/kitchen_sink-d29b49a1b0ae6bb79d4839e876ae443dc774cb16.md) | 2,089 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d29b49a1b0ae6bb79d4839e876ae443dc774cb16

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25577257126)

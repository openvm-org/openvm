| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2946/fibonacci-85eb7307dccd7feac40bca283cda050fbd8bb9f9.md) | 3,028 |  12,000,265 |  667 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2946/keccak-85eb7307dccd7feac40bca283cda050fbd8bb9f9.md) | 16,479 |  18,655,329 |  3,057 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2946/sha2_bench-85eb7307dccd7feac40bca283cda050fbd8bb9f9.md) | 9,196 |  14,793,960 |  1,116 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2946/regex-85eb7307dccd7feac40bca283cda050fbd8bb9f9.md) | 1,160 |  4,137,067 |  348 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2946/ecrecover-85eb7307dccd7feac40bca283cda050fbd8bb9f9.md) | 602 |  123,583 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2946/pairing-85eb7307dccd7feac40bca283cda050fbd8bb9f9.md) | 959 |  1,745,757 |  318 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2946/kitchen_sink-85eb7307dccd7feac40bca283cda050fbd8bb9f9.md) | 4,126 |  2,579,903 |  881 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/85eb7307dccd7feac40bca283cda050fbd8bb9f9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28399011187)

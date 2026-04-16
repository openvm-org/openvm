| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/fibonacci-fdf97881b8ef4c0fed74e14b0f947381eefa471e.md) | 3,830 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/keccak-fdf97881b8ef4c0fed74e14b0f947381eefa471e.md) | 18,698 |  18,655,329 |  3,312 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/sha2_bench-fdf97881b8ef4c0fed74e14b0f947381eefa471e.md) | 9,090 |  14,793,960 |  1,404 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/regex-fdf97881b8ef4c0fed74e14b0f947381eefa471e.md) | 1,412 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/ecrecover-fdf97881b8ef4c0fed74e14b0f947381eefa471e.md) | 652 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/pairing-fdf97881b8ef4c0fed74e14b0f947381eefa471e.md) | 908 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/kitchen_sink-fdf97881b8ef4c0fed74e14b0f947381eefa471e.md) | 2,086 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fdf97881b8ef4c0fed74e14b0f947381eefa471e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24539023409)

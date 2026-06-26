| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/fibonacci-5bf5e9773a7df646fcbd21f0deb3019bceb97273.md) | 3,110 |  12,000,265 |  681 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/keccak-5bf5e9773a7df646fcbd21f0deb3019bceb97273.md) | 16,375 |  18,655,329 |  3,037 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/sha2_bench-5bf5e9773a7df646fcbd21f0deb3019bceb97273.md) | 9,257 |  14,793,960 |  1,129 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/regex-5bf5e9773a7df646fcbd21f0deb3019bceb97273.md) | 1,169 |  4,137,067 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/ecrecover-5bf5e9773a7df646fcbd21f0deb3019bceb97273.md) | 599 |  123,583 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/pairing-5bf5e9773a7df646fcbd21f0deb3019bceb97273.md) | 930 |  1,745,757 |  301 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/kitchen_sink-5bf5e9773a7df646fcbd21f0deb3019bceb97273.md) | 4,098 |  2,579,903 |  878 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5bf5e9773a7df646fcbd21f0deb3019bceb97273

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28268108767)

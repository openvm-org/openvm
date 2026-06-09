| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/fibonacci-af9a5fad837c882e624fb0828ae7321483f5e6fb.md) | 3,701 |  12,000,265 |  909 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/keccak-af9a5fad837c882e624fb0828ae7321483f5e6fb.md) | 18,248 |  18,655,329 |  3,319 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/sha2_bench-af9a5fad837c882e624fb0828ae7321483f5e6fb.md) | 9,985 |  14,793,960 |  1,458 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/regex-af9a5fad837c882e624fb0828ae7321483f5e6fb.md) | 1,403 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/ecrecover-af9a5fad837c882e624fb0828ae7321483f5e6fb.md) | 596 |  123,583 |  248 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/pairing-af9a5fad837c882e624fb0828ae7321483f5e6fb.md) | 883 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/kitchen_sink-af9a5fad837c882e624fb0828ae7321483f5e6fb.md) | 3,850 |  2,579,903 |  948 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/af9a5fad837c882e624fb0828ae7321483f5e6fb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27187180796)

| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/fibonacci-031c8b1e2f21f57a43221aa86544b9aa7173bb03.md) | 3,039 |  12,000,265 | <span style='color: green'>(-3528 [-84.0%])</span> 671 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/keccak-031c8b1e2f21f57a43221aa86544b9aa7173bb03.md) | 16,366 |  18,655,329 |  3,025 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/sha2_bench-031c8b1e2f21f57a43221aa86544b9aa7173bb03.md) | 9,153 |  14,793,960 |  1,121 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/regex-031c8b1e2f21f57a43221aa86544b9aa7173bb03.md) | 1,174 |  4,137,067 | <span style='color: green'>(-11766 [-97.1%])</span> 355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/ecrecover-031c8b1e2f21f57a43221aa86544b9aa7173bb03.md) | 606 |  123,583 | <span style='color: green'>(-5988 [-95.5%])</span> 283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/pairing-031c8b1e2f21f57a43221aa86544b9aa7173bb03.md) | 936 |  1,745,757 | <span style='color: green'>(-6346 [-95.4%])</span> 307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/kitchen_sink-031c8b1e2f21f57a43221aa86544b9aa7173bb03.md) | 4,103 |  2,579,903 |  878 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/fibonacci_e2e-031c8b1e2f21f57a43221aa86544b9aa7173bb03.md) | 1,524 |  12,000,265 |  291 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/regex_e2e-031c8b1e2f21f57a43221aa86544b9aa7173bb03.md) | 771 |  4,137,067 |  166 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/ecrecover_e2e-031c8b1e2f21f57a43221aa86544b9aa7173bb03.md) | 513 |  123,583 |  144 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/pairing_e2e-031c8b1e2f21f57a43221aa86544b9aa7173bb03.md) | 651 |  1,745,757 |  147 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/kitchen_sink_e2e-031c8b1e2f21f57a43221aa86544b9aa7173bb03.md) | 2,451 |  2,579,903 |  383 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/031c8b1e2f21f57a43221aa86544b9aa7173bb03

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28894565656)

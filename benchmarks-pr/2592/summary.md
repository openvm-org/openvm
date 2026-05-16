| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-307dc7e2b3ffbcceb0252c03e59d77171375fc5e.md) | 3,733 |  12,000,265 |  912 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-307dc7e2b3ffbcceb0252c03e59d77171375fc5e.md) | 18,504 |  18,655,329 |  3,257 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-307dc7e2b3ffbcceb0252c03e59d77171375fc5e.md) | 10,201 |  14,793,960 |  1,458 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-307dc7e2b3ffbcceb0252c03e59d77171375fc5e.md) | 1,382 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-307dc7e2b3ffbcceb0252c03e59d77171375fc5e.md) | 621 |  123,583 |  257 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-307dc7e2b3ffbcceb0252c03e59d77171375fc5e.md) | 876 |  1,745,757 |  253 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-307dc7e2b3ffbcceb0252c03e59d77171375fc5e.md) | 1,911 |  2,579,903 |  410 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-307dc7e2b3ffbcceb0252c03e59d77171375fc5e.md) | 1,779 |  12,000,265 |  414 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-307dc7e2b3ffbcceb0252c03e59d77171375fc5e.md) | 817 |  4,137,067 |  170 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-307dc7e2b3ffbcceb0252c03e59d77171375fc5e.md) | 513 |  123,583 |  130 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-307dc7e2b3ffbcceb0252c03e59d77171375fc5e.md) | 636 |  1,745,757 |  132 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-307dc7e2b3ffbcceb0252c03e59d77171375fc5e.md) | 2,009 |  2,579,903 |  399 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/307dc7e2b3ffbcceb0252c03e59d77171375fc5e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25949965130)

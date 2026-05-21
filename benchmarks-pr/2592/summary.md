| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-5ad62532f864a1010b0dcccc0c9d132bed4a5175.md) | 3,776 |  12,000,265 |  915 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-5ad62532f864a1010b0dcccc0c9d132bed4a5175.md) | 18,607 |  18,655,329 |  3,275 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-5ad62532f864a1010b0dcccc0c9d132bed4a5175.md) | 10,423 |  14,793,960 |  1,481 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-5ad62532f864a1010b0dcccc0c9d132bed4a5175.md) | 1,411 |  4,137,067 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-5ad62532f864a1010b0dcccc0c9d132bed4a5175.md) | 606 |  123,583 |  256 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-5ad62532f864a1010b0dcccc0c9d132bed4a5175.md) | 881 |  1,745,757 |  259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-5ad62532f864a1010b0dcccc0c9d132bed4a5175.md) | 1,896 |  2,579,903 |  409 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-5ad62532f864a1010b0dcccc0c9d132bed4a5175.md) | 1,636 |  12,000,265 |  411 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-5ad62532f864a1010b0dcccc0c9d132bed4a5175.md) | 679 |  4,137,067 |  170 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-5ad62532f864a1010b0dcccc0c9d132bed4a5175.md) | 361 |  123,583 |  131 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-5ad62532f864a1010b0dcccc0c9d132bed4a5175.md) | 483 |  1,745,757 |  131 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-5ad62532f864a1010b0dcccc0c9d132bed4a5175.md) | 1,875 |  2,579,903 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5ad62532f864a1010b0dcccc0c9d132bed4a5175

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26242024942)

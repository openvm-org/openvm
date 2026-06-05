| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-2995a72183937d2e96f96e7c42a7ccfa1c650c07.md) | 1,544 |  4,000,051 |  436 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-2995a72183937d2e96f96e7c42a7ccfa1c650c07.md) | 13,691 |  14,365,133 |  2,388 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-2995a72183937d2e96f96e7c42a7ccfa1c650c07.md) | 8,974 |  11,167,961 |  1,418 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-2995a72183937d2e96f96e7c42a7ccfa1c650c07.md) | 1,559 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-2995a72183937d2e96f96e7c42a7ccfa1c650c07.md) | 478 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-2995a72183937d2e96f96e7c42a7ccfa1c650c07.md) | 591 |  592,827 |  254 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-2995a72183937d2e96f96e7c42a7ccfa1c650c07.md) | 3,777 |  1,979,971 |  937 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci_e2e-2995a72183937d2e96f96e7c42a7ccfa1c650c07.md) | 806 |  4,000,051 |  198 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex_e2e-2995a72183937d2e96f96e7c42a7ccfa1c650c07.md) | 899 |  4,090,656 |  173 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover_e2e-2995a72183937d2e96f96e7c42a7ccfa1c650c07.md) | 329 |  112,210 |  135 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing_e2e-2995a72183937d2e96f96e7c42a7ccfa1c650c07.md) | 396 |  592,827 |  127 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink_e2e-2995a72183937d2e96f96e7c42a7ccfa1c650c07.md) | 2,047 |  1,979,971 |  397 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2995a72183937d2e96f96e7c42a7ccfa1c650c07

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27018554888)

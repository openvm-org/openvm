| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/fibonacci-f60795ac29d172302f718ac73e75424928822365.md) | 1,376 |  4,000,051 |  528 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/keccak-f60795ac29d172302f718ac73e75424928822365.md) | 16,381 |  14,365,133 |  3,017 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/sha2_bench-f60795ac29d172302f718ac73e75424928822365.md) | 8,703 |  11,167,961 |  1,139 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/regex-f60795ac29d172302f718ac73e75424928822365.md) | 1,490 |  4,090,656 |  432 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/ecrecover-f60795ac29d172302f718ac73e75424928822365.md) | 475 |  112,210 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/pairing-f60795ac29d172302f718ac73e75424928822365.md) | 632 |  592,827 |  307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/kitchen_sink-f60795ac29d172302f718ac73e75424928822365.md) | 3,917 |  1,979,971 |  860 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f60795ac29d172302f718ac73e75424928822365

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27434940296)

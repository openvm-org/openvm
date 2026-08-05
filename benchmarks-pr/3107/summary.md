| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/fibonacci-d2a6dc392f1a4be9fa4578eba92af57bcbccc9c7.md) | 462 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/keccak-d2a6dc392f1a4be9fa4578eba92af57bcbccc9c7.md) | 7,294 |  14,365,133 |  1,510 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/sha2_bench-d2a6dc392f1a4be9fa4578eba92af57bcbccc9c7.md) | 4,132 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/regex-d2a6dc392f1a4be9fa4578eba92af57bcbccc9c7.md) | 659 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/ecrecover-d2a6dc392f1a4be9fa4578eba92af57bcbccc9c7.md) | 228 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/pairing-d2a6dc392f1a4be9fa4578eba92af57bcbccc9c7.md) | 234 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/kitchen_sink-d2a6dc392f1a4be9fa4578eba92af57bcbccc9c7.md) | 2,051 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d2a6dc392f1a4be9fa4578eba92af57bcbccc9c7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31047864098)

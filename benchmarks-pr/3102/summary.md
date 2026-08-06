| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/fibonacci-ff42742eb75f57e95685926e7c32e18da2c7c3b5.md) | 482 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/keccak-ff42742eb75f57e95685926e7c32e18da2c7c3b5.md) | 7,384 |  14,365,133 |  1,536 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/sha2_bench-ff42742eb75f57e95685926e7c32e18da2c7c3b5.md) | 4,118 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/regex-ff42742eb75f57e95685926e7c32e18da2c7c3b5.md) | 659 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/ecrecover-ff42742eb75f57e95685926e7c32e18da2c7c3b5.md) | 222 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/pairing-ff42742eb75f57e95685926e7c32e18da2c7c3b5.md) | 236 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/kitchen_sink-ff42742eb75f57e95685926e7c32e18da2c7c3b5.md) | 2,036 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ff42742eb75f57e95685926e7c32e18da2c7c3b5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31081948149)

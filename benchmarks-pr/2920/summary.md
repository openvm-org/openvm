| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/fibonacci-8c93f75242f0de9d08c8bf94ad0a0e5b92c54db0.md) | 1,035 |  4,000,051 |  393 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/keccak-8c93f75242f0de9d08c8bf94ad0a0e5b92c54db0.md) | 15,887 |  14,365,133 |  3,044 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/sha2_bench-8c93f75242f0de9d08c8bf94ad0a0e5b92c54db0.md) | 8,081 |  11,167,961 |  991 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/regex-8c93f75242f0de9d08c8bf94ad0a0e5b92c54db0.md) | 1,168 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/ecrecover-8c93f75242f0de9d08c8bf94ad0a0e5b92c54db0.md) | 434 |  112,210 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/pairing-8c93f75242f0de9d08c8bf94ad0a0e5b92c54db0.md) | 593 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/kitchen_sink-8c93f75242f0de9d08c8bf94ad0a0e5b92c54db0.md) | 3,897 |  1,979,971 |  870 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8c93f75242f0de9d08c8bf94ad0a0e5b92c54db0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28249065748)

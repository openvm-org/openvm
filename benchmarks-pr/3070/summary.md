| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/fibonacci-71b32e528f66209c7ec1d18082c62c1bb8865ce0.md) | 469 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/keccak-71b32e528f66209c7ec1d18082c62c1bb8865ce0.md) | 7,331 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/sha2_bench-71b32e528f66209c7ec1d18082c62c1bb8865ce0.md) | 4,737 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/regex-71b32e528f66209c7ec1d18082c62c1bb8865ce0.md) | 669 |  4,090,656 |  222 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/ecrecover-71b32e528f66209c7ec1d18082c62c1bb8865ce0.md) | 227 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/pairing-71b32e528f66209c7ec1d18082c62c1bb8865ce0.md) | 322 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/kitchen_sink-71b32e528f66209c7ec1d18082c62c1bb8865ce0.md) | 2,672 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/71b32e528f66209c7ec1d18082c62c1bb8865ce0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30151361873)

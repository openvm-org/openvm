| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-b25bc649bdbe7c2c8c100b4a5374b96a4f88db0a.md) | 458 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-b25bc649bdbe7c2c8c100b4a5374b96a4f88db0a.md) | 7,445 |  14,365,133 |  1,530 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-b25bc649bdbe7c2c8c100b4a5374b96a4f88db0a.md) | 4,174 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-b25bc649bdbe7c2c8c100b4a5374b96a4f88db0a.md) | 664 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-b25bc649bdbe7c2c8c100b4a5374b96a4f88db0a.md) | 197 |  112,210 |  198 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-b25bc649bdbe7c2c8c100b4a5374b96a4f88db0a.md) | 238 |  592,827 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-b25bc649bdbe7c2c8c100b4a5374b96a4f88db0a.md) | 2,022 |  1,979,971 |  523 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b25bc649bdbe7c2c8c100b4a5374b96a4f88db0a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31672569211)

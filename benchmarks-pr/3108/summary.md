| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-32c4efa9daeb93eeaa5c4218a0417763d863c3c8.md) | 456 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-32c4efa9daeb93eeaa5c4218a0417763d863c3c8.md) | 7,140 |  14,365,133 |  1,579 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-32c4efa9daeb93eeaa5c4218a0417763d863c3c8.md) | 4,057 |  11,167,961 |  513 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-32c4efa9daeb93eeaa5c4218a0417763d863c3c8.md) | 711 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-32c4efa9daeb93eeaa5c4218a0417763d863c3c8.md) | 204 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-32c4efa9daeb93eeaa5c4218a0417763d863c3c8.md) | 245 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-32c4efa9daeb93eeaa5c4218a0417763d863c3c8.md) | 2,166 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/32c4efa9daeb93eeaa5c4218a0417763d863c3c8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31641263481)

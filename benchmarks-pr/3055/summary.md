| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/fibonacci-fff88fa3e3d12ce9922ac9524635810e586aec03.md) | 473 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/keccak-fff88fa3e3d12ce9922ac9524635810e586aec03.md) | 7,306 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/sha2_bench-fff88fa3e3d12ce9922ac9524635810e586aec03.md) | 4,746 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/regex-fff88fa3e3d12ce9922ac9524635810e586aec03.md) | 667 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/ecrecover-fff88fa3e3d12ce9922ac9524635810e586aec03.md) | 225 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/pairing-fff88fa3e3d12ce9922ac9524635810e586aec03.md) | 320 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/kitchen_sink-fff88fa3e3d12ce9922ac9524635810e586aec03.md) | 2,678 |  1,979,971 |  470 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fff88fa3e3d12ce9922ac9524635810e586aec03

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30129775848)

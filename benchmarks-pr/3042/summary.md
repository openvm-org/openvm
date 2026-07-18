| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/fibonacci-eaf7bdb64b86fd08f377884ddfc156f3dd7eec8c.md) | 410 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/keccak-eaf7bdb64b86fd08f377884ddfc156f3dd7eec8c.md) | 8,737 |  14,365,133 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/sha2_bench-eaf7bdb64b86fd08f377884ddfc156f3dd7eec8c.md) | 4,244 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/regex-eaf7bdb64b86fd08f377884ddfc156f3dd7eec8c.md) | 577 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/ecrecover-eaf7bdb64b86fd08f377884ddfc156f3dd7eec8c.md) | 217 |  112,210 |  179 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/pairing-eaf7bdb64b86fd08f377884ddfc156f3dd7eec8c.md) | 281 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/kitchen_sink-eaf7bdb64b86fd08f377884ddfc156f3dd7eec8c.md) | 1,942 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/eaf7bdb64b86fd08f377884ddfc156f3dd7eec8c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29656149047)

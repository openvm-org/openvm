| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-205572f35e0c6625fdd6ff9c09abc7db253ac634.md) | 1,021 |  4,000,051 |  394 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-205572f35e0c6625fdd6ff9c09abc7db253ac634.md) | 16,010 |  14,365,133 |  2,997 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-205572f35e0c6625fdd6ff9c09abc7db253ac634.md) | 8,120 |  11,167,961 |  993 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-205572f35e0c6625fdd6ff9c09abc7db253ac634.md) | 1,184 |  4,090,656 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-205572f35e0c6625fdd6ff9c09abc7db253ac634.md) | 432 |  112,210 |  278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-205572f35e0c6625fdd6ff9c09abc7db253ac634.md) | 588 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-205572f35e0c6625fdd6ff9c09abc7db253ac634.md) | 3,865 |  1,979,971 |  854 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/205572f35e0c6625fdd6ff9c09abc7db253ac634

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28191413832)

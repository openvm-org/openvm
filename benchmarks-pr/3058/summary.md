| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/fibonacci-889398bc6d06db52bd87825d30d0f1ab3ecb8d5d.md) | 471 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/keccak-889398bc6d06db52bd87825d30d0f1ab3ecb8d5d.md) | 7,359 |  14,365,133 |  1,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/sha2_bench-889398bc6d06db52bd87825d30d0f1ab3ecb8d5d.md) | 4,787 |  11,167,961 |  537 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/regex-889398bc6d06db52bd87825d30d0f1ab3ecb8d5d.md) | 672 |  4,090,656 |  222 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/ecrecover-889398bc6d06db52bd87825d30d0f1ab3ecb8d5d.md) | 228 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/pairing-889398bc6d06db52bd87825d30d0f1ab3ecb8d5d.md) | 323 |  592,827 |  191 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/kitchen_sink-889398bc6d06db52bd87825d30d0f1ab3ecb8d5d.md) | 2,652 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/889398bc6d06db52bd87825d30d0f1ab3ecb8d5d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29951463876)

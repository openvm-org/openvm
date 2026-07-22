| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/fibonacci-33349ff5e6acfc3449cfc25d394e9e29534ac8fd.md) | 466 |  4,000,051 |  237 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/keccak-33349ff5e6acfc3449cfc25d394e9e29534ac8fd.md) | 7,322 |  14,365,133 |  1,543 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/sha2_bench-33349ff5e6acfc3449cfc25d394e9e29534ac8fd.md) | 4,752 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/regex-33349ff5e6acfc3449cfc25d394e9e29534ac8fd.md) | 672 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/ecrecover-33349ff5e6acfc3449cfc25d394e9e29534ac8fd.md) | 230 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/pairing-33349ff5e6acfc3449cfc25d394e9e29534ac8fd.md) | 314 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/kitchen_sink-33349ff5e6acfc3449cfc25d394e9e29534ac8fd.md) | 2,688 |  1,979,971 |  478 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/33349ff5e6acfc3449cfc25d394e9e29534ac8fd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29942764447)

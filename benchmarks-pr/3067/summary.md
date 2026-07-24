| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3067/fibonacci-f881f9a743a7bdb382bd022d1c3ee8c10e17018a.md) | 475 |  4,000,051 |  244 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3067/keccak-f881f9a743a7bdb382bd022d1c3ee8c10e17018a.md) | 7,261 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3067/sha2_bench-f881f9a743a7bdb382bd022d1c3ee8c10e17018a.md) | 4,767 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3067/regex-f881f9a743a7bdb382bd022d1c3ee8c10e17018a.md) | 675 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3067/ecrecover-f881f9a743a7bdb382bd022d1c3ee8c10e17018a.md) | 229 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3067/pairing-f881f9a743a7bdb382bd022d1c3ee8c10e17018a.md) | 314 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3067/kitchen_sink-f881f9a743a7bdb382bd022d1c3ee8c10e17018a.md) | 2,658 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f881f9a743a7bdb382bd022d1c3ee8c10e17018a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30103778200)

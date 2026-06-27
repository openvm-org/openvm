| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-48d1f207b3e691951ffcb4a33009757836148b44.md) | 1,037 |  4,000,051 |  393 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-48d1f207b3e691951ffcb4a33009757836148b44.md) | 15,745 |  14,365,133 |  3,030 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-48d1f207b3e691951ffcb4a33009757836148b44.md) | 8,181 |  11,167,961 |  1,005 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-48d1f207b3e691951ffcb4a33009757836148b44.md) | 1,173 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-48d1f207b3e691951ffcb4a33009757836148b44.md) | 430 |  112,210 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-48d1f207b3e691951ffcb4a33009757836148b44.md) | 587 |  592,827 |  301 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-48d1f207b3e691951ffcb4a33009757836148b44.md) | 3,875 |  1,979,971 |  857 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/48d1f207b3e691951ffcb4a33009757836148b44

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28273392950)

| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2949/fibonacci-1e017986a80910140909d97a1a0120101662fd0a.md) | 1,029 |  4,000,051 |  392 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2949/keccak-1e017986a80910140909d97a1a0120101662fd0a.md) | 15,511 |  14,365,133 |  2,977 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2949/sha2_bench-1e017986a80910140909d97a1a0120101662fd0a.md) | 8,096 |  11,167,961 |  992 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2949/regex-1e017986a80910140909d97a1a0120101662fd0a.md) | 1,161 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2949/ecrecover-1e017986a80910140909d97a1a0120101662fd0a.md) | 439 |  112,210 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2949/pairing-1e017986a80910140909d97a1a0120101662fd0a.md) | 584 |  592,827 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2949/kitchen_sink-1e017986a80910140909d97a1a0120101662fd0a.md) | 3,882 |  1,979,971 |  863 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1e017986a80910140909d97a1a0120101662fd0a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28407289844)

| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/fibonacci-fe533fe8383112ba015f5524424c73ede34c9a0b.md) | 475 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/keccak-fe533fe8383112ba015f5524424c73ede34c9a0b.md) | 7,403 |  14,365,133 |  1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/sha2_bench-fe533fe8383112ba015f5524424c73ede34c9a0b.md) | 4,128 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/regex-fe533fe8383112ba015f5524424c73ede34c9a0b.md) | 667 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/ecrecover-fe533fe8383112ba015f5524424c73ede34c9a0b.md) | 227 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/pairing-fe533fe8383112ba015f5524424c73ede34c9a0b.md) | 236 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/kitchen_sink-fe533fe8383112ba015f5524424c73ede34c9a0b.md) | 2,048 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fe533fe8383112ba015f5524424c73ede34c9a0b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31436087710)

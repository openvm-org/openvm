| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/fibonacci-d20d10d946053c97e5427a503f5317bb60f95e94.md) | 442 |  4,000,051 |  225 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/keccak-d20d10d946053c97e5427a503f5317bb60f95e94.md) | 8,370 |  14,365,133 |  1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/sha2_bench-d20d10d946053c97e5427a503f5317bb60f95e94.md) | 3,952 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/regex-d20d10d946053c97e5427a503f5317bb60f95e94.md) | 576 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/ecrecover-d20d10d946053c97e5427a503f5317bb60f95e94.md) | 219 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/pairing-d20d10d946053c97e5427a503f5317bb60f95e94.md) | 275 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/kitchen_sink-d20d10d946053c97e5427a503f5317bb60f95e94.md) | 1,901 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d20d10d946053c97e5427a503f5317bb60f95e94

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29374451642)

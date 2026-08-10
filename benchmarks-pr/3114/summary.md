| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/fibonacci-8ff5b9ed2d1287494ff671094bbc2a554e85308c.md) | 483 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/keccak-8ff5b9ed2d1287494ff671094bbc2a554e85308c.md) | 7,392 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/sha2_bench-8ff5b9ed2d1287494ff671094bbc2a554e85308c.md) | 4,191 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/regex-8ff5b9ed2d1287494ff671094bbc2a554e85308c.md) | 660 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/ecrecover-8ff5b9ed2d1287494ff671094bbc2a554e85308c.md) | 225 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/pairing-8ff5b9ed2d1287494ff671094bbc2a554e85308c.md) | 237 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/kitchen_sink-8ff5b9ed2d1287494ff671094bbc2a554e85308c.md) | 2,050 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8ff5b9ed2d1287494ff671094bbc2a554e85308c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31435643705)

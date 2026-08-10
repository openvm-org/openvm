| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/fibonacci-ffb14104bd54973fdd82631e320c30a3987d8639.md) | 471 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/keccak-ffb14104bd54973fdd82631e320c30a3987d8639.md) | 7,313 |  14,365,133 |  1,504 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/sha2_bench-ffb14104bd54973fdd82631e320c30a3987d8639.md) | 4,178 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/regex-ffb14104bd54973fdd82631e320c30a3987d8639.md) | 656 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/ecrecover-ffb14104bd54973fdd82631e320c30a3987d8639.md) | 228 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/pairing-ffb14104bd54973fdd82631e320c30a3987d8639.md) | 238 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/kitchen_sink-ffb14104bd54973fdd82631e320c30a3987d8639.md) | 2,029 |  1,979,971 |  455 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ffb14104bd54973fdd82631e320c30a3987d8639

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31430738750)

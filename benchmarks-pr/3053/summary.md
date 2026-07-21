| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3053/fibonacci-b05fb80b67cefc335026ffc136c8898e36fe97e8.md) | 421 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3053/keccak-b05fb80b67cefc335026ffc136c8898e36fe97e8.md) | 8,723 |  14,365,133 |  1,552 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3053/sha2_bench-b05fb80b67cefc335026ffc136c8898e36fe97e8.md) | 4,180 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3053/regex-b05fb80b67cefc335026ffc136c8898e36fe97e8.md) | 567 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3053/ecrecover-b05fb80b67cefc335026ffc136c8898e36fe97e8.md) | 223 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3053/pairing-b05fb80b67cefc335026ffc136c8898e36fe97e8.md) | 287 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3053/kitchen_sink-b05fb80b67cefc335026ffc136c8898e36fe97e8.md) | 1,907 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b05fb80b67cefc335026ffc136c8898e36fe97e8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29846259424)

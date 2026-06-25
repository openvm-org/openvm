| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/fibonacci-f850f6fc71b30f100a6c7d9235af3fc086515c04.md) | 1,028 |  4,000,051 |  391 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/keccak-f850f6fc71b30f100a6c7d9235af3fc086515c04.md) | 15,829 |  14,365,133 |  3,042 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/sha2_bench-f850f6fc71b30f100a6c7d9235af3fc086515c04.md) | 8,116 |  11,167,961 |  994 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/regex-f850f6fc71b30f100a6c7d9235af3fc086515c04.md) | 1,212 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/ecrecover-f850f6fc71b30f100a6c7d9235af3fc086515c04.md) | 440 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/pairing-f850f6fc71b30f100a6c7d9235af3fc086515c04.md) | 592 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/kitchen_sink-f850f6fc71b30f100a6c7d9235af3fc086515c04.md) | 3,886 |  1,979,971 |  861 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f850f6fc71b30f100a6c7d9235af3fc086515c04

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28174707366)

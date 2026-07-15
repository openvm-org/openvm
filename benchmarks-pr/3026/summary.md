| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/fibonacci-82bbb5b823fd0e50d150ed1bcdc1dfe120040cbc.md) | 409 |  4,000,051 |  226 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/keccak-82bbb5b823fd0e50d150ed1bcdc1dfe120040cbc.md) | 8,476 |  14,365,133 |  1,550 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/sha2_bench-82bbb5b823fd0e50d150ed1bcdc1dfe120040cbc.md) | 3,939 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/regex-82bbb5b823fd0e50d150ed1bcdc1dfe120040cbc.md) | 569 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/ecrecover-82bbb5b823fd0e50d150ed1bcdc1dfe120040cbc.md) | 221 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/pairing-82bbb5b823fd0e50d150ed1bcdc1dfe120040cbc.md) | 275 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/kitchen_sink-82bbb5b823fd0e50d150ed1bcdc1dfe120040cbc.md) | 1,882 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/82bbb5b823fd0e50d150ed1bcdc1dfe120040cbc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29452707073)

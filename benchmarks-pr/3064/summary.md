| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3064/fibonacci-fa6a87266ddf2ef82f45bb08595b3ab973fb88b3.md) | 470 |  4,000,051 |  245 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3064/keccak-fa6a87266ddf2ef82f45bb08595b3ab973fb88b3.md) | 7,316 |  14,365,133 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3064/sha2_bench-fa6a87266ddf2ef82f45bb08595b3ab973fb88b3.md) | 4,698 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3064/regex-fa6a87266ddf2ef82f45bb08595b3ab973fb88b3.md) | 672 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3064/ecrecover-fa6a87266ddf2ef82f45bb08595b3ab973fb88b3.md) | 228 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3064/pairing-fa6a87266ddf2ef82f45bb08595b3ab973fb88b3.md) | 320 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3064/kitchen_sink-fa6a87266ddf2ef82f45bb08595b3ab973fb88b3.md) | 2,665 |  1,979,971 |  470 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fa6a87266ddf2ef82f45bb08595b3ab973fb88b3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29948873915)

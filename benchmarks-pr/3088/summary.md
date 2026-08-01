| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3088/fibonacci-fa6dec6ebef2472880c0c239c31864cb1f7494a7.md) | 481 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3088/keccak-fa6dec6ebef2472880c0c239c31864cb1f7494a7.md) | 7,427 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3088/sha2_bench-fa6dec6ebef2472880c0c239c31864cb1f7494a7.md) | 4,136 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3088/regex-fa6dec6ebef2472880c0c239c31864cb1f7494a7.md) | 659 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3088/ecrecover-fa6dec6ebef2472880c0c239c31864cb1f7494a7.md) | 236 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3088/pairing-fa6dec6ebef2472880c0c239c31864cb1f7494a7.md) | 241 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3088/kitchen_sink-fa6dec6ebef2472880c0c239c31864cb1f7494a7.md) | 2,047 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fa6dec6ebef2472880c0c239c31864cb1f7494a7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30688055486)

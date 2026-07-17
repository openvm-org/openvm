| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3034/fibonacci-5fc1c1cd8173350c4777e71c303b7682df330533.md) | 406 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3034/keccak-5fc1c1cd8173350c4777e71c303b7682df330533.md) | 8,551 |  14,365,133 |  1,517 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3034/sha2_bench-5fc1c1cd8173350c4777e71c303b7682df330533.md) | 4,234 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3034/regex-5fc1c1cd8173350c4777e71c303b7682df330533.md) | 567 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3034/ecrecover-5fc1c1cd8173350c4777e71c303b7682df330533.md) | 221 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3034/pairing-5fc1c1cd8173350c4777e71c303b7682df330533.md) | 290 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3034/kitchen_sink-5fc1c1cd8173350c4777e71c303b7682df330533.md) | 1,903 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5fc1c1cd8173350c4777e71c303b7682df330533

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29587739561)

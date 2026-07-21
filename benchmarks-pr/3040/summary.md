| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/fibonacci-6df30fd5246f66732eab948ac8de16b2c8221cfb.md) | 419 |  4,000,051 |  236 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/keccak-6df30fd5246f66732eab948ac8de16b2c8221cfb.md) | 8,616 |  14,365,133 |  1,518 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/sha2_bench-6df30fd5246f66732eab948ac8de16b2c8221cfb.md) | 4,145 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/regex-6df30fd5246f66732eab948ac8de16b2c8221cfb.md) | 569 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/ecrecover-6df30fd5246f66732eab948ac8de16b2c8221cfb.md) | 219 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/pairing-6df30fd5246f66732eab948ac8de16b2c8221cfb.md) | 289 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/kitchen_sink-6df30fd5246f66732eab948ac8de16b2c8221cfb.md) | 1,923 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6df30fd5246f66732eab948ac8de16b2c8221cfb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29818326795)

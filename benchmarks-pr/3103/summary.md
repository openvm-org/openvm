| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-2c43fb246c83da6a57b15b7eec212339b96e9798.md) | 459 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-2c43fb246c83da6a57b15b7eec212339b96e9798.md) | 7,382 |  14,365,133 |  1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-2c43fb246c83da6a57b15b7eec212339b96e9798.md) | 4,167 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-2c43fb246c83da6a57b15b7eec212339b96e9798.md) | 662 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-2c43fb246c83da6a57b15b7eec212339b96e9798.md) | 195 |  112,210 |  197 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-2c43fb246c83da6a57b15b7eec212339b96e9798.md) | 233 |  592,827 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-2c43fb246c83da6a57b15b7eec212339b96e9798.md) | 2,032 |  1,979,971 |  525 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2c43fb246c83da6a57b15b7eec212339b96e9798

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31618070275)

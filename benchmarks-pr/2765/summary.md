| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-19d5a2c5fe91e336393737e92590df5929f21ac5.md) | 1,884 |  4,000,051 |  536 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-19d5a2c5fe91e336393737e92590df5929f21ac5.md) | 13,371 |  14,365,133 |  2,190 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-19d5a2c5fe91e336393737e92590df5929f21ac5.md) | 9,563 |  11,167,961 |  1,429 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-19d5a2c5fe91e336393737e92590df5929f21ac5.md) | 1,610 |  4,090,656 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-19d5a2c5fe91e336393737e92590df5929f21ac5.md) | 637 |  112,210 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-19d5a2c5fe91e336393737e92590df5929f21ac5.md) | 765 |  592,827 |  277 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-19d5a2c5fe91e336393737e92590df5929f21ac5.md) | 2,035 |  1,979,971 |  427 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/19d5a2c5fe91e336393737e92590df5929f21ac5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25843248957)

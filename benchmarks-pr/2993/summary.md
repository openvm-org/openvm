| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-8b88d76af92ae981cdc0c27bb313045699b33425.md) | 417 |  4,000,051 |  236 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-8b88d76af92ae981cdc0c27bb313045699b33425.md) | 8,749 |  14,365,133 |  1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-8b88d76af92ae981cdc0c27bb313045699b33425.md) | 4,256 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-8b88d76af92ae981cdc0c27bb313045699b33425.md) | 567 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-8b88d76af92ae981cdc0c27bb313045699b33425.md) | 222 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-8b88d76af92ae981cdc0c27bb313045699b33425.md) | 283 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-8b88d76af92ae981cdc0c27bb313045699b33425.md) | 1,928 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8b88d76af92ae981cdc0c27bb313045699b33425

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29494726205)

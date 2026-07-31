| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-232b782c7b4cff80461942a816d1cd06776c0b50.md) | 479 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-232b782c7b4cff80461942a816d1cd06776c0b50.md) | 7,338 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-232b782c7b4cff80461942a816d1cd06776c0b50.md) | 4,144 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-232b782c7b4cff80461942a816d1cd06776c0b50.md) | 658 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-232b782c7b4cff80461942a816d1cd06776c0b50.md) | 237 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-232b782c7b4cff80461942a816d1cd06776c0b50.md) | 240 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-232b782c7b4cff80461942a816d1cd06776c0b50.md) | 2,046 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/232b782c7b4cff80461942a816d1cd06776c0b50

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30668190043)

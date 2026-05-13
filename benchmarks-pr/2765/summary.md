| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-de4ffe39ae984c9860ac54eee9b14b07a607f94f.md) | 1,887 |  4,000,051 |  532 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-de4ffe39ae984c9860ac54eee9b14b07a607f94f.md) | 13,928 |  14,365,133 |  2,289 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-de4ffe39ae984c9860ac54eee9b14b07a607f94f.md) | 9,535 |  11,167,961 |  1,419 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-de4ffe39ae984c9860ac54eee9b14b07a607f94f.md) | 1,605 |  4,090,656 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-de4ffe39ae984c9860ac54eee9b14b07a607f94f.md) | 634 |  112,210 |  289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-de4ffe39ae984c9860ac54eee9b14b07a607f94f.md) | 759 |  592,827 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-de4ffe39ae984c9860ac54eee9b14b07a607f94f.md) | 2,030 |  1,979,971 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/de4ffe39ae984c9860ac54eee9b14b07a607f94f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25826046369)

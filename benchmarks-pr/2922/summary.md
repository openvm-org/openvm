| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/fibonacci-ab73dc022b031b6a5e9da8e8bd013aa36964f552.md) | 1,032 |  4,000,051 |  392 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/keccak-ab73dc022b031b6a5e9da8e8bd013aa36964f552.md) | 15,373 |  14,365,133 |  2,997 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/sha2_bench-ab73dc022b031b6a5e9da8e8bd013aa36964f552.md) | 7,760 |  11,167,961 |  1,003 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/regex-ab73dc022b031b6a5e9da8e8bd013aa36964f552.md) | 1,028 |  4,090,656 |  297 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/ecrecover-ab73dc022b031b6a5e9da8e8bd013aa36964f552.md) | 434 |  112,210 |  278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/pairing-ab73dc022b031b6a5e9da8e8bd013aa36964f552.md) | 558 |  592,827 |  297 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/kitchen_sink-ab73dc022b031b6a5e9da8e8bd013aa36964f552.md) | 3,777 |  1,979,971 |  861 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ab73dc022b031b6a5e9da8e8bd013aa36964f552

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28259247060)

| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-8c9d578d88cec06a67a020d5552f11dd4402f4ab.md) | 479 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-8c9d578d88cec06a67a020d5552f11dd4402f4ab.md) | 7,353 |  14,365,133 |  1,517 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-8c9d578d88cec06a67a020d5552f11dd4402f4ab.md) | 4,160 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-8c9d578d88cec06a67a020d5552f11dd4402f4ab.md) | 664 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-8c9d578d88cec06a67a020d5552f11dd4402f4ab.md) | 231 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-8c9d578d88cec06a67a020d5552f11dd4402f4ab.md) | 239 |  592,827 |  190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-8c9d578d88cec06a67a020d5552f11dd4402f4ab.md) | 2,033 |  1,979,971 |  456 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8c9d578d88cec06a67a020d5552f11dd4402f4ab

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30502430601)

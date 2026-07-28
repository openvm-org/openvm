| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-d91c410525ef173d89c3bc79acf3822b4b18536f.md) | 484 |  4,000,051 |  217 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-d91c410525ef173d89c3bc79acf3822b4b18536f.md) | 7,309 |  14,365,133 |  1,474 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-d91c410525ef173d89c3bc79acf3822b4b18536f.md) | 4,128 |  11,167,961 |  507 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-d91c410525ef173d89c3bc79acf3822b4b18536f.md) | 668 |  4,090,656 |  202 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-d91c410525ef173d89c3bc79acf3822b4b18536f.md) | 214 |  112,210 |  169 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-d91c410525ef173d89c3bc79acf3822b4b18536f.md) | 229 |  592,827 |  173 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-d91c410525ef173d89c3bc79acf3822b4b18536f.md) | 2,004 |  1,979,971 |  446 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d91c410525ef173d89c3bc79acf3822b4b18536f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30374413747)

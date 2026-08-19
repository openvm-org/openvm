| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-53145cab77ab2dede45541ae2c3d5cd9612a3a46.md) | 456 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-53145cab77ab2dede45541ae2c3d5cd9612a3a46.md) | 7,238 |  14,365,133 |  1,580 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-53145cab77ab2dede45541ae2c3d5cd9612a3a46.md) | 4,110 |  11,167,961 |  515 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-53145cab77ab2dede45541ae2c3d5cd9612a3a46.md) | 715 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-53145cab77ab2dede45541ae2c3d5cd9612a3a46.md) | 207 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-53145cab77ab2dede45541ae2c3d5cd9612a3a46.md) | 242 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-53145cab77ab2dede45541ae2c3d5cd9612a3a46.md) | 2,178 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/53145cab77ab2dede45541ae2c3d5cd9612a3a46

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32292941475)

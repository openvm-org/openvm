| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/fibonacci-bb70b80579ebbc01fac31c56058e81e8718d0442.md) | 411 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/keccak-bb70b80579ebbc01fac31c56058e81e8718d0442.md) | 8,684 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/sha2_bench-bb70b80579ebbc01fac31c56058e81e8718d0442.md) | 4,204 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/regex-bb70b80579ebbc01fac31c56058e81e8718d0442.md) | 567 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/ecrecover-bb70b80579ebbc01fac31c56058e81e8718d0442.md) | 223 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/pairing-bb70b80579ebbc01fac31c56058e81e8718d0442.md) | 291 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/kitchen_sink-bb70b80579ebbc01fac31c56058e81e8718d0442.md) | 1,908 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bb70b80579ebbc01fac31c56058e81e8718d0442

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29816254976)

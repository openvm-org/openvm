| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/fibonacci-a1ebb18d378d87c225e72883422871fb9a2c9718.md) | 438 |  4,000,051 |  225 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/keccak-a1ebb18d378d87c225e72883422871fb9a2c9718.md) | 8,452 |  14,365,133 |  1,527 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/sha2_bench-a1ebb18d378d87c225e72883422871fb9a2c9718.md) | 4,011 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/regex-a1ebb18d378d87c225e72883422871fb9a2c9718.md) | 581 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/ecrecover-a1ebb18d378d87c225e72883422871fb9a2c9718.md) | 216 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/pairing-a1ebb18d378d87c225e72883422871fb9a2c9718.md) | 261 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/kitchen_sink-a1ebb18d378d87c225e72883422871fb9a2c9718.md) | 1,887 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a1ebb18d378d87c225e72883422871fb9a2c9718

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29407555174)

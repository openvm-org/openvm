| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-d2a5ee491261536992cc9fdc3290f3c30f6835bc.md) | 1,037 |  4,000,051 |  397 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-d2a5ee491261536992cc9fdc3290f3c30f6835bc.md) | 15,568 |  14,365,133 |  2,991 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-d2a5ee491261536992cc9fdc3290f3c30f6835bc.md) | 8,218 |  11,167,961 |  1,014 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-d2a5ee491261536992cc9fdc3290f3c30f6835bc.md) | 1,155 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-d2a5ee491261536992cc9fdc3290f3c30f6835bc.md) | 436 |  112,210 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-d2a5ee491261536992cc9fdc3290f3c30f6835bc.md) | 588 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-d2a5ee491261536992cc9fdc3290f3c30f6835bc.md) | 3,896 |  1,979,971 |  868 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d2a5ee491261536992cc9fdc3290f3c30f6835bc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28292799530)

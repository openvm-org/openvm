| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/fibonacci-8b0a2d8ac8d8b8f7be4bc65b494c773cdc9b5499.md) | 1,549 |  4,000,051 |  431 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/keccak-8b0a2d8ac8d8b8f7be4bc65b494c773cdc9b5499.md) | 13,865 |  14,365,133 |  2,203 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/sha2_bench-8b0a2d8ac8d8b8f7be4bc65b494c773cdc9b5499.md) | 9,151 |  11,167,961 |  1,421 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/regex-8b0a2d8ac8d8b8f7be4bc65b494c773cdc9b5499.md) | 1,465 |  4,090,656 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/ecrecover-8b0a2d8ac8d8b8f7be4bc65b494c773cdc9b5499.md) | 470 |  112,210 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/pairing-8b0a2d8ac8d8b8f7be4bc65b494c773cdc9b5499.md) | 591 |  592,827 |  256 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/kitchen_sink-8b0a2d8ac8d8b8f7be4bc65b494c773cdc9b5499.md) | 2,222 |  1,979,971 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8b0a2d8ac8d8b8f7be4bc65b494c773cdc9b5499

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26237068316)

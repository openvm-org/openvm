| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-5f5fa9086c00a34b3194c93b3531ebc1bcb2f28d.md) | 1,025 |  4,000,051 |  392 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-5f5fa9086c00a34b3194c93b3531ebc1bcb2f28d.md) | 15,806 |  14,365,133 |  3,045 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-5f5fa9086c00a34b3194c93b3531ebc1bcb2f28d.md) | 8,152 |  11,167,961 |  1,000 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-5f5fa9086c00a34b3194c93b3531ebc1bcb2f28d.md) | 1,167 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-5f5fa9086c00a34b3194c93b3531ebc1bcb2f28d.md) | 435 |  112,210 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-5f5fa9086c00a34b3194c93b3531ebc1bcb2f28d.md) | 589 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-5f5fa9086c00a34b3194c93b3531ebc1bcb2f28d.md) | 3,855 |  1,979,971 |  860 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5f5fa9086c00a34b3194c93b3531ebc1bcb2f28d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28292482021)

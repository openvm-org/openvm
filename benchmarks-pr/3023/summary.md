| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-521e560d3212a4410037d2c99f32197b2e7b32f5.md) | 406 |  4,000,051 |  227 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-521e560d3212a4410037d2c99f32197b2e7b32f5.md) | 8,454 |  14,365,133 |  1,521 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-521e560d3212a4410037d2c99f32197b2e7b32f5.md) | 4,137 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-521e560d3212a4410037d2c99f32197b2e7b32f5.md) | 494 |  4,090,656 |  189 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-521e560d3212a4410037d2c99f32197b2e7b32f5.md) | 219 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-521e560d3212a4410037d2c99f32197b2e7b32f5.md) | 265 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-521e560d3212a4410037d2c99f32197b2e7b32f5.md) | 1,881 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/521e560d3212a4410037d2c99f32197b2e7b32f5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29423564862)

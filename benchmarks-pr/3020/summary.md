| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-26c8f8c68d8f55d3606e073aca7e515add87ac29.md) | 475 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-26c8f8c68d8f55d3606e073aca7e515add87ac29.md) | 7,334 |  14,365,133 |  1,543 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-26c8f8c68d8f55d3606e073aca7e515add87ac29.md) | 4,708 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-26c8f8c68d8f55d3606e073aca7e515add87ac29.md) | 669 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-26c8f8c68d8f55d3606e073aca7e515add87ac29.md) | 230 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-26c8f8c68d8f55d3606e073aca7e515add87ac29.md) | 274 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-26c8f8c68d8f55d3606e073aca7e515add87ac29.md) | 2,731 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/26c8f8c68d8f55d3606e073aca7e515add87ac29

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30024235942)

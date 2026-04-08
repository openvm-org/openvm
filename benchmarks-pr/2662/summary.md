| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/fibonacci-7b0f7ae40189579aa17f3e750a61af775124c9ce.md) | 3,832 |  12,000,265 |  951 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/keccak-7b0f7ae40189579aa17f3e750a61af775124c9ce.md) | 18,889 |  18,655,329 |  3,395 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/regex-7b0f7ae40189579aa17f3e750a61af775124c9ce.md) | 1,427 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/ecrecover-7b0f7ae40189579aa17f3e750a61af775124c9ce.md) | 642 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/pairing-7b0f7ae40189579aa17f3e750a61af775124c9ce.md) | 904 |  1,745,757 |  292 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/kitchen_sink-7b0f7ae40189579aa17f3e750a61af775124c9ce.md) | 2,158 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7b0f7ae40189579aa17f3e750a61af775124c9ce

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24158154274)

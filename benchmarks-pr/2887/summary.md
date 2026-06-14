| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2887/fibonacci-0924674415f510c6719c5dc2c13658a1186563c8.md) | 1,661 |  4,000,051 |  527 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2887/keccak-0924674415f510c6719c5dc2c13658a1186563c8.md) | 16,369 |  14,365,133 |  3,039 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2887/sha2_bench-0924674415f510c6719c5dc2c13658a1186563c8.md) | 10,404 |  11,167,961 |  1,946 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2887/regex-0924674415f510c6719c5dc2c13658a1186563c8.md) | 1,543 |  4,090,656 |  432 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2887/ecrecover-0924674415f510c6719c5dc2c13658a1186563c8.md) | 482 |  112,210 |  312 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2887/pairing-0924674415f510c6719c5dc2c13658a1186563c8.md) | 627 |  592,827 |  304 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2887/kitchen_sink-0924674415f510c6719c5dc2c13658a1186563c8.md) | 3,934 |  1,979,971 |  867 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0924674415f510c6719c5dc2c13658a1186563c8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27493494516)

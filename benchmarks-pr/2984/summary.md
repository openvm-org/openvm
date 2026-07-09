| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/fibonacci-f02812af21dc89b341db1e3f853751424588df54.md) | 870 |  4,000,051 |  394 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/keccak-f02812af21dc89b341db1e3f853751424588df54.md) | 15,431 |  14,365,133 |  3,033 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/sha2_bench-f02812af21dc89b341db1e3f853751424588df54.md) | 8,127 |  11,167,961 |  1,017 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/regex-f02812af21dc89b341db1e3f853751424588df54.md) | 1,034 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/ecrecover-f02812af21dc89b341db1e3f853751424588df54.md) | 280 |  112,210 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/pairing-f02812af21dc89b341db1e3f853751424588df54.md) | 367 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/kitchen_sink-f02812af21dc89b341db1e3f853751424588df54.md) | 3,819 |  1,979,971 |  859 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f02812af21dc89b341db1e3f853751424588df54

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29026245291)

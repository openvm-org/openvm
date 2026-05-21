| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-8122b5c10b48f7d44327717f7ebddc6c7432ae53.md) | 1,926 |  4,000,051 |  529 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-8122b5c10b48f7d44327717f7ebddc6c7432ae53.md) | 13,540 |  14,365,133 |  2,210 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-8122b5c10b48f7d44327717f7ebddc6c7432ae53.md) | 9,478 |  11,167,961 |  1,415 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-8122b5c10b48f7d44327717f7ebddc6c7432ae53.md) | 1,575 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-8122b5c10b48f7d44327717f7ebddc6c7432ae53.md) | 602 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-8122b5c10b48f7d44327717f7ebddc6c7432ae53.md) | 750 |  592,827 |  259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-8122b5c10b48f7d44327717f7ebddc6c7432ae53.md) | 1,868 |  1,979,971 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8122b5c10b48f7d44327717f7ebddc6c7432ae53

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26251485528)

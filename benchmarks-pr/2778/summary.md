| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-f8c629e7db4568479f3a9407c95c1fcf773a42cb.md) | 1,602 |  4,000,051 |  462 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-f8c629e7db4568479f3a9407c95c1fcf773a42cb.md) | 13,779 |  14,365,133 |  2,396 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-f8c629e7db4568479f3a9407c95c1fcf773a42cb.md) | 9,227 |  11,167,961 |  1,407 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-f8c629e7db4568479f3a9407c95c1fcf773a42cb.md) | 1,500 |  4,090,656 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-f8c629e7db4568479f3a9407c95c1fcf773a42cb.md) | 509 |  112,210 |  290 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-f8c629e7db4568479f3a9407c95c1fcf773a42cb.md) | 623 |  592,827 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-f8c629e7db4568479f3a9407c95c1fcf773a42cb.md) | 1,927 |  1,979,971 |  428 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f8c629e7db4568479f3a9407c95c1fcf773a42cb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25877982036)

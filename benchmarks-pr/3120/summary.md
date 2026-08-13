| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/fibonacci-61e6bf7b203febc6993694951ce11720d633b60a.md) | 450 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/keccak-61e6bf7b203febc6993694951ce11720d633b60a.md) | 7,211 |  14,365,133 |  1,619 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/sha2_bench-61e6bf7b203febc6993694951ce11720d633b60a.md) | 4,115 |  11,167,961 |  514 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/regex-61e6bf7b203febc6993694951ce11720d633b60a.md) | 743 |  4,090,656 |  231 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/ecrecover-61e6bf7b203febc6993694951ce11720d633b60a.md) | 206 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/pairing-61e6bf7b203febc6993694951ce11720d633b60a.md) | 240 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/kitchen_sink-61e6bf7b203febc6993694951ce11720d633b60a.md) | 2,179 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/61e6bf7b203febc6993694951ce11720d633b60a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31754671638)

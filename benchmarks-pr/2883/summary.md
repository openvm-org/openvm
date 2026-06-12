| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/fibonacci-068dbdc9f530e2893d789f0fb26bae3f4bec70f6.md) | 1,609 |  4,000,051 |  527 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/keccak-068dbdc9f530e2893d789f0fb26bae3f4bec70f6.md) | 16,550 |  14,365,133 |  3,009 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/sha2_bench-068dbdc9f530e2893d789f0fb26bae3f4bec70f6.md) | 8,896 |  11,167,961 |  1,148 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/regex-068dbdc9f530e2893d789f0fb26bae3f4bec70f6.md) | 1,603 |  4,090,656 |  428 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/ecrecover-068dbdc9f530e2893d789f0fb26bae3f4bec70f6.md) | 476 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/pairing-068dbdc9f530e2893d789f0fb26bae3f4bec70f6.md) | 620 |  592,827 |  297 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/kitchen_sink-068dbdc9f530e2893d789f0fb26bae3f4bec70f6.md) | 4,015 |  1,979,971 |  867 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/068dbdc9f530e2893d789f0fb26bae3f4bec70f6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27422797595)

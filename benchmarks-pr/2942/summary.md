| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/fibonacci-f94a2b5742f881baa2adac11db1a7635abbac552.md) | 850 |  4,000,051 |  387 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/keccak-f94a2b5742f881baa2adac11db1a7635abbac552.md) | 15,430 |  14,365,133 |  3,038 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/sha2_bench-f94a2b5742f881baa2adac11db1a7635abbac552.md) | 8,093 |  11,167,961 |  1,031 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/regex-f94a2b5742f881baa2adac11db1a7635abbac552.md) | 1,024 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/ecrecover-f94a2b5742f881baa2adac11db1a7635abbac552.md) | 303 |  112,210 |  279 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/pairing-f94a2b5742f881baa2adac11db1a7635abbac552.md) | 450 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/kitchen_sink-f94a2b5742f881baa2adac11db1a7635abbac552.md) | 3,754 |  1,979,971 |  869 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f94a2b5742f881baa2adac11db1a7635abbac552

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28410217836)

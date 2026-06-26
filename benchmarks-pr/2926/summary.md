| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/fibonacci-7ffea93ba404fd2054fb3e5f0e2358ed025e5a06.md) | 1,039 |  4,000,051 |  393 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/keccak-7ffea93ba404fd2054fb3e5f0e2358ed025e5a06.md) | 15,786 |  14,365,133 |  3,035 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/sha2_bench-7ffea93ba404fd2054fb3e5f0e2358ed025e5a06.md) | 8,230 |  11,167,961 |  1,012 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/regex-7ffea93ba404fd2054fb3e5f0e2358ed025e5a06.md) | 1,166 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/ecrecover-7ffea93ba404fd2054fb3e5f0e2358ed025e5a06.md) | 439 |  112,210 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/pairing-7ffea93ba404fd2054fb3e5f0e2358ed025e5a06.md) | 597 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2926/kitchen_sink-7ffea93ba404fd2054fb3e5f0e2358ed025e5a06.md) | 3,904 |  1,979,971 |  871 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7ffea93ba404fd2054fb3e5f0e2358ed025e5a06

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28252522852)

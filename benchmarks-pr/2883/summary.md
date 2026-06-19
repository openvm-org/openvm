| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/fibonacci-1ad89094adaae344a240c2d91bcbd072fdddb7aa.md) | 1,038 |  4,000,051 |  391 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/keccak-1ad89094adaae344a240c2d91bcbd072fdddb7aa.md) | 16,256 |  14,365,133 |  3,020 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/sha2_bench-1ad89094adaae344a240c2d91bcbd072fdddb7aa.md) | 8,108 |  11,167,961 |  993 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/regex-1ad89094adaae344a240c2d91bcbd072fdddb7aa.md) | 1,210 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/ecrecover-1ad89094adaae344a240c2d91bcbd072fdddb7aa.md) | 433 |  112,210 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/pairing-1ad89094adaae344a240c2d91bcbd072fdddb7aa.md) | 593 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/kitchen_sink-1ad89094adaae344a240c2d91bcbd072fdddb7aa.md) | 3,895 |  1,979,971 |  864 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1ad89094adaae344a240c2d91bcbd072fdddb7aa

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27826616571)

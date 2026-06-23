| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/fibonacci-b328605f45fbd756668ceba55574a00140e0b4cf.md) | 1,018 |  4,000,051 |  391 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/keccak-b328605f45fbd756668ceba55574a00140e0b4cf.md) | 16,142 |  14,365,133 |  3,009 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/sha2_bench-b328605f45fbd756668ceba55574a00140e0b4cf.md) | 8,092 |  11,167,961 |  990 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/regex-b328605f45fbd756668ceba55574a00140e0b4cf.md) | 1,189 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/ecrecover-b328605f45fbd756668ceba55574a00140e0b4cf.md) | 437 |  112,210 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/pairing-b328605f45fbd756668ceba55574a00140e0b4cf.md) | 602 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/kitchen_sink-b328605f45fbd756668ceba55574a00140e0b4cf.md) | 3,839 |  1,979,971 |  853 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b328605f45fbd756668ceba55574a00140e0b4cf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28050704701)

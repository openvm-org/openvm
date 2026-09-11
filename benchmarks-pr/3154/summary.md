| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/fibonacci-00d0a72351d92b56f39e8073f828a7a07c802bc5.md) | 475 |  4,000,051 |  236 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/keccak-00d0a72351d92b56f39e8073f828a7a07c802bc5.md) | 7,491 |  14,365,133 |  1,603 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/sha2_bench-00d0a72351d92b56f39e8073f828a7a07c802bc5.md) | 4,290 |  11,167,961 |  536 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/regex-00d0a72351d92b56f39e8073f828a7a07c802bc5.md) | 764 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/ecrecover-00d0a72351d92b56f39e8073f828a7a07c802bc5.md) | 206 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/pairing-00d0a72351d92b56f39e8073f828a7a07c802bc5.md) | 252 |  592,827 |  174 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/kitchen_sink-00d0a72351d92b56f39e8073f828a7a07c802bc5.md) | 2,240 |  1,979,971 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/00d0a72351d92b56f39e8073f828a7a07c802bc5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34605090484)

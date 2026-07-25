| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-1540026efcd7a9ac8570f828a0152c9f84bac95f.md) | 479 |  4,000,051 |  244 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-1540026efcd7a9ac8570f828a0152c9f84bac95f.md) | 10,281 |  14,365,133 |  1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-1540026efcd7a9ac8570f828a0152c9f84bac95f.md) | 4,680 |  11,167,961 |  534 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-1540026efcd7a9ac8570f828a0152c9f84bac95f.md) | 680 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-1540026efcd7a9ac8570f828a0152c9f84bac95f.md) | 232 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-1540026efcd7a9ac8570f828a0152c9f84bac95f.md) | 274 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-1540026efcd7a9ac8570f828a0152c9f84bac95f.md) | 2,383 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1540026efcd7a9ac8570f828a0152c9f84bac95f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30140920681)

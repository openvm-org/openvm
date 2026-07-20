| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/fibonacci-e22f33a51a0a3c6f4a2279ff65c0d7c81cd3050b.md) | 409 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/keccak-e22f33a51a0a3c6f4a2279ff65c0d7c81cd3050b.md) | 8,733 |  14,365,133 |  1,536 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/sha2_bench-e22f33a51a0a3c6f4a2279ff65c0d7c81cd3050b.md) | 4,268 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/regex-e22f33a51a0a3c6f4a2279ff65c0d7c81cd3050b.md) | 581 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/ecrecover-e22f33a51a0a3c6f4a2279ff65c0d7c81cd3050b.md) | 218 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/pairing-e22f33a51a0a3c6f4a2279ff65c0d7c81cd3050b.md) | 288 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/kitchen_sink-e22f33a51a0a3c6f4a2279ff65c0d7c81cd3050b.md) | 1,919 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e22f33a51a0a3c6f4a2279ff65c0d7c81cd3050b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29764248893)
